from __future__ import annotations

import argparse
import base64
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from .configs import DATASETS, DatasetConfig, resolve_path


PROMPT = """Please directly identify the category name of the image from the
candidate category name list. Always choose the most likely category from the
list whenever possible, and only output 'None' if the image clearly does not
belong to any category in the list. Do not output sentences or explanations;
output only one category name exactly as listed (respect case and
singular/plural form).

Example:
Q: There are 3 categories, [Leopards, wild_cat, electric_guitar].
Which category does the image belong to?
A: wild_cat

Now answer:
Q: There are {count} categories, [{labels}].
Which category does the image belong to?
A:"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Divide-and-Conquer Inference with an OpenAI-compatible MLLM server."
    )
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--model", required=True, help="Model name exposed by the API server.")
    parser.add_argument(
        "--api-base",
        default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
    )
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--image-root", help="Dataset image root; overrides the default path.")
    parser.add_argument("--metadata", help="Metadata JSON path; overrides the bundled path.")
    parser.add_argument("--k-values", nargs="+", type=int)
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--samples-per-class", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--baseline", action="store_true", help="Use one flat prompt.")
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--max-tokens", type=int, default=64)
    return parser.parse_args()


def load_dataset(
    repo_root: Path,
    config: DatasetConfig,
    metadata_override: str | None,
) -> tuple[dict[str, str], list[str]]:
    metadata_path = resolve_path(repo_root, metadata_override or config.metadata)
    with metadata_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    if config.split == "mapping":
        image_to_label = {
            image: item["class_name"] for image, item in payload.items()
        }
    else:
        image_to_label = {
            item[0]: item[-1] for item in payload[config.split]
        }

    if config.label_file:
        label_path = resolve_path(repo_root, config.label_file)
        with label_path.open(encoding="utf-8") as handle:
            labels = [
                line.rstrip("\n").split(",", maxsplit=2)[1].strip()
                for line in handle
                if "," in line
            ]
    else:
        labels = sorted(set(image_to_label.values()))

    return image_to_label, list(dict.fromkeys(labels))


def sample_images(
    image_to_label: dict[str, str],
    samples_per_class: int | None,
    max_samples: int | None,
    rng: random.Random,
) -> list[str]:
    if samples_per_class is None:
        images = sorted(image_to_label)
    else:
        grouped: dict[str, list[str]] = {}
        for image, label in image_to_label.items():
            grouped.setdefault(label, []).append(image)
        images = []
        for label in sorted(grouped):
            candidates = sorted(grouped[label])
            images.extend(rng.sample(candidates, min(samples_per_class, len(candidates))))

    if max_samples is not None and len(images) > max_samples:
        images = rng.sample(images, max_samples)
    return images


class DCIClassifier:
    def __init__(
        self,
        client: OpenAI,
        model: str,
        max_workers: int,
        max_tokens: int,
    ) -> None:
        self.client = client
        self.model = model
        self.max_workers = max_workers
        self.max_tokens = max_tokens

    @staticmethod
    def encode_image(image_path: Path) -> str:
        with image_path.open("rb") as handle:
            return base64.b64encode(handle.read()).decode("utf-8")

    def query(self, image_path: Path, labels: list[str], allow_none: bool) -> str:
        prompt = PROMPT.format(count=len(labels), labels=", ".join(labels))
        if not allow_none:
            prompt = prompt.replace(
                "Always choose the most likely category from the\nlist whenever possible, and only output 'None' if the image clearly does not\nbelong to any category in the list.",
                "Always choose the most likely category from the list.",
            )
        image = self.encode_image(image_path)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_tokens=self.max_tokens,
        )
        return (response.choices[0].message.content or "").strip()

    @staticmethod
    def groups(labels: list[str], size: int) -> list[list[str]]:
        return [labels[index : index + size] for index in range(0, len(labels), size)]

    @staticmethod
    def normalize_prediction(prediction: str, candidates: list[str]) -> str | None:
        cleaned = prediction.strip().strip("`\"'. ")
        if cleaned.lower() == "none":
            return None
        exact = {label.casefold(): label for label in candidates}
        return exact.get(cleaned.casefold())

    def classify(self, image_path: Path, labels: list[str], k: int) -> str:
        active = labels[:]
        while len(active) > k:
            groups = self.groups(active, k)
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                raw = list(
                    executor.map(
                        lambda group: self.query(image_path, group, allow_none=True),
                        groups,
                    )
                )
            survivors = [
                normalized
                for prediction, group in zip(raw, groups)
                if (normalized := self.normalize_prediction(prediction, group))
            ]
            if not survivors:
                # A forced global decision guarantees progress if every branch rejects.
                return self.query(image_path, active, allow_none=False)
            active = list(dict.fromkeys(survivors))

        prediction = self.query(image_path, active, allow_none=False)
        return self.normalize_prediction(prediction, active) or prediction


def evaluate(path: Path, elapsed: float) -> None:
    total = correct = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            total += 1
            correct += row["prediction"] == row["target"]
    accuracy = 100 * correct / total if total else 0.0
    report = (
        f"Total samples: {total}\n"
        f"Correct predictions: {correct}\n"
        f"Accuracy: {accuracy:.2f}%\n"
        f"Elapsed time: {elapsed:.2f}s\n"
        f"Time per image: {elapsed / total if total else 0:.2f}s\n"
    )
    path.with_suffix(".txt").write_text(report, encoding="utf-8")
    print(report)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config = DATASETS[args.dataset]
    image_to_label, labels = load_dataset(repo_root, config, args.metadata)
    image_root = resolve_path(
        repo_root, args.image_root or config.default_image_root
    )
    if not image_root.exists():
        raise FileNotFoundError(
            f"Image root not found: {image_root}. See README.md for dataset setup."
        )

    client = OpenAI(api_key=args.api_key, base_url=args.api_base, timeout=args.timeout)
    classifier = DCIClassifier(client, args.model, args.max_workers, args.max_tokens)
    rng = random.Random(args.seed)
    images = sample_images(
        image_to_label, args.samples_per_class, args.max_samples, rng
    )
    k_values = args.k_values or list(config.default_k_values)
    if args.baseline:
        k_values = [len(labels)]
    if any(k < 2 for k in k_values):
        raise ValueError("Every K value must be at least 2.")

    output_root = Path(args.output_dir) / args.dataset / args.model.replace("/", "--")
    output_root.mkdir(parents=True, exist_ok=True)

    for k in k_values:
        mode = "baseline" if args.baseline else f"k-{k}"
        output_path = output_root / f"{mode}.jsonl"
        completed: set[str] = set()
        if output_path.exists():
            with output_path.open(encoding="utf-8") as handle:
                completed = {json.loads(line)["image"] for line in handle}

        pending = [image for image in images if image not in completed]
        start = time.perf_counter()
        with output_path.open("a", encoding="utf-8") as handle:
            for image in tqdm(pending, desc=f"{args.dataset} / {mode}"):
                image_path = image_root / image
                prediction = classifier.classify(image_path, labels, k)
                handle.write(
                    json.dumps(
                        {
                            "image": image,
                            "prediction": prediction,
                            "target": image_to_label[image],
                            "k": k,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                handle.flush()
        evaluate(output_path, time.perf_counter() - start)


if __name__ == "__main__":
    main()
