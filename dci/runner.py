from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import platform
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm

from .configs import DATASETS, DatasetConfig, resolve_path


PROMPT = """Please directly identify the category name of the image from the
candidate category name list. {decision_rule} Do not output sentences or
explanations; output only one category name exactly as listed (respect case and
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
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum retries for transient API failures.",
    )
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
        mime_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
        with image_path.open("rb") as handle:
            encoded = base64.b64encode(handle.read()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    def query(self, image_data_url: str, labels: list[str], allow_none: bool) -> str:
        decision_rule = (
            "Always choose the most likely category from the list whenever possible, "
            "and only output 'None' if the image clearly does not belong to any "
            "category in the list."
            if allow_none
            else "Always choose the most likely category from the list."
        )
        prompt = PROMPT.format(
            decision_rule=decision_rule,
            count=len(labels),
            labels=", ".join(labels),
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_url},
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
        image_data_url = self.encode_image(image_path)
        active = labels[:]
        while len(active) > k:
            groups = self.groups(active, k)
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                raw = list(
                    executor.map(
                        lambda group: self.query(
                            image_data_url, group, allow_none=True
                        ),
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
                return self.query(image_data_url, active, allow_none=False)
            active = list(dict.fromkeys(survivors))

        prediction = self.query(image_data_url, active, allow_none=False)
        return self.normalize_prediction(prediction, active) or prediction


def read_completed(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                completed.add(json.loads(line)["image"])
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                print(
                    f"Warning: ignoring malformed record at {path}:{line_number}: {exc}",
                    file=sys.stderr,
                )
    return completed


def git_revision(repo_root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    repo_root: Path,
    *,
    k: int,
    mode: str,
    num_images: int,
    num_labels: int,
) -> None:
    arguments = vars(args).copy()
    arguments.pop("api_key", None)
    payload: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(repo_root),
        "python": sys.version,
        "platform": platform.platform(),
        "mode": mode,
        "k": k,
        "num_images": num_images,
        "num_labels": num_labels,
        "arguments": arguments,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def evaluate(path: Path, elapsed: float, new_samples: int) -> None:
    total = correct = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            correct += row["prediction"] == row["target"]
    accuracy = 100 * correct / total if total else 0.0
    report = (
        f"Total samples: {total}\n"
        f"Correct predictions: {correct}\n"
        f"Accuracy: {accuracy:.2f}%\n"
        f"New samples this run: {new_samples}\n"
        f"Elapsed time this run: {elapsed:.2f}s\n"
        f"Time per new image: {elapsed / new_samples if new_samples else 0:.2f}s\n"
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

    client = OpenAI(
        api_key=args.api_key,
        base_url=args.api_base,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
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
        completed = read_completed(output_path)
        pending = [image for image in images if image not in completed]
        write_manifest(
            output_root / f"{mode}.manifest.json",
            args,
            repo_root,
            k=k,
            mode=mode,
            num_images=len(images),
            num_labels=len(labels),
        )
        start = time.perf_counter()
        with output_path.open("a", encoding="utf-8") as handle:
            for image in tqdm(pending, desc=f"{args.dataset} / {mode}"):
                image_path = image_root / image
                if not image_path.is_file():
                    raise FileNotFoundError(f"Image not found: {image_path}")
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
        evaluate(output_path, time.perf_counter() - start, len(pending))


if __name__ == "__main__":
    main()
