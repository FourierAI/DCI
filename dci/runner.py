from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import APIError, OpenAI
from tqdm import tqdm

from .configs import DATASETS, DatasetConfig, resolve_path
from .prompts import DCI_PROMPT, FLAT_PROMPT, build_prompt
from .validation import VALIDATION_VERSION, validate_prediction

# Compatibility alias for callers that previously imported the DCI template.
PROMPT = DCI_PROMPT

IMAGE_SUFFIXES = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass(frozen=True)
class LoadedDataset:
    image_root: Path
    image_to_label: dict[str, str]
    labels: list[str]
    catalog_sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the paper-aligned Divide-and-Conquer Inference protocol with "
            "an OpenAI-compatible MLLM server."
        )
    )
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument(
        "--model", required=True, help="Model name exposed by the API server."
    )
    parser.add_argument(
        "--api-base",
        default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
    )
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument(
        "--image-root", help="Dataset image root; overrides the default path."
    )
    parser.add_argument(
        "--metadata",
        help=(
            "Optional JSON image index. Required only when ImageNet-21K should "
            "not be indexed directly from --image-root."
        ),
    )
    parser.add_argument(
        "--b-values",
        "--k-values",
        dest="b_values",
        nargs="+",
        type=int,
        help="Maximum local group sizes B. --k-values is retained as an alias.",
    )
    parser.add_argument(
        "--candidate-counts",
        nargs="+",
        type=int,
        help="Candidate-set sizes N; defaults to the complete vocabulary.",
    )
    parser.add_argument(
        "--runs", type=int, default=5, help="Independent runs; the paper uses 5."
    )
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--samples-per-class", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument(
        "--baseline", action="store_true", help="Use one monolithic prompt."
    )
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help=(
            "Retries for API failures; 0 (the paper default) retries until a "
            "response is obtained."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Optional response-token cap; omitted to retain the model setting.",
    )
    parser.add_argument(
        "--temperature", type=float, help="Omit to retain the serving default."
    )
    parser.add_argument(
        "--top-p", type=float, help="Omit to retain the serving default."
    )
    parser.add_argument(
        "--sd-ddof",
        type=int,
        choices=(0, 1),
        default=0,
        help="Run-level SD denominator: runs-ddof; recorded in summary.json.",
    )
    parser.add_argument(
        "--save-traces",
        action="store_true",
        help="Include group candidates, raw responses and validation outcomes in JSONL.",
    )
    return parser.parse_args()


def _sha256_text(values: list[str]) -> str:
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validated_dataset(
    config: DatasetConfig,
    image_root: Path,
    image_to_label: dict[str, str],
    labels: list[str],
) -> LoadedDataset:
    if len(labels) != config.expected_labels:
        raise ValueError(
            f"Expected {config.expected_labels} unique candidate labels, "
            f"found {len(labels)}."
        )
    if (
        config.expected_images is not None
        and len(image_to_label) != config.expected_images
    ):
        raise ValueError(
            f"Expected {config.expected_images} evaluation images, "
            f"found {len(image_to_label)}."
        )
    unknown = sorted(set(image_to_label.values()) - set(labels))
    if unknown:
        raise ValueError(
            f"Image index contains labels absent from the catalog: {unknown[:5]}"
        )
    return LoadedDataset(image_root, image_to_label, labels, _sha256_text(labels))


def _load_mapping(path: Path) -> tuple[dict[str, str], list[str]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    image_to_label = {
        image: item["class_name"] if isinstance(item, dict) else str(item)
        for image, item in payload.items()
    }
    return image_to_label, sorted(set(image_to_label.values()))


def _load_json_split(path: Path, split: str) -> tuple[dict[str, str], list[str]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload[split]
    id_to_name: dict[str, str] = {}
    for row in rows:
        class_id, name = str(row[1]), str(row[-1])
        previous = id_to_name.setdefault(class_id, name)
        if previous != name:
            raise ValueError(
                f"Class {class_id} has conflicting names: {previous!r}, {name!r}"
            )

    name_counts: dict[str, int] = {}
    for name in id_to_name.values():
        name_counts[name] = name_counts.get(name, 0) + 1
    display = {
        class_id: name if name_counts[name] == 1 else f"{name} [class {class_id}]"
        for class_id, name in id_to_name.items()
    }
    labels = [
        display[class_id] for class_id in sorted(display, key=lambda value: int(value))
    ]
    image_to_label = {str(row[0]): display[str(row[1])] for row in rows}
    return image_to_label, labels


def _find_dataset_root(image_root: Path, marker: str) -> Path:
    for candidate in (image_root, image_root.parent):
        if (candidate / marker).is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find {marker} beside {image_root}. Use the official dataset layout."
    )


def _read_index(path: Path) -> dict[int, str]:
    result: dict[int, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            key, value = line.rstrip("\n").split(maxsplit=1)
            result[int(key)] = value
    return result


def _load_cub_official(
    image_root: Path,
) -> tuple[Path, dict[str, str], list[str]]:
    root = _find_dataset_root(image_root, "train_test_split.txt")
    actual_image_root = root / "images"
    images = _read_index(root / "images.txt")
    class_ids = {
        key: int(value)
        for key, value in _read_index(root / "image_class_labels.txt").items()
    }
    split = {
        key: int(value)
        for key, value in _read_index(root / "train_test_split.txt").items()
    }
    classes = _read_index(root / "classes.txt")
    image_to_label = {
        images[image_id]: classes[class_ids[image_id]].split(".", maxsplit=1)[-1]
        for image_id in sorted(images)
        if split[image_id] == 0
    }
    labels = [
        classes[class_id].split(".", maxsplit=1)[-1] for class_id in sorted(classes)
    ]
    return actual_image_root, image_to_label, labels


def _load_food101_official(
    image_root: Path,
) -> tuple[Path, dict[str, str], list[str]]:
    for root in (image_root, image_root.parent):
        test_file = root / "meta" / "test.txt"
        if test_file.is_file():
            break
    else:
        raise FileNotFoundError(
            f"Could not find meta/test.txt beside {image_root}. "
            "Use the official Food-101 layout."
        )
    actual_image_root = root / "images"
    entries = [
        line.strip()
        for line in test_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    image_to_label = {
        f"{entry}.jpg": entry.split("/", maxsplit=1)[0] for entry in entries
    }
    labels = sorted(set(image_to_label.values()))
    return actual_image_root, image_to_label, labels


def _load_imagenet21k_catalog(path: Path) -> tuple[dict[str, str], list[str]]:
    rows: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        wnid, description = line.split(",", maxsplit=1)
        rows.append((wnid.strip(), description.strip()))
    counts: dict[str, int] = {}
    for _, description in rows:
        key = description.casefold()
        counts[key] = counts.get(key, 0) + 1
    by_wnid = {
        wnid: (
            description
            if counts[description.casefold()] == 1
            else f"{description} [{wnid}]"
        )
        for wnid, description in rows
    }
    return by_wnid, [by_wnid[wnid] for wnid, _ in rows]


def _load_imagenet21k_index(
    image_root: Path,
    metadata_path: Path | None,
    by_wnid: dict[str, str],
) -> dict[str, str]:
    if metadata_path is not None:
        with metadata_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        image_to_label: dict[str, str] = {}
        for image, value in payload.items():
            wnid = value.get("wnid") if isinstance(value, dict) else str(value)
            if wnid not in by_wnid:
                raise ValueError(f"Unknown ImageNet-21K WNID {wnid!r} for {image!r}")
            image_to_label[image] = by_wnid[wnid]
        return image_to_label

    image_to_label = {}
    for path in image_root.rglob("*"):
        if not path.is_file() or path.suffix.casefold() not in IMAGE_SUFFIXES:
            continue
        relative = path.relative_to(image_root)
        wnid = next((part for part in relative.parts if part in by_wnid), None)
        if wnid is not None:
            image_to_label[relative.as_posix()] = by_wnid[wnid]
    if not image_to_label:
        raise FileNotFoundError(
            "No ImageNet-21K images were indexed. Arrange images under WNID "
            "directories or pass --metadata with a {relative_path: wnid} JSON index."
        )
    return image_to_label


def load_dataset(
    repo_root: Path,
    config: DatasetConfig,
    metadata_override: str | None,
    image_root_override: str | None,
) -> LoadedDataset:
    image_root = resolve_path(
        repo_root, image_root_override or config.default_image_root
    )
    metadata_path = (
        resolve_path(repo_root, metadata_override) if metadata_override else None
    )

    if metadata_override and config.loader != "imagenet21k":
        image_to_label, labels = _load_mapping(metadata_path)
    elif config.loader == "mapping":
        image_to_label, labels = _load_mapping(
            resolve_path(repo_root, config.metadata or "")
        )
    elif config.loader == "json_split":
        image_to_label, labels = _load_json_split(
            resolve_path(repo_root, config.metadata or ""), config.split or ""
        )
    elif config.loader == "cub_official":
        image_root, image_to_label, labels = _load_cub_official(image_root)
    elif config.loader == "food101_official":
        image_root, image_to_label, labels = _load_food101_official(image_root)
    elif config.loader == "imagenet21k":
        by_wnid, labels = _load_imagenet21k_catalog(
            resolve_path(repo_root, config.label_file or "")
        )
        image_to_label = _load_imagenet21k_index(image_root, metadata_path, by_wnid)
    else:
        raise ValueError(f"Unsupported dataset loader: {config.loader}")

    return _validated_dataset(config, image_root, image_to_label, labels)


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
            images.extend(
                rng.sample(candidates, min(samples_per_class, len(candidates)))
            )

    if max_samples is not None and len(images) > max_samples:
        images = rng.sample(images, max_samples)
    return images


def select_candidate_problem(
    image_to_label: dict[str, str],
    labels: list[str],
    candidate_count: int,
    samples_per_class: int | None,
    max_samples: int | None,
    rng: random.Random,
) -> tuple[dict[str, str], list[str], list[str]]:
    if not 2 <= candidate_count <= len(labels):
        raise ValueError(f"Candidate count must be between 2 and {len(labels)}.")
    if candidate_count == len(labels):
        selected_labels = labels[:]
    else:
        available = set(image_to_label.values())
        eligible = [label for label in labels if label in available]
        if candidate_count > len(eligible):
            raise ValueError(
                f"Only {len(eligible)} candidate labels have indexed evaluation "
                f"images; cannot sample N={candidate_count}."
            )
        selected = set(rng.sample(eligible, candidate_count))
        selected_labels = [label for label in labels if label in selected]
    selected_set = set(selected_labels)
    selected_mapping = {
        image: label for image, label in image_to_label.items() if label in selected_set
    }
    images = sample_images(selected_mapping, samples_per_class, max_samples, rng)
    return selected_mapping, selected_labels, images


class DCIClassifier:
    def __init__(
        self,
        client: OpenAI,
        model: str,
        max_workers: int,
        max_tokens: int | None,
        seed: int = 0,
        max_retries: int = 5,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.max_workers = max_workers
        self.max_tokens = max_tokens
        self.seed = seed
        self.max_retries = max_retries
        self.temperature = temperature
        self.top_p = top_p
        self.last_trace: list[dict[str, Any]] = []

    @staticmethod
    def encode_image(image_path: Path) -> str:
        mime_type = (
            mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
        )
        with image_path.open("rb") as handle:
            encoded = base64.b64encode(handle.read()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    def query(
        self, image_data_url: str, labels: list[str], *, flat: bool = False
    ) -> str:
        prompt = build_prompt(labels, flat=flat)
        retries = 0
        while True:
            try:
                request: dict[str, Any] = {
                    "model": self.model,
                    "messages": [
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
                }
                if self.max_tokens is not None:
                    request["max_tokens"] = self.max_tokens
                if self.temperature is not None:
                    request["temperature"] = self.temperature
                if self.top_p is not None:
                    request["top_p"] = self.top_p
                response = self.client.chat.completions.create(**request)
                return (response.choices[0].message.content or "").strip()
            except APIError:
                retries += 1
                if self.max_retries > 0 and retries > self.max_retries:
                    raise
                time.sleep(min(2 ** min(retries - 1, 5), 30))

    @staticmethod
    def groups(
        labels: list[str], size: int, rng: random.Random | None = None
    ) -> list[list[str]]:
        shuffled = labels[:]
        if rng is not None:
            rng.shuffle(shuffled)
        return [
            shuffled[index : index + size] for index in range(0, len(shuffled), size)
        ]

    @staticmethod
    def normalize_prediction(prediction: str, candidates: list[str]) -> str | None:
        return validate_prediction(prediction, candidates).label

    def _record_level(self, groups: list[list[str]], raw: list[str]) -> list[str]:
        outcomes = [
            validate_prediction(response, group) for group, response in zip(groups, raw)
        ]
        survivors = [item.label for item in outcomes if item.label is not None]
        self.last_trace.append(
            {
                "level": len(self.last_trace) + 1,
                "input_count": sum(map(len, groups)),
                "output_count": len(survivors),
                "groups": [
                    {
                        "candidates": group,
                        "response": response,
                        "prediction": item.label,
                        "status": item.status,
                    }
                    for group, response, item in zip(groups, raw, outcomes)
                ],
            }
        )
        return survivors

    def classify_flat(self, image_path: Path, labels: list[str]) -> str | None:
        self.last_trace = []
        if not labels:
            raise ValueError("Flat inference requires a nonempty candidate list.")
        raw = self.query(self.encode_image(image_path), labels, flat=True)
        survivors = self._record_level([labels], [raw])
        return survivors[0] if survivors else None

    def _image_rng(self, image_path: Path) -> random.Random:
        payload = f"{self.seed}\0{image_path.as_posix()}".encode()
        digest = hashlib.sha256(payload).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))

    def classify(self, image_path: Path, labels: list[str], b: int) -> str | None:
        if b < 2:
            raise ValueError("B must be at least 2.")
        self.last_trace = []
        if len(labels) <= 1:
            return labels[0] if labels else None
        image_data_url = self.encode_image(image_path)
        active = labels[:]
        rng = self._image_rng(image_path)

        while True:
            if not active:
                return None
            if len(active) == 1:
                return active[0]
            if len(active) <= b:
                survivors = self._record_level(
                    [active], [self.query(image_data_url, active)]
                )
                return survivors[0] if survivors else None

            groups = self.groups(active, b, rng)
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                raw = list(
                    executor.map(
                        lambda group: self.query(image_data_url, group), groups
                    )
                )
            active = self._record_level(groups, raw)


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
                    f"Warning: ignoring malformed record at {path}:"
                    f"{line_number}: {exc}",
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
    run_index: int,
    run_seed: int,
    b: int,
    mode: str,
    images: list[str],
    selected_labels: list[str],
    full_catalog_sha256: str,
) -> None:
    arguments = vars(args).copy()
    arguments.pop("api_key", None)
    payload: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(repo_root),
        "python": sys.version,
        "platform": platform.platform(),
        "mode": mode,
        "run_index": run_index,
        "run_seed": run_seed,
        "B": b,
        "num_images": len(images),
        "num_labels": len(selected_labels),
        "evaluation_images": images,
        "candidate_labels": selected_labels,
        "full_catalog_sha256": full_catalog_sha256,
        "prompt_sha256": hashlib.sha256(
            (FLAT_PROMPT if mode == "baseline" else DCI_PROMPT).encode("utf-8")
        ).hexdigest(),
        "validation_version": VALIDATION_VERSION,
        "decoding": {
            key: getattr(args, key, None)
            for key in ("max_tokens", "temperature", "top_p")
        },
        "arguments": arguments,
    }
    records_path = path.with_name(path.name.removesuffix(".manifest.json") + ".jsonl")
    if records_path.exists() and records_path.stat().st_size:
        if not path.exists():
            raise ValueError(
                "Existing predictions have no manifest; use a new --output-dir."
            )
        previous = json.loads(path.read_text(encoding="utf-8"))
        keys = (
            "mode",
            "run_index",
            "run_seed",
            "B",
            "evaluation_images",
            "candidate_labels",
            "full_catalog_sha256",
            "prompt_sha256",
            "validation_version",
            "decoding",
        )
        changed = [key for key in keys if previous.get(key) != payload[key]]
        for key in ("model", "api_base", "max_workers", "save_traces"):
            if previous.get("arguments", {}).get(key) != arguments.get(key):
                changed.append(key)
        if changed:
            raise ValueError(
                f"Cannot mix existing predictions with changed settings ({', '.join(changed)}); use a new --output-dir."
            )
        return
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def evaluate(path: Path, elapsed: float, new_samples: int) -> dict[str, float | int | None]:
    total = correct = 0
    image_latencies = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            correct += row["prediction"] == row["target"]
            if "latency_seconds" in row:
                image_latencies.append(row["latency_seconds"])
    accuracy = 100 * correct / total if total else 0.0
    seconds_per_new_image = elapsed / new_samples if new_samples else 0.0
    report = (
        f"Total samples: {total}\n"
        f"Correct predictions: {correct}\n"
        f"Accuracy: {accuracy:.4f}%\n"
        f"New samples this invocation: {new_samples}\n"
        f"Elapsed time this invocation: {elapsed:.2f}s\n"
        f"Time per new image: {seconds_per_new_image:.4f}s\n"
    )
    path.with_suffix(".txt").write_text(report, encoding="utf-8")
    print(report)
    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "elapsed_seconds": elapsed,
        "new_samples": new_samples,
        "seconds_per_new_image": seconds_per_new_image,
        "latency_mean": statistics.mean(image_latencies)
        if image_latencies and len(image_latencies) == total
        else None,
    }


def write_summary(
    path: Path, results: dict[str, list[dict[str, float | int | None]]], *, ddof: int = 0
) -> None:
    if ddof not in (0, 1):
        raise ValueError("ddof must be 0 or 1.")

    def sd(values):
        if len(values) <= ddof:
            return None
        return statistics.stdev(values) if ddof else statistics.pstdev(values)

    settings: dict[str, Any] = {}
    for name, runs in results.items():
        accuracies = [float(run["accuracy"]) for run in runs]
        latencies = [
            float(run["latency_mean"])
            for run in runs
            if run.get("latency_mean") is not None
        ]
        settings[name] = {
            "runs": runs,
            "accuracy_mean": statistics.mean(accuracies),
            "accuracy_sd": sd(accuracies),
            "latency_mean": statistics.mean(latencies) if latencies else None,
            "latency_sd": sd(latencies),
            "latency_run_count": len(latencies),
            "sd_ddof": ddof,
        }
    path.write_text(json.dumps({"settings": settings}, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be at least 1.")
    if args.max_retries < 0:
        raise ValueError("--max-retries cannot be negative.")
    if args.max_workers < 1:
        raise ValueError("--max-workers must be at least 1.")
    if args.top_p is not None and not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1].")
    if args.temperature is not None and args.temperature < 0:
        raise ValueError("--temperature cannot be negative.")

    repo_root = Path(__file__).resolve().parents[1]
    config = DATASETS[args.dataset]
    dataset = load_dataset(repo_root, config, args.metadata, args.image_root)
    if not dataset.image_root.exists():
        raise FileNotFoundError(
            f"Image root not found: {dataset.image_root}. See data/README.md for setup."
        )

    client = OpenAI(
        api_key=args.api_key,
        base_url=args.api_base,
        timeout=args.timeout,
        max_retries=0,
    )
    candidate_counts = args.candidate_counts or [len(dataset.labels)]
    b_values = args.b_values or list(config.default_b_values)
    if any(b < 2 for b in b_values):
        raise ValueError("Every B value must be at least 2.")

    max_samples = args.max_samples
    if max_samples is None:
        max_samples = config.default_max_samples

    output_root = Path(args.output_dir) / args.dataset / args.model.replace("/", "--")
    output_root.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, list[dict[str, float | int | None]]] = {}

    for run_index in range(1, args.runs + 1):
        run_seed = args.seed + run_index - 1
        for candidate_count in candidate_counts:
            problem_rng = random.Random(f"{run_seed}:{candidate_count}")
            image_to_label, selected_labels, images = select_candidate_problem(
                dataset.image_to_label,
                dataset.labels,
                candidate_count,
                args.samples_per_class,
                max_samples,
                problem_rng,
            )
            settings = (
                [("baseline", candidate_count)]
                if args.baseline
                else [(f"b-{b}", b) for b in b_values]
            )
            for mode, b in settings:
                run_root = output_root / f"n-{candidate_count}" / f"run-{run_index:02d}"
                run_root.mkdir(parents=True, exist_ok=True)
                output_path = run_root / f"{mode}.jsonl"
                completed = read_completed(output_path)
                pending = [image for image in images if image not in completed]
                write_manifest(
                    run_root / f"{mode}.manifest.json",
                    args,
                    repo_root,
                    run_index=run_index,
                    run_seed=run_seed,
                    b=b,
                    mode=mode,
                    images=images,
                    selected_labels=selected_labels,
                    full_catalog_sha256=dataset.catalog_sha256,
                )
                classifier = DCIClassifier(
                    client,
                    args.model,
                    args.max_workers,
                    args.max_tokens,
                    seed=run_seed,
                    max_retries=args.max_retries,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                start = time.perf_counter()
                with output_path.open("a", encoding="utf-8") as handle:
                    for image in tqdm(
                        pending,
                        desc=(
                            f"{args.dataset} / N={candidate_count} / "
                            f"run={run_index} / {mode}"
                        ),
                    ):
                        image_path = dataset.image_root / image
                        if not image_path.is_file():
                            raise FileNotFoundError(f"Image not found: {image_path}")
                        image_start = time.perf_counter()
                        prediction = (
                            classifier.classify_flat(image_path, selected_labels)
                            if args.baseline
                            else classifier.classify(image_path, selected_labels, b)
                        )
                        image_latency = time.perf_counter() - image_start
                        row = {
                            "image": image,
                            "prediction": prediction,
                            "target": image_to_label[image],
                            "N": candidate_count,
                            "B": b,
                            "mode": mode,
                            "run": run_index,
                            "seed": run_seed,
                            "latency_seconds": image_latency,
                            "call_count": sum(
                                len(level["groups"]) for level in classifier.last_trace
                            ),
                        }
                        if args.save_traces:
                            row["trace"] = classifier.last_trace
                        handle.write(
                            json.dumps(
                                row,
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        handle.flush()
                metrics = evaluate(
                    output_path,
                    time.perf_counter() - start,
                    len(pending),
                )
                key = f"n-{candidate_count}/{mode}"
                all_results.setdefault(key, []).append(
                    {"run": run_index, "seed": run_seed, **metrics}
                )

    write_summary(output_root / "summary.json", all_results, ddof=args.sd_ddof)


if __name__ == "__main__":
    main()
