from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetConfig:
    metadata: str
    split: str
    default_image_root: str
    default_k_values: tuple[int, ...]
    label_file: str | None = None


DATASETS: dict[str, DatasetConfig] = {
    "cifar100": DatasetConfig(
        metadata="data/metadata/cifar100/labels.json",
        split="mapping",
        default_image_root="data/images/cifar100",
        default_k_values=(100, 50, 20, 10, 5, 2),
    ),
    "cub200": DatasetConfig(
        metadata="data/metadata/cub200/split_TAI_CUB-200-2011.json",
        split="train",
        default_image_root="data/images/cub200",
        default_k_values=(200, 100, 50, 20, 10, 5, 2),
    ),
    "caltech256": DatasetConfig(
        metadata="data/metadata/caltech256/split_TAI_Caltech256.json",
        split="train",
        default_image_root="data/images/caltech256",
        default_k_values=(257, 100, 50, 20, 10, 5, 2),
    ),
    "food101": DatasetConfig(
        metadata="data/metadata/food101/split_zhou_Food101.json",
        split="train",
        default_image_root="data/images/food101",
        default_k_values=(101, 50, 20, 10, 5, 2),
    ),
    "imagenet1k": DatasetConfig(
        metadata="data/metadata/imagenet1k/split_TAI_imagenet_val.json",
        split="val",
        default_image_root="data/images/imagenet1k",
        default_k_values=(1000, 100, 50, 20, 10, 5, 2),
    ),
    "imagenet21k": DatasetConfig(
        metadata="data/metadata/imagenet1k/split_TAI_imagenet_val.json",
        split="val",
        default_image_root="data/images/imagenet1k",
        default_k_values=(21843, 1000, 200, 100, 50),
        label_file="data/metadata/imagenet21k/im21K.txt",
    ),
}


def resolve_path(repo_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else repo_root / path
