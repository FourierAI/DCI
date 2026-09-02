from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetConfig:
    loader: str
    default_image_root: str
    default_b_values: tuple[int, ...]
    expected_images: int | None
    expected_labels: int
    metadata: str | None = None
    split: str | None = None
    label_file: str | None = None
    default_max_samples: int | None = None


DATASETS: dict[str, DatasetConfig] = {
    "cifar100": DatasetConfig(
        loader="mapping",
        metadata="data/metadata/cifar100/labels.json",
        split="mapping",
        default_image_root="data/images/cifar100",
        default_b_values=(10,),
        expected_images=10_000,
        expected_labels=100,
    ),
    "cub200": DatasetConfig(
        loader="cub_official",
        default_image_root="data/images/CUB_200_2011/images",
        default_b_values=(10,),
        expected_images=5_794,
        expected_labels=200,
    ),
    "food101": DatasetConfig(
        loader="food101_official",
        default_image_root="data/images/food-101/images",
        default_b_values=(10,),
        expected_images=25_250,
        expected_labels=101,
    ),
    "imagenet1k": DatasetConfig(
        loader="json_split",
        metadata="data/metadata/imagenet1k/split_TAI_imagenet_val.json",
        split="val",
        default_image_root="data/images/imagenet1k",
        default_b_values=(10,),
        expected_images=50_000,
        expected_labels=1_000,
    ),
    "imagenet21k": DatasetConfig(
        loader="imagenet21k",
        default_image_root="data/images/imagenet21k",
        default_b_values=(100,),
        expected_images=None,
        expected_labels=21_843,
        label_file="data/metadata/imagenet21k/im21K.txt",
        default_max_samples=1_000,
    ),
}


def resolve_path(repo_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else repo_root / path
