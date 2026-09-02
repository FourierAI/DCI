from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runner import IMAGE_SUFFIXES, _load_imagenet21k_catalog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a reusable ImageNet-21K {relative_path: WNID} index."
    )
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--label-file",
        type=Path,
        default=Path("data/metadata/imagenet21k/im21K.txt"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    by_wnid, _ = _load_imagenet21k_catalog(args.label_file)
    index: dict[str, str] = {}
    for path in args.image_root.rglob("*"):
        if not path.is_file() or path.suffix.casefold() not in IMAGE_SUFFIXES:
            continue
        relative = path.relative_to(args.image_root)
        wnid = next((part for part in relative.parts if part in by_wnid), None)
        if wnid is not None:
            index[relative.as_posix()] = wnid
    if not index:
        raise FileNotFoundError(
            "No images under recognized ImageNet-21K WNID directories."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Indexed {len(index)} images to {args.output}")


if __name__ == "__main__":
    main()
