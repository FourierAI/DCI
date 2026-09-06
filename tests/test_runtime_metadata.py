"""Offline checks for software runtime configuration and input catalogs."""

from __future__ import annotations

import json
import re
from argparse import Namespace
from pathlib import Path

from dci import __version__
from dci.configs import (
    DATASETS,
    DEFAULT_BASE_SEED,
    DEFAULT_RUNS,
    IMAGENET21K_B_VALUES,
    STANDARD_B_VALUES,
)
from dci.runner import (
    _load_imagenet21k_catalog,
    git_revision,
    git_worktree_dirty,
    write_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


def test_runtime_defaults_are_centralized():
    assert DEFAULT_RUNS == 5
    assert DEFAULT_BASE_SEED == 0
    assert all(
        DATASETS[name].default_b_values == STANDARD_B_VALUES == (10,)
        for name in ("cifar100", "cub200", "food101", "imagenet1k")
    )
    imagenet21k = DATASETS["imagenet21k"]
    assert imagenet21k.default_b_values == IMAGENET21K_B_VALUES == (100,)
    assert imagenet21k.default_max_samples == 1_000
    assert imagenet21k.expected_images is None


def test_imagenet21k_source_catalog_matches_runtime_candidates():
    config = DATASETS["imagenet21k"]
    source = ROOT / str(config.label_file)
    by_wnid, candidate_names = _load_imagenet21k_catalog(source)
    rows = [
        line.split(",", maxsplit=2)
        for line in source.read_text(encoding="utf-8").splitlines()
        if "," in line
    ]
    source_wnids = [row[0].strip() for row in rows]
    source_names = [row[1].strip() for row in rows]

    assert config.expected_labels == len(candidate_names) == 20_101
    assert len(source_wnids) == len(set(source_wnids)) == len(by_wnid) == 21_843
    assert len(candidate_names) == len(set(candidate_names)) == 20_101
    assert candidate_names == list(dict.fromkeys(source_names))
    assert by_wnid == dict(zip(source_wnids, source_names))


def test_runtime_manifest_records_seed_decoding_and_code_revision(tmp_path):
    args = Namespace(
        model="offline-test",
        api_key="must-not-be-recorded",
        api_base="http://localhost/v1",
        max_workers=10,
        max_tokens=None,
        temperature=None,
        top_p=None,
        save_traces=True,
        seed=DEFAULT_BASE_SEED,
        runs=DEFAULT_RUNS,
    )
    path = tmp_path / "b-100.manifest.json"
    write_manifest(
        path,
        args,
        ROOT,
        run_index=3,
        run_seed=2,
        b=100,
        mode="b-100",
        images=["n00000001/example.jpg"],
        selected_labels=["example"],
        full_catalog_sha256="catalog",
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))

    assert manifest["run_index"] == 3 and manifest["run_seed"] == 2
    assert manifest["decoding"] == {
        "max_tokens": None,
        "temperature": None,
        "top_p": None,
    }
    assert manifest["git_revision"] == git_revision(ROOT)
    assert re.fullmatch(r"[0-9a-f]{40}", manifest["git_revision"])
    assert manifest["git_worktree_dirty"] == git_worktree_dirty(ROOT)
    assert isinstance(manifest["git_worktree_dirty"], bool)
    assert "must-not-be-recorded" not in path.read_text(encoding="utf-8")


def test_release_version_is_consistent():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert f'version = "{__version__}"' in pyproject
    assert f"version: {__version__}" in citation
    assert "date-released: 2026-09-05" in citation
