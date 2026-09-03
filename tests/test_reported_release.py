"""Offline consistency checks for the published aggregate-data snapshot."""

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data/results"


def table(name):
    payload = json.loads((RESULTS / "reported_results.json").read_text())
    return payload["tables"][name]["rows"]


def test_reported_file_hashes_match_the_release_manifest():
    manifest = json.loads((RESULTS / "manifest.json").read_text())
    for record in manifest["files"]:
        assert (
            hashlib.sha256((RESULTS / record["path"]).read_bytes()).hexdigest()
            == record["sha256"]
        )


def test_main_gains_and_paper_macro_average():
    rows = [row for row in table("Main DCI") if row[1] != "Avg."]
    assert len(rows) == 24
    assert all(row[4] > row[2] for row in rows)
    for row in rows:
        assert row[6] == pytest.approx(row[4] - row[2], abs=1e-9)
    assert sum(row[6] for row in rows) / 24 == pytest.approx(4.670416666666667)


def test_candidate_scaling_and_shared_main_values():
    main = {row[0]: row for row in table("Main DCI") if row[1] == "ImageNet-1K"}
    for row in table("Suppression"):
        if row[3] == 1000 and row[2] in main:
            assert row[4:8] == main[row[2]][2:6]
        if row[3] in (10, 20):
            step = 0.04 if row[3] == 10 else 0.02
            assert row[4] / step == pytest.approx(round(row[4] / step))
            assert row[6] / step == pytest.approx(round(row[6] / step))
    kimi = next(
        row
        for row in table("Suppression")
        if row[2] == "Kimi-VL-A3B-Instruct" and row[3] == 10
    )
    assert kimi[4] == 94.28 and kimi[6] == 93.16
    gains = [row[6] - row[4] for row in table("Suppression") if row[3] == 10]
    assert sum(gains) / 6 == pytest.approx(-1.0533333333333321)


def test_imagenet21k_gains_and_grid():
    for row in table("IN21K Main"):
        assert row[8] == pytest.approx(row[5] - row[2], abs=1e-9)
        for mean in (row[2], row[5]):
            assert mean / 0.02 == pytest.approx(round(mean / 0.02))


def test_only_current_tts_panels_are_exported():
    rows = table("TTS")
    assert len(rows) == 16
    assert {row[0] for row in rows} == {"CUB-200-2011", "ImageNet-1K"}


def test_current_figures_and_legacy_aliases_are_synchronized():
    manifest = json.loads((ROOT / "assets/figures/manifest.json").read_text())
    assert [item["figure"] for item in manifest["figures"]] == list(range(1, 13))
    assert manifest["figures"][-1]["basename"] == "dci_B10"
    for item in manifest["figures"]:
        name = item["basename"]
        for ext in ("pdf", "png"):
            source = ROOT / f"assets/figures/{name}.{ext}"
            assert (
                hashlib.sha256(source.read_bytes()).hexdigest() == item[f"{ext}_sha256"]
            )
        assert (ROOT / f"assets/figures/{name}.png").read_bytes() == (
            ROOT / f"docs/assets/{name}.png"
        ).read_bytes()
    for alias, name in manifest["compatibility_aliases"].items():
        assert (ROOT / f"assets/figures/{alias}.png").read_bytes() == (
            ROOT / f"assets/figures/{name}.png"
        ).read_bytes()


@pytest.mark.parametrize(
    "filename,counts",
    [("cub200_B10.json", [200, 19, 2, 1]), ("food101_B10.json", [101, 5, 1])],
)
def test_qualitative_trace_counts_and_validity(filename, counts):
    trace = json.loads((RESULTS / "qualitative" / filename).read_text())
    assert "/" not in trace["image"]
    assert trace["config"]["k"] == 10
    assert [trace["config"]["initial_label_count"]] + [
        len(r["output_labels"]) for r in trace["rounds"]
    ] == counts
    for level in trace["rounds"]:
        assert all(len(g["candidate_labels"]) <= 10 for g in level["groups"])
        assert all(
            g["selected_label"] is None or g["selected_label"] in g["candidate_labels"]
            for g in level["groups"]
        )


def test_current_authors_are_consistent_across_public_text():
    for filename in ["README.md", "docs/index.html", "CITATION.cff"]:
        text = (ROOT / filename).read_text()
        assert "Xihang" not in text and "Qian Qiao" not in text
        assert "Kaixin" in text and "Hao" in text


def test_release_is_identified_as_reported_aggregates():
    payload = json.loads((RESULTS / "reported_results.json").read_text())
    assert payload["provenance"]["original_five_run_logs_available"] is False
    assert payload["manuscript"]["pages"] == 36
