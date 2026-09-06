"""Offline consistency checks for final results and project-page figures."""

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data/results"


def table(name):
    payload = json.loads((RESULTS / "reported_results.json").read_text())
    return payload["tables"][name]["rows"]


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


def test_current_figures_and_project_page_copies_match():
    names = (
        "scaling_performance",
        "method",
        "prompt",
        "dci_complexity_analysis",
        "dci_main",
        "dci_suppression",
        "imagenet21k_dci_advantage",
        "dci_imagenet21k_tradeoff",
        "dci_group_size_tradeoff",
        "tts_accuracy_latency_tradeoff",
        "gs_ablation",
        "dci_B10",
    )
    for name in names:
        pdf = (ROOT / f"assets/figures/{name}.pdf").read_bytes()
        png = (ROOT / f"assets/figures/{name}.png").read_bytes()
        assert pdf.startswith(b"%PDF-")
        assert png.startswith(b"\x89PNG\r\n\x1a\n")
        assert png == (ROOT / f"docs/assets/{name}.png").read_bytes()


def test_current_authors_are_consistent_across_public_text():
    for filename in ["README.md", "docs/index.html", "CITATION.cff"]:
        text = (ROOT / filename).read_text()
        assert "Xihang" not in text and "Qian Qiao" not in text
        assert "Kaixin" in text and "Hao" in text


def test_final_results_are_complete_and_unchanged():
    payload = json.loads((RESULTS / "reported_results.json").read_text())
    assert payload["schema_version"] == 1
    assert payload["kind"] == "final aggregate results"
    assert payload["title"] == (
        "Divide-and-Conquer Inference for Large-Scale Image Classification "
        "with Multimodal Large Language Models"
    )
    tables = payload["tables"]
    assert {name: len(data["rows"]) for name, data in tables.items()} == {
        "Main DCI": 30,
        "Scaling Wide": 10,
        "Suppression": 36,
        "IN21K Main": 10,
        "IN21K Group Size": 24,
        "Group Size": 24,
        "TTS": 16,
        "Grouping": 6,
    }
    for data in tables.values():
        assert all(len(row) == len(data["columns"]) for row in data["rows"])
    values = [
        value for data in tables.values() for row in data["rows"] for value in row
    ]
    assert sum(type(value) in (int, float) for value in values) == 860
    assert sum(value is None for value in values) == 54
    canonical = json.dumps(tables, ensure_ascii=False, separators=(",", ":"))
    assert hashlib.sha256(canonical.encode()).hexdigest() == (
        "4bcf7880b0dd250a3a1662c28fa579aa13b6c723412586fa1487ccba9f957d55"
    )
