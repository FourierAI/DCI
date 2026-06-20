import json
import random

from dci.runner import DCIClassifier, read_completed, sample_images


def test_groups_preserve_order_and_remainder():
    assert DCIClassifier.groups(["a", "b", "c", "d", "e"], 2) == [
        ["a", "b"],
        ["c", "d"],
        ["e"],
    ]


def test_normalize_prediction_is_case_insensitive():
    candidates = ["Black-footed Albatross", "wild_cat"]
    assert (
        DCIClassifier.normalize_prediction(
            '"black-footed albatross"', candidates
        )
        == "Black-footed Albatross"
    )
    assert DCIClassifier.normalize_prediction("None", candidates) is None
    assert DCIClassifier.normalize_prediction("unknown", candidates) is None


def test_sampling_is_deterministic_and_balanced():
    mapping = {
        "a1.jpg": "a",
        "a2.jpg": "a",
        "b1.jpg": "b",
        "b2.jpg": "b",
    }
    first = sample_images(mapping, 1, None, random.Random(7))
    second = sample_images(mapping, 1, None, random.Random(7))
    assert first == second
    assert len(first) == 2
    assert {mapping[path] for path in first} == {"a", "b"}


def test_encode_image_uses_detected_mime_type(tmp_path):
    image = tmp_path / "sample.png"
    image.write_bytes(b"not-a-real-png")
    assert DCIClassifier.encode_image(image).startswith("data:image/png;base64,")


def test_read_completed_skips_malformed_rows(tmp_path):
    output = tmp_path / "results.jsonl"
    output.write_text(
        json.dumps({"image": "valid.jpg"}) + "\n" + "{unfinished",
        encoding="utf-8",
    )
    assert read_completed(output) == {"valid.jpg"}
