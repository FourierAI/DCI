import json
import random
from pathlib import Path

from dci.configs import DATASETS
from dci.runner import (
    PROMPT,
    DCIClassifier,
    _load_cub_official,
    _load_food101_official,
    _load_imagenet21k_catalog,
    _load_json_split,
    load_dataset,
    read_completed,
    sample_images,
    select_candidate_problem,
)


def classifier(seed=0):
    return DCIClassifier(None, "model", 2, 8, seed=seed)


def test_groups_preserve_order_and_remainder_without_rng():
    assert DCIClassifier.groups(["a", "b", "c", "d", "e"], 2) == [
        ["a", "b"],
        ["c", "d"],
        ["e"],
    ]


def test_groups_are_randomized_reproducibly():
    labels = list("abcdefgh")
    first = DCIClassifier.groups(labels, 3, random.Random(7))
    second = DCIClassifier.groups(labels, 3, random.Random(7))
    assert first == second
    assert [item for group in first for item in group] != labels
    assert sorted(item for group in first for item in group) == labels


def test_normalize_prediction_uses_the_paper_regex_rule():
    candidates = ["Black-footed Albatross", "wild_cat", "cat"]
    assert (
        DCIClassifier.normalize_prediction(
            'Answer: "black-footed albatross".', candidates
        )
        == "Black-footed Albatross"
    )
    assert DCIClassifier.normalize_prediction("None", candidates) is None
    assert (
        DCIClassifier.normalize_prediction("None, maybe wild_cat", candidates) is None
    )
    assert DCIClassifier.normalize_prediction("wild_cat or cat", candidates) is None
    assert DCIClassifier.normalize_prediction("unknown", candidates) is None


def test_prompt_uses_the_final_paper_wording():
    assert "from the candidate list" in PROMPT
    assert "candidate category name list" not in PROMPT


def test_query_retains_model_token_default_unless_overridden():
    class Completions:
        def __init__(self):
            self.requests = []

        def create(self, **kwargs):
            self.requests.append(kwargs)
            message = type("Message", (), {"content": "cat"})()
            choice = type("Choice", (), {"message": message})()
            return type("Response", (), {"choices": [choice]})()

    completions = Completions()
    client = type(
        "Client",
        (),
        {"chat": type("Chat", (), {"completions": completions})()},
    )()
    DCIClassifier(client, "model", 1, None).query(
        "data:image/jpeg;base64,eA==", ["cat"]
    )
    assert "max_tokens" not in completions.requests[0]


def test_nested_candidate_names_do_not_create_a_false_multiple_match():
    assert (
        DCIClassifier.normalize_prediction("The answer is hot dog.", ["dog", "hot dog"])
        == "hot dog"
    )


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


def test_candidate_subset_keeps_all_images_from_selected_classes():
    mapping = {f"{label}{i}.jpg": label for label in "abc" for i in range(2)}
    selected_mapping, labels, images = select_candidate_problem(
        mapping, list("abc"), 2, None, None, random.Random(3)
    )
    assert len(labels) == 2
    assert len(images) == 4
    assert set(selected_mapping.values()) == set(labels)


def test_encode_image_uses_detected_mime_type(tmp_path):
    image = tmp_path / "sample.png"
    image.write_bytes(b"not-a-real-png")
    assert DCIClassifier.encode_image(image).startswith("data:image/png;base64,")


def test_empty_combined_set_returns_none_without_global_fallback(tmp_path):
    image = tmp_path / "sample.jpg"
    image.write_bytes(b"image")
    model = classifier()
    calls = []

    def reject(_image, labels):
        calls.append(labels)
        return "None"

    model.query = reject
    assert model.classify(image, ["a", "b", "c", "d"], 2) is None
    assert len(calls) == 2


def test_single_survivor_is_returned_without_an_extra_call(tmp_path):
    image = tmp_path / "sample.jpg"
    image.write_bytes(b"image")
    model = classifier(seed=4)
    calls = []

    def keep_one(_image, labels):
        calls.append(labels)
        return labels[0] if len(calls) == 1 else "None"

    model.query = keep_one
    prediction = model.classify(image, ["a", "b", "c", "d"], 2)
    assert prediction in {"a", "b", "c", "d"}
    assert len(calls) == 2


def test_read_completed_skips_malformed_rows(tmp_path):
    output = tmp_path / "results.jsonl"
    output.write_text(
        json.dumps({"image": "valid.jpg"}) + "\n" + "{unfinished",
        encoding="utf-8",
    )
    assert read_completed(output) == {"valid.jpg"}


def test_imagenet1k_catalog_keeps_all_1000_class_ids():
    mapping, labels = _load_json_split(
        Path(DATASETS["imagenet1k"].metadata),
        "val",
    )
    assert len(mapping) == 50_000
    assert len(labels) == 1_000
    assert sum(label.startswith("crane [class ") for label in labels) == 2


def test_imagenet21k_catalog_keeps_all_synsets_unique():
    path = Path(DATASETS["imagenet21k"].label_file)
    by_wnid, labels = _load_imagenet21k_catalog(path)
    assert len(by_wnid) == 21_843
    assert len(labels) == len(set(labels)) == 21_843


def test_bundled_cifar_protocol_count_is_validated():
    dataset = load_dataset(Path.cwd(), DATASETS["cifar100"], None, None)
    assert len(dataset.image_to_label) == 10_000
    assert len(dataset.labels) == 100


def test_cub_loader_uses_only_official_test_rows(tmp_path):
    root = tmp_path / "CUB_200_2011"
    (root / "images").mkdir(parents=True)
    (root / "images.txt").write_text(
        "1 001.Bird/one.jpg\n2 002.Other/two.jpg\n", encoding="utf-8"
    )
    (root / "image_class_labels.txt").write_text("1 1\n2 2\n", encoding="utf-8")
    (root / "train_test_split.txt").write_text("1 0\n2 1\n", encoding="utf-8")
    (root / "classes.txt").write_text("1 001.Bird\n2 002.Other\n", encoding="utf-8")
    image_root, mapping, labels = _load_cub_official(root)
    assert image_root == root / "images"
    assert mapping == {"001.Bird/one.jpg": "Bird"}
    assert labels == ["Bird", "Other"]


def test_food101_loader_uses_official_test_file(tmp_path):
    root = tmp_path / "food-101"
    (root / "images").mkdir(parents=True)
    (root / "meta").mkdir()
    (root / "meta" / "test.txt").write_text(
        "apple_pie/one\nbeet_salad/two\n", encoding="utf-8"
    )
    image_root, mapping, labels = _load_food101_official(root / "images")
    assert image_root == root / "images"
    assert mapping == {
        "apple_pie/one.jpg": "apple_pie",
        "beet_salad/two.jpg": "beet_salad",
    }
    assert labels == ["apple_pie", "beet_salad"]
