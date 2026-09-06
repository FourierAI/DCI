import json
import random
from pathlib import Path

import pytest

from dci.configs import DATASETS
from dci.prompts import DCI_PROMPT, FLAT_PROMPT, build_prompt
from dci.runner import (
    PROMPT,
    DCIClassifier,
    _load_cub_official,
    _load_food101_official,
    _load_imagenet21k_catalog,
    _load_imagenet21k_index,
    _load_json_split,
    evaluate,
    load_dataset,
    read_completed,
    sample_images,
    select_candidate_problem,
    write_manifest,
    write_summary,
)
from dci.validation import validate_prediction


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


def test_grouping_seed_uses_the_stable_image_identifier(tmp_path):
    model = classifier(seed=9)
    labels = list("abcdefghijkl")
    first = model.groups(
        labels,
        3,
        model._image_rng(tmp_path / "first/root/image.jpg", "class/image.jpg"),
    )
    second = model.groups(
        labels,
        3,
        model._image_rng(tmp_path / "other/root/image.jpg", "class/image.jpg"),
    )
    assert first == second


def test_validate_prediction_label_uses_exact_raw_string_membership():
    candidates = ["Black-footed Albatross", "wild_cat", "cat"]
    assert DCIClassifier.validate_prediction_label("wild_cat", candidates) == "wild_cat"
    rejected = (
        " wild_cat",
        "wild_cat.",
        "Wild_cat",
        '"wild_cat"',
        "Answer: wild_cat",
        "None",
        "None, maybe wild_cat",
        "wild_cat or cat",
        "unknown",
    )
    assert all(
        DCIClassifier.validate_prediction_label(raw, candidates) is None
        for raw in rejected
    )


def test_prompt_uses_the_final_paper_wording():
    assert PROMPT == DCI_PROMPT
    assert "candidate category name list" in PROMPT
    assert "'None'" in DCI_PROMPT
    assert "None" not in FLAT_PROMPT
    assert "wild_cat and electric_guitar" in DCI_PROMPT
    assert "There are 2 categories, [cat,dog]" in build_prompt(["cat", "dog"])


def test_query_retains_model_token_default_unless_overridden():
    class Completions:
        def __init__(self):
            self.requests = []

        def create(self, **kwargs):
            self.requests.append(kwargs)
            message = type("Message", (), {"content": " cat.\n"})()
            choice = type("Choice", (), {"message": message})()
            return type("Response", (), {"choices": [choice]})()

    completions = Completions()
    client = type(
        "Client",
        (),
        {"chat": type("Chat", (), {"completions": completions})()},
    )()
    raw = DCIClassifier(client, "model", 1, None).query(
        "data:image/jpeg;base64,eA==", ["cat"]
    )
    assert raw == " cat.\n"
    assert "max_tokens" not in completions.requests[0]


def test_nested_candidate_names_require_an_exact_complete_match():
    candidates = ["dog", "hot dog"]
    assert DCIClassifier.validate_prediction_label("hot dog", candidates) == "hot dog"
    assert (
        DCIClassifier.validate_prediction_label("The answer is hot dog.", candidates)
        is None
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


def test_imagenet21k_catalog_has_20101_first_names_and_keeps_all_wnids():
    path = Path(DATASETS["imagenet21k"].label_file)
    by_wnid, labels = _load_imagenet21k_catalog(path)
    assert len(by_wnid) == 21_843
    assert len(labels) == len(set(labels)) == 20_101
    assert by_wnid["n02012849"] == by_wnid["n03126707"] == "crane"
    assert DATASETS["imagenet21k"].expected_labels == len(labels)
    source_names = [
        line.split(",", maxsplit=2)[1].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if "," in line
    ]
    assert list(dict.fromkeys(source_names)) == labels


def test_imagenet21k_catalog_preserves_first_name_spelling_and_order(tmp_path):
    catalog = tmp_path / "labels.txt"
    catalog.write_text(
        "n00000001, crane , lifting_device\n"
        "n00000002,crane, bird\n"
        "n00000003,rock_climbing, rock climbing\n"
        "n00000004,sea lion, seal\n"
        "n00000005,Crane, capitalized_alias\n",
        encoding="utf-8",
    )
    by_wnid, labels = _load_imagenet21k_catalog(catalog)
    assert by_wnid == {
        "n00000001": "crane",
        "n00000002": "crane",
        "n00000003": "rock_climbing",
        "n00000004": "sea lion",
        "n00000005": "Crane",
    }
    assert labels == ["crane", "rock_climbing", "sea lion", "Crane"]


@pytest.mark.parametrize("use_json_index", [False, True])
def test_imagenet21k_shared_names_keep_images_from_both_wnids(
    tmp_path, use_json_index
):
    catalog = tmp_path / "labels.txt"
    catalog.write_text(
        "n00000001,crane, lifting_device\nn00000002,crane, bird\n",
        encoding="utf-8",
    )
    by_wnid, labels = _load_imagenet21k_catalog(catalog)
    image_root = tmp_path / "images"
    expected = {"n00000001/one.jpg": "crane", "n00000002/two.jpg": "crane"}
    for name in expected:
        image = image_root / name
        image.parent.mkdir(parents=True, exist_ok=True)
        image.touch()
    metadata_path = None
    if use_json_index:
        metadata_path = tmp_path / "index.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "n00000001/one.jpg": "n00000001",
                    "n00000002/two.jpg": {"wnid": "n00000002"},
                }
            ),
            encoding="utf-8",
        )
    assert labels == ["crane"]
    assert _load_imagenet21k_index(image_root, metadata_path, by_wnid) == expected


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


@pytest.mark.parametrize(
    "raw",
    [
        "tiger cat",
        "The answer is tiger cat.",
        " cat",
        "cat ",
        "cat.",
        "Cat",
        "'cat'",
        '"cat"',
        "`cat`",
        "Answer: cat",
        "cat or dog",
        "cat, dog",
        "unknown",
        "apple, orchard_apple_tree, Malus_pumila",
        "pear, pear_tree, Pyrus_communis",
        "cat and an unknown class",
        "None, maybe cat",
        "This is probably cat",
    ],
)
def test_out_of_group_and_multiple_label_outputs_are_invalid(raw):
    result = validate_prediction(raw, ["cat", "dog", "apple", "pear"])
    assert result.label is None
    assert result.status == "invalid"


def test_only_exact_none_is_distinguished_from_invalid():
    assert validate_prediction("None", ["cat"]).status == "none"
    for raw in ("none", "NONE", " None", "None ", "'None'", "Answer: None."):
        assert validate_prediction(raw, ["cat"]).status == "invalid"


def test_caller_supplied_comma_label_is_matched_as_a_whole():
    # Generic validator coverage: ImageNet-21K itself uses only the first name.
    label = "apple, orchard_apple_tree, Malus_pumila"
    assert validate_prediction(label, ["apple", label]).label == label
    assert validate_prediction(label, ["apple"]).label is None


def test_case_distinct_candidates_are_resolved_by_exact_spelling():
    assert validate_prediction("cat", ["cat", "CAT"]).label == "cat"
    assert validate_prediction("CAT", ["cat", "CAT"]).label == "CAT"
    assert validate_prediction("Cat", ["cat", "CAT"]).status == "invalid"


def test_each_call_validates_against_only_its_supplied_candidate_group():
    model = classifier()
    survivors = model._record_level(
        [["cat", "lynx"], ["dog", "wolf"]],
        ["dog", "dog"],
    )
    assert survivors == ["dog"]
    first, second = model.last_trace[0]["groups"]
    assert first["response"] == "dog" and first["status"] == "invalid"
    assert second["response"] == "dog" and second["status"] == "selected"


def test_flat_records_and_rejects_an_untouched_raw_variant(tmp_path):
    model = classifier()
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")
    model.query = lambda image, labels, **kwargs: " cat\n"
    assert model.classify_flat(image, ["cat", "dog"]) is None
    group = model.last_trace[0]["groups"][0]
    assert group["response"] == " cat\n"
    assert group["prediction"] is None
    assert group["status"] == "invalid"


def test_flat_and_dci_single_calls_use_distinct_templates(tmp_path):
    from types import SimpleNamespace

    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="cat"))]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    model = DCIClassifier(client, "test", 1, 16, temperature=0.2, top_p=0.9)
    image = tmp_path / "image.png"
    image.write_bytes(b"image")
    assert model.classify_flat(image, ["cat", "dog"]) == "cat"
    assert model.classify(image, ["cat", "dog"], 2) == "cat"
    texts = [request["messages"][0]["content"][1]["text"] for request in requests]
    assert texts == [
        build_prompt(["cat", "dog"], flat=True),
        build_prompt(["cat", "dog"]),
    ]
    assert all(
        request["temperature"] == 0.2 and request["top_p"] == 0.9
        for request in requests
    )


def test_invalid_final_response_is_filtered_and_recorded(tmp_path):
    model = classifier()
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")
    model.query = lambda image, labels: "tiger cat"
    assert model.classify(image, ["cat", "dog"], 2) is None
    assert model.last_trace[0]["groups"][0]["status"] == "invalid"


def test_empty_and_singleton_inputs_do_not_read_an_image(tmp_path):
    model = classifier()
    assert model.classify(tmp_path / "missing.jpg", [], 2) is None
    assert model.classify(tmp_path / "missing.jpg", ["cat"], 2) == "cat"
    assert model.last_trace == []


def test_below_two_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="at least 2"):
        classifier().classify(tmp_path / "missing.jpg", ["cat"], 1)


def test_group_size_cannot_exceed_the_initial_candidate_count(tmp_path):
    with pytest.raises(ValueError, match="cannot exceed"):
        classifier().classify(tmp_path / "missing.jpg", ["cat", "dog"], 3)


def test_all_invalid_groups_are_pruned_without_retry(tmp_path):
    model = classifier()
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")
    model.query = lambda image, labels: "outside"
    assert model.classify(image, ["a", "b", "c", "d"], 2) is None
    assert len(model.last_trace) == 1
    assert len(model.last_trace[0]["groups"]) == 2
    assert all(g["status"] == "invalid" for g in model.last_trace[0]["groups"])


def test_resumed_latency_uses_all_stored_image_timings(tmp_path):
    path = tmp_path / "result.jsonl"
    rows = [
        {"prediction": "cat", "target": "cat", "latency_seconds": 1.0},
        {"prediction": None, "target": "cat", "latency_seconds": 3.0},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = evaluate(path, elapsed=3.0, new_samples=1)
    assert result["accuracy"] == 50
    assert result["correct"] == 1 and result["total"] == 2
    assert result["latency_mean"] == 2.0
    assert evaluate(path, elapsed=0, new_samples=0)["latency_mean"] == 2.0


def test_summary_records_ddof_without_rounding_run_accuracies(tmp_path):
    runs = [{"accuracy": value, "latency_mean": 1.0} for value in [10, 20, 30, 40, 50]]
    path = tmp_path / "summary.json"
    write_summary(path, {"test": runs}, ddof=0)
    population = json.loads(path.read_text())["settings"]["test"]
    write_summary(path, {"test": runs}, ddof=1)
    sample = json.loads(path.read_text())["settings"]["test"]
    assert sample["accuracy_mean"] == population["accuracy_mean"] == 30
    assert sample["accuracy_sd"] / population["accuracy_sd"] == pytest.approx(
        (5 / 4) ** 0.5
    )
    assert sample["sd_ddof"] == 1


def test_manifest_blocks_reusing_predictions_after_protocol_changes(tmp_path):
    from argparse import Namespace

    args = Namespace(
        model="test",
        api_key="never-store-this",
        api_base="http://localhost",
        max_workers=1,
        max_tokens=16,
        temperature=None,
        top_p=None,
        save_traces=False,
    )
    path = tmp_path / "baseline.manifest.json"
    kwargs = {
        "run_index": 1,
        "run_seed": 0,
        "b": 2,
        "mode": "baseline",
        "images": ["a.jpg"],
        "selected_labels": ["cat", "dog"],
        "full_catalog_sha256": "catalog",
    }
    write_manifest(path, args, tmp_path, **kwargs)
    assert "never-store-this" not in path.read_text()
    (tmp_path / "baseline.jsonl").write_text('{"image":"a.jpg"}\n')
    write_manifest(path, args, tmp_path, **kwargs)
    original = path.read_text()
    args.temperature = 0.5
    with pytest.raises(ValueError, match="Cannot mix"):
        write_manifest(path, args, tmp_path, **kwargs)
    assert path.read_text() == original
    args.temperature = None
    args.max_retries = 7
    with pytest.raises(ValueError, match="max_retries"):
        write_manifest(path, args, tmp_path, **kwargs)


def test_cli_dispatches_flat_separately_and_saves_traces(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from dci import runner

    image = tmp_path / "a.jpg"
    image.write_bytes(b"image")
    dataset = runner.LoadedDataset(
        tmp_path, {"a.jpg": "cat"}, ["cat", "dog"], "catalog"
    )
    monkeypatch.setattr(runner, "load_dataset", lambda *args: dataset)
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="cat"))]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    monkeypatch.setattr(runner, "OpenAI", lambda **kwargs: client)
    common = [
        "dci-eval",
        "--dataset",
        "imagenet1k",
        "--model",
        "offline-test",
        "--image-root",
        str(tmp_path),
        "--output-dir",
        str(tmp_path / "out"),
        "--runs",
        "1",
        "--save-traces",
    ]
    monkeypatch.setattr(sys, "argv", common + ["--baseline"])
    runner.main()
    monkeypatch.setattr(sys, "argv", common + ["--b-values", "2"])
    runner.main()
    texts = [request["messages"][0]["content"][1]["text"] for request in requests]
    assert "None" not in texts[0] and "'None'" in texts[1]
    root = tmp_path / "out/imagenet1k/offline-test/n-2/run-01"
    flat = json.loads((root / "baseline.jsonl").read_text())
    assert flat["trace"][0]["groups"][0]["response"] == "cat"
    assert flat["call_count"] == 1
    flat_manifest = json.loads((root / "baseline.manifest.json").read_text())
    dci_manifest = json.loads((root / "b-2.manifest.json").read_text())
    assert flat_manifest["prompt_sha256"] != dci_manifest["prompt_sha256"]
    assert flat_manifest["evaluation_images"] == dci_manifest["evaluation_images"]
