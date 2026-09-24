import json
import shutil
from collections import Counter
from pathlib import Path

import pytest

from careguide.external_data import CAREGUIDE_INTENT_LABELS
from careguide.intent_v4_dataset import (
    DATASET_ROOT,
    load_v4_test_cases,
    load_v4_training_cases,
    load_v4_validation_cases,
    normalize_message,
)
from scripts.build_careguide_intent_v4_dataset import build
from scripts.run_careguide_intent_v4_validation import evaluate


EXPECTED_COUNTS = {"train": 5132, "validation": 1102, "test": 1097}


def test_v4_loaders_validate_all_frozen_splits():
    splits = {
        "train": load_v4_training_cases(),
        "validation": load_v4_validation_cases(),
        "test": load_v4_test_cases(),
    }
    assert {name: len(cases) for name, cases in splits.items()} == EXPECTED_COUNTS
    for name, cases in splits.items():
        assert {case.split for case in cases} == {name}
        assert {case.label for case in cases} == set(CAREGUIDE_INTENT_LABELS)
        custom = Counter(case.label for case in cases)
        expected_custom = 140 if name == "train" else 30
        assert custom["appointment_action"] == expected_custom
        assert custom["emergency"] == expected_custom


def test_v4_splits_have_no_family_or_normalized_message_leakage():
    splits = (
        load_v4_training_cases(),
        load_v4_validation_cases(),
        load_v4_test_cases(),
    )
    family_sets = [{case.scenario_family for case in cases} for cases in splits]
    message_sets = [{normalize_message(case.message) for case in cases} for cases in splits]
    for left in range(len(splits)):
        for right in range(left + 1, len(splits)):
            assert family_sets[left].isdisjoint(family_sets[right])
            assert message_sets[left].isdisjoint(message_sets[right])


def test_v4_appointment_validation_and_test_cover_six_workflow_actions():
    expected = {
        "doctor_search", "slot_search", "appointment_create",
        "appointment_reschedule", "appointment_cancel", "appointment_status",
    }
    for cases in (load_v4_validation_cases(), load_v4_test_cases()):
        actions = {
            case.upstream_label
            for case in cases
            if case.label == "appointment_action"
            and case.source_dataset == "careguide-intent-v4-curated"
        }
        assert actions == expected


def test_v4_loader_rejects_tampered_content(tmp_path: Path):
    copied = tmp_path / "dataset"
    shutil.copytree(DATASET_ROOT, copied)
    train = copied / "train.jsonl"
    train.write_bytes(train.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="integrity mismatch"):
        load_v4_training_cases(copied)


def test_v4_builder_is_deterministic(tmp_path: Path):
    generated = tmp_path / "generated"
    rebuilt = build(generated)
    frozen = json.loads((DATASET_ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert rebuilt == frozen
    for filename in ("train.jsonl", "validation.jsonl", "test.jsonl"):
        assert (generated / filename).read_bytes() == (DATASET_ROOT / filename).read_bytes()


def test_v4_validation_baseline_never_claims_test_access():
    report = evaluate()
    assert report["split"] == "validation"
    assert report["test_split_accessed"] is False
    assert report["count"] == EXPECTED_COUNTS["validation"]
    assert set(report["per_label"]) == set(CAREGUIDE_INTENT_LABELS)


def test_v4_qwen_centroid_artifact_is_train_only():
    artifact_path = (DATASET_ROOT.parents[1] / "external" / "cblue" /
                     "careguide13_qwen3_centroids_v1.json")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["version"] == "careguide-13-qwen3-train-centroids-v1"
    assert artifact["source_dataset_version"] == "careguide-intent-v4-13class-v1"
    assert artifact["embedding_model"] == "Qwen/Qwen3-Embedding-0.6B"
    assert artifact["embedding_dimensions"] == 1024
    assert set(artifact["labels"]) == set(CAREGUIDE_INTENT_LABELS)
    assert all(len(vector) == 1024 for vector in artifact["labels"].values())
    assert "test" not in artifact
