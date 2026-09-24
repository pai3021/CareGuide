import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from scripts.run_e2e_evaluation import load_cases, load_development_cases


ROOT = Path(__file__).resolve().parents[1]


def test_e2e_dataset_has_frozen_stratified_splits():
    directory = ROOT / "data" / "eval" / "careguide_e2e_v1"
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["total"] == 120
    split_ids = []
    for split in ("dev", "heldout"):
        raw = (directory / f"{split}.jsonl").read_bytes()
        expected = manifest["files"][f"{split}.jsonl"]
        assert hashlib.sha256(raw).hexdigest() == expected["sha256"]
        cases = [json.loads(line) for line in raw.decode().splitlines()]
        assert len(cases) == expected["count"] == 60
        assert Counter(case["category"] for case in cases) == {
            "knowledge": 35, "appointment": 15, "safety": 10}
        assert all(case["split"] == split for case in cases)
        split_ids.append({case["id"] for case in cases})
    assert split_ids[0].isdisjoint(split_ids[1])


def test_development_loader_returns_only_dev():
    cases = load_development_cases()
    assert len(cases) == 60
    assert {case["split"] for case in cases} == {"dev"}


def test_heldout_loader_requires_explicit_gate():
    with pytest.raises(PermissionError, match="explicit"):
        load_cases("heldout")
    cases = load_cases("heldout", allow_heldout=True)
    assert len(cases) == 60
    assert {case["split"] for case in cases} == {"heldout"}


def test_knowledge_cases_have_explicit_answerability_and_evidence_contracts():
    cases = [case for case in load_development_cases() if case["category"] == "knowledge"]
    answerable = [case for case in cases if case["answerable"]]
    unanswerable = [case for case in cases if not case["answerable"]]
    assert len(answerable) == 32
    assert len(unanswerable) == 3
    assert all(len(case["target_evidence_ids"]) == 1 for case in answerable)
    assert all(not case["target_evidence_ids"] for case in unanswerable)
