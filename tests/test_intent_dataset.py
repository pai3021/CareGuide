import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from careguide.intent_dataset import DATASET_ROOT, IntentCase, _read_cases, load_development_cases
from scripts.verify_intent_dataset import audit


def test_dataset_contract_and_split_isolation():
    result = audit()
    assert result['case_count'] == 360
    assert result['near_duplicate_flags'] == []


def test_development_loader_never_opens_heldout(monkeypatch):
    original = Path.read_bytes
    def guarded(path):
        assert path.name != 'heldout.jsonl'
        return original(path)
    monkeypatch.setattr(Path, 'read_bytes', guarded)
    assert len(load_development_cases()) == 240


def test_tampering_detected(tmp_path):
    (tmp_path/'manifest.json').write_bytes((DATASET_ROOT/'manifest.json').read_bytes())
    (tmp_path/'dev.jsonl').write_bytes((DATASET_ROOT/'dev.jsonl').read_bytes()+b'\n')
    with pytest.raises(ValueError, match='integrity mismatch'):
        _read_cases(tmp_path, 'dev')


def test_emergency_cannot_execute_ordinary_agent():
    rows = [json.loads(line) for line in (DATASET_ROOT/'dev.jsonl').read_text(encoding='utf-8').splitlines()]
    emergency = next(row for row in rows if row['critical'])
    emergency['expected_agents'] = ['MedicationAgent']
    with pytest.raises(ValidationError, match='preempt'):
        IntentCase.model_validate(emergency)


def test_unknown_task_cannot_be_forced_into_domain():
    rows = [json.loads(line) for line in (DATASET_ROOT/'dev.jsonl').read_text(encoding='utf-8').splitlines()]
    boundary = next(row for row in rows if row['disposition'] == 'needs_clarification')
    boundary['primary_intent'] = 'symptom_query'
    with pytest.raises(ValidationError, match='forced'):
        IntentCase.model_validate(boundary)
