"""Independent fine-intent dataset contract; runtime reads development cases only."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from careguide.domain import MedicalTask
from careguide.intent_dataset import Disposition, HistoryTurn

DATASET_ROOT = Path(__file__).resolve().parents[1] / 'data' / 'eval' / 'medical_intent_v2'
DOMAIN_BY_FINE = {
    **{name.value: 'triage' for name in (
        MedicalTask.SYMPTOM_INFORMATION, MedicalTask.CARE_PREPARATION, MedicalTask.DEPARTMENT_QUERY)},
    **{name.value: 'medication' for name in (
        MedicalTask.DRUG_LABEL, MedicalTask.DRUG_DOSAGE_INFORMATION,
        MedicalTask.DRUG_CONTRAINDICATIONS, MedicalTask.DRUG_PRECAUTIONS,
        MedicalTask.DRUG_INTERACTIONS)},
    **{name.value: 'appointment' for name in (
        MedicalTask.APPOINTMENT_CHANNELS, MedicalTask.APPOINTMENT_PROCESS,
        MedicalTask.APPOINTMENT_MATERIALS, MedicalTask.APPOINTMENT_CANCELLATION,
        MedicalTask.FOLLOWUP_GUIDANCE)},
    **{name.value: 'local_workflow' for name in (
        MedicalTask.HANDOFF_REQUEST, MedicalTask.HANDOFF_QUERY)},
}
FINE_LABELS = tuple(DOMAIN_BY_FINE)
BOUNDARY_LABELS = ('urgent_handoff', 'needs_clarification', 'out_of_scope')


class FineIntentCase(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    id: str = Field(min_length=1)
    split: Literal['dev', 'heldout']
    scenario_family: str = Field(min_length=1)
    message: str = Field(min_length=1)
    history: tuple[HistoryTurn, ...]
    disposition: Disposition
    primary_fine_intent: MedicalTask | None
    fine_intents: tuple[MedicalTask, ...]
    primary_domain: Literal['triage', 'medication', 'appointment', 'local_workflow'] | None
    policy_action: Literal['sourced_information_only', 'redirect_to_professional',
                           'guidance_only_no_transaction', 'local_handoff_only',
                           'local_handoff_lookup', 'urgent_handoff', 'clarify_task', 'out_of_scope']
    missing_entities: tuple[str, ...]
    source_coverage: Literal['not_assessed', 'covered', 'uncovered']
    annotation_rationale: str = Field(min_length=1)
    provenance: Literal['developer_synthetic']
    clinical_reviewed: Literal[False]

    @model_validator(mode='after')
    def validate_contract(self):
        if len(set(self.fine_intents)) != len(self.fine_intents):
            raise ValueError('Duplicate fine intents')
        if self.disposition == Disposition.ROUTE:
            if not self.primary_fine_intent or self.primary_fine_intent not in self.fine_intents:
                raise ValueError('Route requires a primary fine intent')
            if self.primary_domain != DOMAIN_BY_FINE[self.primary_fine_intent.value]:
                raise ValueError('Fine intent and primary domain disagree')
        elif self.primary_fine_intent or self.fine_intents or self.primary_domain:
            raise ValueError('Boundary states cannot be forced into fine intents')
        if self.disposition == Disposition.URGENT_HANDOFF and self.policy_action != 'urgent_handoff':
            raise ValueError('Urgent cases require urgent action')
        if self.source_coverage != 'not_assessed':
            raise ValueError('Synthetic intent labels do not assess source coverage')
        return self


def _read(split: Literal['dev', 'heldout'], root: Path = DATASET_ROOT) -> tuple[FineIntentCase, ...]:
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['version'] != 'careguide-medical-intent-v2' or manifest['labels'] != list(FINE_LABELS):
        raise ValueError('Unsupported V2 intent manifest')
    filename = f'{split}.jsonl'
    raw = (root / filename).read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest['files'][filename]['sha256']:
        raise ValueError(f'V2 intent dataset integrity mismatch: {filename}')
    cases = tuple(FineIntentCase.model_validate_json(line) for line in raw.decode('utf-8').splitlines() if line.strip())
    if len(cases) != manifest['files'][filename]['count'] or any(case.split != split for case in cases):
        raise ValueError('V2 intent split count mismatch')
    expected = {**{label: 20 if split == 'dev' else 10 for label in FINE_LABELS},
                **{label: 20 if split == 'dev' else 10 for label in BOUNDARY_LABELS}}
    actual = Counter(case.primary_fine_intent.value if case.primary_fine_intent else case.disposition.value
                     for case in cases)
    if dict(actual) != expected or len({case.id for case in cases}) != len(cases):
        raise ValueError('V2 intent label balance or identities disagree with manifest')
    return cases


def load_v2_development_cases() -> tuple[FineIntentCase, ...]:
    """Only loader available to prompts, templates and development tuning."""
    return _read('dev')


def load_v2_evaluation_cases() -> tuple[FineIntentCase, ...]:
    """Held-out access is explicit and must never feed recognition branches."""
    return _read('heldout')
