"""Intent DATA contracts only. This module does not implement recognition."""
from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from careguide.domain import IntentCategory

DATASET_ROOT = Path(__file__).resolve().parents[1] / 'data' / 'eval' / 'medical_intent_v1'


class Disposition(str, Enum):
    ROUTE = 'route'
    URGENT_HANDOFF = 'urgent_handoff'
    NEEDS_CLARIFICATION = 'needs_clarification'
    OUT_OF_SCOPE = 'out_of_scope'


class HistoryTurn(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    role: Literal['user', 'assistant']
    content: str = Field(min_length=1)


class IntentCase(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    id: str
    split: Literal['dev', 'heldout']
    scenario_family: str
    message: str = Field(min_length=1)
    history: tuple[HistoryTurn, ...]
    primary_intent: IntentCategory | None
    expected_intents: tuple[IntentCategory, ...]
    disposition: Disposition
    expected_agents: tuple[str, ...]
    critical: bool
    policy_action: Literal['sourced_information_only', 'redirect_to_professional',
                           'guidance_only_no_transaction', 'urgent_handoff',
                           'clarify_task', 'out_of_scope']
    tags: tuple[str, ...]
    annotation_rationale: str
    annotation_status: Literal['developer_routing_annotation']
    clinical_reviewed: Literal[False]
    provenance: Literal['hand_authored_synthetic_user_utterance']
    emergency_reference_urls: tuple[str, ...]

    @model_validator(mode='after')
    def validate_routing_contract(self):
        expected = set(self.expected_intents)
        if len(expected) != len(self.expected_intents):
            raise ValueError('Duplicate intents in case')
        if self.disposition in {Disposition.NEEDS_CLARIFICATION, Disposition.OUT_OF_SCOPE}:
            if expected or self.primary_intent is not None or self.expected_agents or self.critical:
                raise ValueError('Boundary states must not be forced into a domain intent')
        elif self.primary_intent not in expected:
            raise ValueError('Primary intent must occur in expected intent set')
        if self.disposition == Disposition.URGENT_HANDOFF:
            if expected != {IntentCategory.EMERGENCY} or not self.critical or self.expected_agents:
                raise ValueError('Urgent cases preempt all domain execution')
            if not self.emergency_reference_urls:
                raise ValueError('Urgent annotation requires supporting public references')
        elif self.critical or IntentCategory.EMERGENCY in expected:
            raise ValueError('Emergency may not silently route into ordinary domains')
        return self


def _read_cases(root: Path, split: str) -> tuple[IntentCase, ...]:
    filename = {'dev': 'dev.jsonl', 'heldout': 'heldout.jsonl'}[split]
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    raw = (root/filename).read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest['files'][filename]['sha256']:
        raise ValueError(f'Intent dataset integrity mismatch: {filename}')
    cases = tuple(IntentCase.model_validate_json(line) for line in raw.decode('utf-8').splitlines() if line.strip())
    if len(cases) != manifest['files'][filename]['count'] or any(c.split != split for c in cases):
        raise ValueError('Intent split counts or identities disagree with manifest')
    return cases


def load_development_cases() -> tuple[IntentCase, ...]:
    """The ONLY example/template loader. It never opens heldout.jsonl."""
    return _read_cases(DATASET_ROOT, 'dev')


def load_evaluation_cases() -> tuple[IntentCase, ...]:
    """Explicit evaluator access; forbidden for prompts, templates or tuning."""
    return _read_cases(DATASET_ROOT, 'heldout')
