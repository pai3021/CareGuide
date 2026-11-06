"""Evaluation-only business labels, aligned to the action catalog, not QIC-13."""
import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from careguide.orchestration.candidate_routing import READ_ACTIONS
from careguide.orchestration.contracts import ACTION_ARGUMENTS, Contract

DATASET = Path(__file__).with_name('business_intent_cases.json')


class BusinessCase(Contract):
    id: str
    group: str
    text: str = Field(min_length=1, max_length=1000)
    operation: Literal['new', 'change', 'answer', 'cancel', 'clarify', 'direct', 'boundary']
    actions: list[str]
    arguments: dict[str, dict] = Field(default_factory=dict)
    forbidden_actions: list[str] = Field(default_factory=list)
    local_eligible: bool = False
    forbid_local_plan: bool = False
    workflow_event: dict | None = None

    @model_validator(mode='after')
    def labels(self):
        known = set(ACTION_ARGUMENTS)
        if (len(self.actions) != len(set(self.actions)) or
                not set(self.actions+self.forbidden_actions) <= known or
                not set(self.arguments) <= set(self.actions) or set(self.actions) & set(self.forbidden_actions)):
            raise ValueError('Invalid business action labels')
        if self.operation == 'new' and not self.actions:
            raise ValueError('New business requests need action labels')
        if self.local_eligible and (self.operation != 'new' or self.forbid_local_plan or
                                   len(self.actions) != 1 or self.actions[0] not in READ_ACTIONS):
            raise ValueError('Local-eligible labels are single read-only new requests')
        for action, arguments in self.arguments.items():
            # Ground truth annotates a subset; check field names and values
            # without requiring fields deliberately left to user clarification.
            model = ACTION_ARGUMENTS[action]
            if not set(arguments) <= set(model.model_fields):
                raise ValueError('Unknown annotated argument')
            for key, value in arguments.items():
                from pydantic import TypeAdapter
                TypeAdapter(model.model_fields[key].rebuild_annotation()).validate_python(value, strict=True)
        return self


def load_cases(path=DATASET):
    path = Path(path)
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('version') != 'business-intent-eval-v1' or data.get('split') != 'evaluation_only':
        raise ValueError('Unknown business evaluation protocol')
    cases = [BusinessCase.model_validate(row) for row in data['cases']]
    if (not cases or len({c.id for c in cases}) != len(cases) or
            len({c.text for c in cases}) != len(cases)):
        raise ValueError('Duplicate or empty business cases')
    return data, cases, hashlib.sha256(path.read_bytes()).hexdigest()
