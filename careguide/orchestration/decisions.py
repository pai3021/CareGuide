"""Typed boundaries between local understanding, planning and task delegation."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from careguide.orchestration.contracts import Contract, PlanChange, PlanProposal


class MessageSource(Contract):
    id: str = Field(min_length=1, max_length=100)
    content: str = Field(min_length=1, max_length=4000)
    at: str | None


class Goal(Contract):
    id: str
    fragment: str
    action: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    constraints: list[str] = Field(default_factory=list)
    sources: dict[str, str] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)


class IntentUnderstanding(Contract):
    goals: list[Goal] = Field(default_factory=list, max_length=12)
    relation: Literal['single', 'independent', 'doctor_slots', 'unknown'] = 'unknown'
    request_relation: Literal['new', 'existing', 'unknown'] = 'unknown'
    ambiguities: list[str] = Field(default_factory=list)
    unresolved_spans: list[str] = Field(default_factory=list)
    candidates: list[dict] = Field(default_factory=list)
    recognition_source: list[str] = Field(default_factory=list)
    diagnostics: list[str] = Field(default_factory=list)
    semantic_retrieval: dict = Field(default_factory=dict)


class TaskAnswer(Contract):
    task_id: str = Field(min_length=1, max_length=64)
    task_revision: int = Field(ge=1)
    values: dict[str, Any] = Field(min_length=1, max_length=12)


class RequestDecision(Contract):
    kind: Literal['new', 'change', 'answer', 'cancel', 'clarify', 'boundary', 'run']
    plan: PlanProposal | None = None
    change: PlanChange | None = None
    workflow_id: str | None = None
    base_version: int | None = Field(default=None, ge=1)
    answers: list[TaskAnswer] = Field(default_factory=list, max_length=12)
    cancel: list[str] = Field(default_factory=list, max_length=12)
    question: str = Field(default='', max_length=1000)
    boundary: Literal['urgent_handoff', 'diagnosis_request', 'individualized_medication', 'out_of_scope'] | None = None

    @model_validator(mode='after')
    def shape(self):
        if (self.kind == 'new') != (self.plan is not None):
            raise ValueError('Only new requests carry a plan')
        if (self.kind == 'change') != (self.change is not None):
            raise ValueError('Only changes carry a PlanChange')
        if self.kind in {'change', 'answer', 'cancel', 'run'} and (
                not self.workflow_id or self.base_version is None):
            raise ValueError('Existing-task decisions require workflow and version')
        if self.kind == 'change' and self.change.base_version != self.base_version:
            raise ValueError('Conflicting base versions')
        if self.answers and self.kind not in {'answer', 'change'} or self.kind == 'answer' and not self.answers:
            raise ValueError('Invalid answer payload')
        if self.cancel and self.kind != 'cancel' or self.kind == 'cancel' and not self.cancel:
            raise ValueError('Invalid cancellation payload')
        if (self.kind == 'clarify') != bool(self.question):
            raise ValueError('Clarifications require a question')
        if (self.kind == 'boundary') != (self.boundary is not None):
            raise ValueError('Boundary decisions require a boundary')
        return self


class TaskBrief(Contract):
    workflow_id: str
    task_id: str
    task_revision: int
    run_id: str | None
    actor: str
    conversation: str
    objective: str
    purpose: str
    known_facts: dict = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    excluded_options: list[dict] = Field(default_factory=list)
    inputs: dict = Field(default_factory=dict)
    allowed_actions: list[str]
    output_contract: list[str]
    completion_criteria: str
    context_policy: Literal['brief'] = 'brief'


class ArtifactReference(Contract):
    id: str
    workflow_id: str
    task_id: str
    task_revision: int
    sha256: str
    size: int
    content_type: Literal['application/json'] = 'application/json'


class WorkerResult(Contract):
    status: str
    summary: str
    data: dict = Field(default_factory=dict)
    evidence_refs: list[str] = Field(default_factory=list)
    missing_input: list[str] = Field(default_factory=list)
    failure: dict | None = None
    artifact_refs: list[ArtifactReference] = Field(default_factory=list)
