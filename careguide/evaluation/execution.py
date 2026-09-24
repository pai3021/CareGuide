"""Pure trace-to-contract scoring for CareGuide workflows.

The evaluator never executes an agent or changes a business record. Its labels
describe engineering contract failures, not clinical correctness.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping

from careguide.domain import AGENTS, AgentType
from careguide.trace_store import RequestTrace, validate_trace_completeness


FAILURE_LAYERS = (
    ('UNDERSTANDING_FAILURE', 'understanding'),
    ('ROUTING_FAILURE', 'routing'),
    ('SKILL_SELECTION_FAILURE', 'skill_selection'),
    ('MISSING_REQUIRED_TOOL', 'tool_selection'),
    ('UNNECESSARY_TOOL', 'tool_selection'),
    ('FORBIDDEN_TOOL', 'tool_selection'),
    ('TOOL_ARGUMENT_FAILURE', 'tool_arguments'),
    ('TOOL_EXECUTION_FAILURE', 'tool_execution'),
    ('MISSING_EVIDENCE', 'evidence'),
    ('SIDE_EFFECT_FAILURE', 'side_effect'),
    ('FINAL_STATUS_FAILURE', 'final'),
    ('TRACE_INCOMPLETE', 'trace'),
)


@dataclass(frozen=True)
class ExecutionCase:
    case_id: str
    message: str
    task_type: str
    primary_agent: str | None
    workflow_status: str
    supporting_agents: tuple[str, ...] = ()
    required_tools: tuple[str, ...] = ()
    allowed_tools: tuple[str, ...] | None = None
    required_evidence_ids: tuple[str, ...] = ()
    expected_skills: Mapping[str, str | None] = field(default_factory=dict)
    expected_tool_arguments: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    expected_side_effect: str | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> 'ExecutionCase':
        allowed = set(cls.__dataclass_fields__)
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f'Unsupported execution case fields: {sorted(unknown)}')
        fields = dict(value)
        for name in ('supporting_agents', 'required_tools', 'required_evidence_ids'):
            fields[name] = tuple(fields.get(name, ()))
        if 'allowed_tools' in fields and fields['allowed_tools'] is not None:
            fields['allowed_tools'] = tuple(fields['allowed_tools'])
        case = cls(**fields)
        if not case.case_id or not case.message or not case.task_type or not case.workflow_status:
            raise ValueError('Execution case identity, message, task type and status are required')
        return case


def evaluate_execution(case: ExecutionCase, trace: RequestTrace | Mapping[str, Any]) -> dict[str, Any]:
    payload = trace.to_dict() if isinstance(trace, RequestTrace) else dict(trace)
    understanding = payload.get('task_understanding') or {}
    routing = payload.get('routing') or {}
    execution = payload.get('execution') or {}
    calls = execution.get('tool_calls') or []
    evidence = payload.get('evidence') or []
    effects = payload.get('side_effects') or []
    final = payload.get('final') or {}
    failures: set[str] = set()

    if understanding.get('task_type') != case.task_type:
        failures.add('UNDERSTANDING_FAILURE')
    if (routing.get('primary_agent') != case.primary_agent
            or tuple(routing.get('supporting_agents') or ()) != case.supporting_agents):
        failures.add('ROUTING_FAILURE')
    selected = {agent.get('agent_type'): (agent.get('selected_skill') or {}).get('skill_id')
                for wave in execution.get('execution_waves') or [] for agent in wave.get('agents') or []}
    if any(selected.get(agent) != skill for agent, skill in case.expected_skills.items()):
        failures.add('SKILL_SELECTION_FAILURE')

    successful = {call.get('tool_name') for call in calls if call.get('success')}
    if set(case.required_tools) - successful:
        failures.add('MISSING_REQUIRED_TOOL')
    if case.allowed_tools is not None and any(call.get('tool_name') not in case.allowed_tools for call in calls):
        failures.add('UNNECESSARY_TOOL')
    for call in calls:
        agent = call.get('agent_type')
        if agent in {kind.value for kind in AgentType}:
            if call.get('tool_name') not in AGENTS[AgentType(agent)].tools:
                failures.add('FORBIDDEN_TOOL')
        elif agent != 'handoff':
            failures.add('FORBIDDEN_TOOL')
        expected = case.expected_tool_arguments.get(call.get('tool_name'))
        if expected is not None and any((call.get('input') or {}).get(key) != value
                                        for key, value in expected.items()):
            failures.add('TOOL_ARGUMENT_FAILURE')
        if not call.get('success'):
            failures.add('TOOL_EXECUTION_FAILURE')
    if set(case.required_evidence_ids) - {item.get('record_id') for item in evidence}:
        failures.add('MISSING_EVIDENCE')
    if case.expected_side_effect is None:
        if effects:
            failures.add('SIDE_EFFECT_FAILURE')
    elif not any(item.get('type') == case.expected_side_effect for item in effects):
        failures.add('SIDE_EFFECT_FAILURE')
    if final.get('workflow_status') != case.workflow_status:
        failures.add('FINAL_STATUS_FAILURE')
    completeness = validate_trace_completeness(payload)
    if not completeness['complete']:
        failures.add('TRACE_INCOMPLETE')
    labels = [label for label, _ in FAILURE_LAYERS if label in failures]
    return {'case_id': case.case_id, 'task_success': not labels,
            'failure_labels': labels,
            'earliest_failure_layer': next((layer for label, layer in FAILURE_LAYERS if label in failures), None),
            'required_tool_recall': (len(set(case.required_tools) & successful) / len(set(case.required_tools))
                                     if case.required_tools else None),
            'evidence_recall': (len(set(case.required_evidence_ids) & {item.get('record_id') for item in evidence}) /
                                len(set(case.required_evidence_ids)) if case.required_evidence_ids else None),
            'trace_complete': completeness['complete'],
            'trace_missing_fields': completeness['missing_fields'],
            'observed_status': final.get('workflow_status')}


def summarize_execution(results: list[Mapping[str, Any]]) -> dict[str, Any]:
    labels = Counter(label for row in results for label in row['failure_labels'])
    return {'total': len(results), 'passed': sum(bool(row['task_success']) for row in results),
            'task_success_rate': round(sum(bool(row['task_success']) for row in results)/len(results), 4)
            if results else None, 'failure_labels': dict(sorted(labels.items())), 'cases': list(results),
            'clinical_validity_evaluated': False}
