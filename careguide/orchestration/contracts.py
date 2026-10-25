"""Task DAG contracts; model proposals never grant execution authority."""
from __future__ import annotations

from enum import Enum
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr

from careguide.domain import AgentType, require_tool
from careguide.tools import ARGUMENT_TYPES, Arguments


class Contract(BaseModel):
    model_config = ConfigDict(extra='forbid')


class DoctorArguments(Arguments):
    department: StrictStr | None = Field(default=None, max_length=100)


class SlotArguments(DoctorArguments):
    practitioner_id: StrictStr | None = Field(default=None, max_length=100)
    date: StrictStr | None = Field(default=None, max_length=40)
    time_period: Literal['morning', 'afternoon', 'evening'] | None = None


class AppointmentArguments(Arguments):
    appointment_id: StrictStr | None = Field(default=None, max_length=100)


class PrepareArguments(AppointmentArguments):
    slot_id: StrictStr | None = Field(default=None, max_length=100)


class CreateArguments(Arguments):
    slot_id: StrictStr | None = Field(default=None, max_length=100)


HOSPITAL_ACTIONS = {
    'search_doctors': (DoctorArguments, 'doctor_search'),
    'list_available_slots': (SlotArguments, 'slot_search'),
    'get_appointment': (AppointmentArguments, 'appointment_status'),
    'prepare_appointment': (CreateArguments, 'appointment_create'),
    'prepare_reschedule': (PrepareArguments, 'appointment_reschedule'),
    'prepare_cancellation': (AppointmentArguments, 'appointment_cancel'),
}
ACTION_ARGUMENTS = {**ARGUMENT_TYPES, **{key: value[0] for key, value in HOSPITAL_ACTIONS.items()}}
PREPARE_ACTIONS = frozenset({'prepare_appointment', 'prepare_reschedule', 'prepare_cancellation'})


def fanout_authorized(action, parameter, text):
    if action == 'list_available_slots' and parameter == 'practitioner_id':
        return bool(re.search(r'(?:全部|所有|每位|每个)'
            r'(?:(?:返回|查到|找到|符合条件|可选|本次|匹配)的?){0,2}'
            r'(?:[\u4e00-\u9fff]{1,8}科)?(?:医生|医师)[^，。；]{0,20}(?:号源|可预约)', text))
    return bool(re.search(r'全部|所有|每位|每个|分别查', text))


class InputReference(Contract):
    task_id: str = Field(min_length=1, max_length=64)
    path: list[str] = Field(default_factory=lambda: ['items'], max_length=6)
    mode: Literal['value', 'selected', 'each'] = 'value'
    item_field: str | None = Field(default=None, max_length=64)
    result_version: int | None = Field(default=None, ge=1)


class SourceFragment(Contract):
    message_id: str = Field(min_length=1, max_length=100)
    quote: str = Field(min_length=1, max_length=1000)


class TaskSpec(Contract):
    id: str = Field(pattern=r'^[a-zA-Z][a-zA-Z0-9_-]{0,63}$')
    fragment: str = Field(min_length=1, max_length=4000)
    action: str = Field(min_length=1, max_length=64)
    agent: AgentType
    arguments: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list, max_length=12)
    inputs: dict[str, InputReference] = Field(default_factory=dict)
    completion: Literal['source_evidence', 'items', 'prepared', 'business_record']
    expected_output: list[str] = Field(default_factory=list, max_length=12)
    max_attempts: int = Field(default=2, ge=1, le=3)
    source_fragments: list[SourceFragment] = Field(default_factory=list, max_length=8)


class PlanProposal(Contract):
    goal: str = Field(min_length=1, max_length=4000)
    tasks: list[TaskSpec] = Field(min_length=1, max_length=12)


class PlanChange(Contract):
    base_version: int = Field(ge=1)
    reason: str = Field(min_length=1, max_length=1000)
    add: list[TaskSpec] = Field(default_factory=list, max_length=12)
    update: list[TaskSpec] = Field(default_factory=list, max_length=12)
    cancel: list[str] = Field(default_factory=list, max_length=12)


class TaskStatus(str, Enum):
    WAITING_DEPENDENCIES = 'waiting_dependencies'
    READY = 'ready'
    RUNNING = 'running'
    SUCCEEDED = 'succeeded'
    FAILED = 'failed'
    WAITING_INPUT = 'waiting_input'
    WAITING_CONFIRMATION = 'waiting_confirmation'
    BLOCKED = 'blocked'
    CANCELLED = 'cancelled'
    INVALIDATED = 'invalidated'
    RECONCILING = 'reconciling'
    EXPIRED = 'expired'


TRANSITIONS = {
    TaskStatus.WAITING_DEPENDENCIES: {TaskStatus.READY, TaskStatus.BLOCKED,
        TaskStatus.CANCELLED, TaskStatus.INVALIDATED, TaskStatus.EXPIRED},
    TaskStatus.READY: {TaskStatus.RUNNING, TaskStatus.WAITING_INPUT,
        TaskStatus.CANCELLED, TaskStatus.INVALIDATED, TaskStatus.EXPIRED, TaskStatus.FAILED, TaskStatus.BLOCKED},
    TaskStatus.RUNNING: {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.WAITING_INPUT,
        TaskStatus.WAITING_CONFIRMATION, TaskStatus.READY, TaskStatus.RECONCILING,
        TaskStatus.CANCELLED, TaskStatus.INVALIDATED},
    TaskStatus.WAITING_INPUT: {TaskStatus.READY, TaskStatus.EXPIRED,
        TaskStatus.CANCELLED, TaskStatus.INVALIDATED},
    TaskStatus.WAITING_CONFIRMATION: {TaskStatus.SUCCEEDED, TaskStatus.FAILED,
        TaskStatus.EXPIRED, TaskStatus.RECONCILING, TaskStatus.CANCELLED, TaskStatus.INVALIDATED},
    TaskStatus.RECONCILING: {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.INVALIDATED},
    TaskStatus.SUCCEEDED: {TaskStatus.INVALIDATED},
    TaskStatus.FAILED: {TaskStatus.READY, TaskStatus.INVALIDATED, TaskStatus.CANCELLED},
    TaskStatus.BLOCKED: {TaskStatus.READY, TaskStatus.INVALIDATED, TaskStatus.CANCELLED},
}


class TaskRecord(Contract):
    spec: TaskSpec
    revision: int = Field(default=1, ge=1)
    status: TaskStatus = TaskStatus.WAITING_DEPENDENCIES
    attempts: int = 0
    run_id: str | None = None
    run_plan_version: int | None = None
    lease_expires_at: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    origins: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    failure: dict[str, Any] | None = None
    history: list[dict[str, Any]] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    options: list[dict[str, Any]] = Field(default_factory=list)
    action_id: str | None = None
    expires_at: str | None = None
    result_observed_at: str | None = None
    result_valid_until: str | None = None
    worker_result: dict | None = None

    def transition(self, status: TaskStatus, *, at: str, reason: str = ''):
        status = TaskStatus(status)
        if status not in TRANSITIONS.get(self.status, set()):
            raise ValueError(f'Invalid task transition: {self.status.value} -> {status.value}')
        self.history.append({'from': self.status.value, 'to': status.value,
                             'at': at, 'reason': reason, 'revision': self.revision})
        self.status = status


def validate_plan(proposal: PlanProposal, user_text: str, *, max_tasks=12, source_messages=()):
    """Validate authority, shapes and graph structure before any dispatch."""
    if len(proposal.tasks) > max_tasks:
        raise ValueError('Task count exceeds workflow budget')
    ids = [task.id for task in proposal.tasks]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate task IDs')
    lookup = {task.id: task for task in proposal.tasks}
    sources = {item['id']: item['content'] for item in source_messages if item.get('id')}
    for task in proposal.tasks:
        if task.action not in ACTION_ARGUMENTS:
            raise PermissionError('Action is not in the execution catalog')
        require_tool(task.agent, task.action)
        if task.fragment not in user_text:
            raise ValueError('Task fragment must refer to the original user request')
        for source in task.source_fragments:
            if source.message_id not in sources or source.quote not in sources[source.message_id]:
                raise ValueError('Unverified historical message source')
        if task.action in PREPARE_ACTIONS:
            expressions = {
                'prepare_appointment': r'预约|挂号',
                'prepare_reschedule': r'改期|改到|更换',
                'prepare_cancellation': r'取消|退号|退掉',
            }
            forbidden_create = task.action == 'prepare_appointment' and re.search(r'改期|退号|取消', task.fragment)
            if (not re.search(expressions[task.action], task.fragment) or forbidden_create
                    or re.search(r'(?:先别|别|不要)(?:.{0,4})(?:预约|挂号|改期|退号)|不需要(?:.{0,4})(?:预约|挂号)|'
                                 r'(?:不|无需|暂不|先不)(?:进行)?(?:预约|挂号)', task.fragment)):
                raise PermissionError('Appointment preparation requires an explicit user request')
        if len(set(task.depends_on)) != len(task.depends_on):
            raise ValueError('Duplicate dependencies')
        if any(identity not in lookup for identity in task.depends_on):
            raise ValueError('Unknown task dependency')
        schema = ACTION_ARGUMENTS[task.action]
        fields = schema.model_fields
        if set(task.arguments) - fields.keys() or set(task.inputs) - fields.keys():
            raise ValueError('Unknown action parameter')
        if set(task.arguments) & task.inputs.keys():
            raise ValueError('A parameter cannot have two input bindings')
        # Missing fields are a resumable user-input condition, not a malformed
        # plan. Validate every value that the model actually supplied.
        for key, value in task.arguments.items():
            from pydantic import TypeAdapter
            TypeAdapter(fields[key].rebuild_annotation()).validate_python(value, strict=True)
        for parameter, reference in task.inputs.items():
            if reference.task_id not in task.depends_on:
                raise ValueError('Input reference requires an explicit dependency')
            if any(not p or p.startswith('_') or p.isdigit() for p in reference.path):
                raise ValueError('Unsafe input path; candidate indices are not user selections')
            if reference.mode in {'selected', 'each'} and not reference.item_field:
                raise ValueError('Candidate binding requires an item field')
            if reference.item_field and not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,63}', reference.item_field):
                raise ValueError('Invalid candidate field')
            if reference.mode == 'each' and task.action in PREPARE_ACTIONS:
                raise PermissionError('Candidate fan-out cannot prepare bulk writes')
            if reference.mode == 'each' and not fanout_authorized(task.action, parameter, task.fragment):
                raise PermissionError('Querying every candidate requires user authorization')
        expected = ('prepared' if task.action in PREPARE_ACTIONS else
                    'items' if task.action in HOSPITAL_ACTIONS else 'source_evidence')
        if task.completion != expected:
            raise ValueError('Completion condition disagrees with action contract')
    visiting, visited = set(), set()
    def visit(identity):
        if identity in visiting:
            raise ValueError('Task dependency cycle')
        if identity in visited:
            return
        visiting.add(identity)
        for dependency in lookup[identity].depends_on:
            visit(dependency)
        visiting.remove(identity)
        visited.add(identity)
    for identity in ids:
        visit(identity)
    return proposal.model_copy(deep=True)
