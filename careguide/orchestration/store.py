"""Authoritative workflow state with identity and optimistic-version checks."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
import uuid
import re
import hashlib

from pydantic import Field, TypeAdapter

from careguide.action_store import ActionStore
from careguide.errors import ErrorCode, Failure
from careguide.orchestration.bindings import CRITICAL_FIELDS, explicit_value

from careguide.orchestration.contracts import (
    ACTION_ARGUMENTS, Contract, PlanChange, PlanProposal, PREPARE_ACTIONS, TaskRecord,
    TaskStatus, TRANSITIONS, fanout_authorized, validate_plan)


def utc_now():
    return datetime.now(timezone.utc)


class VersionConflict(ValueError):
    pass


class WorkflowExpired(ValueError):
    pass


class WorkflowState(Contract):
    id: str
    actor: str
    conversation: str
    goal: str
    version: int = 1
    revision: int = 0
    tasks: dict[str, TaskRecord]
    source_messages: list[dict] = Field(default_factory=list)
    preference_snapshot: dict = Field(default_factory=dict)
    status: str = 'ready'
    created_at: str
    expires_at: str
    events: list[dict] = Field(default_factory=list)
    archived_results: list[dict] = Field(default_factory=list)
    active_run: str | None = None
    lease_expires_at: str | None = None
    budget_id: str | None = None


class WorkflowStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._active_connection = ContextVar('workflow_transaction', default=None)
        with self.connect() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS orchestration_workflows (
                id TEXT PRIMARY KEY, actor TEXT NOT NULL, conversation TEXT NOT NULL,
                version INTEGER NOT NULL, revision INTEGER NOT NULL,
                created_at TEXT NOT NULL, payload TEXT NOT NULL)''')
            db.execute('CREATE INDEX IF NOT EXISTS workflow_owner ON '
                       'orchestration_workflows(actor,conversation,created_at)')

    @contextmanager
    def connect(self):
        active = self._active_connection.get()
        if active is not None:
            yield active
            return
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _begin(db):
        if not db.in_transaction:
            db.execute('BEGIN IMMEDIATE')

    @staticmethod
    def _load(db, identity, actor, conversation):
        row = db.execute('SELECT payload FROM orchestration_workflows '
            'WHERE id=? AND actor=? AND conversation=?', (identity, actor, conversation)).fetchone()
        if row is None:
            raise LookupError('Workflow not found')
        return WorkflowState.model_validate_json(row['payload'])

    @staticmethod
    def _write(db, state):
        state.revision += 1
        db.execute('UPDATE orchestration_workflows SET version=?,revision=?,payload=? WHERE id=?',
            (state.version, state.revision, state.model_dump_json(), state.id))

    def create(self, actor, conversation, proposal: PlanProposal, *, source_text,
               ttl_seconds=1800, identity=None, source_at=None):
        proposal = validate_plan(proposal, source_text)
        now = utc_now()
        state = WorkflowState(id=identity or uuid.uuid4().hex, actor=actor, conversation=conversation,
            goal=proposal.goal, tasks={task.id: TaskRecord(spec=task) for task in proposal.tasks},
            source_messages=[{'id': uuid.uuid4().hex, 'content': source_text, 'at': source_at or now.isoformat()}],
            created_at=now.isoformat(), expires_at=(now+timedelta(seconds=ttl_seconds)).isoformat())
        with self.connect() as db:
            db.execute('INSERT INTO orchestration_workflows VALUES (?,?,?,?,?,?,?)',
                (state.id, actor, conversation, state.version, state.revision,
                 state.created_at, state.model_dump_json()))
        return state

    def get(self, identity, actor, conversation):
        with self.connect() as db:
            return self._load(db, identity, actor, conversation)

    def create_pending(self, actor, conversation, source_text, *, ttl_seconds=1800, source_at=None,
                       identity=None, source_message_id=None):
        """Allocate identity before classification/planning so their costs persist."""
        now = utc_now()
        state = WorkflowState(id=identity or uuid.uuid4().hex, actor=actor, conversation=conversation,
            goal=source_text, tasks={}, status='planning',
            source_messages=[{'id': source_message_id or uuid.uuid4().hex, 'content': source_text,
                              'at': source_at or now.isoformat()}],
            created_at=now.isoformat(), expires_at=(now+timedelta(seconds=ttl_seconds)).isoformat())
        with self.connect() as db:
            db.execute('INSERT INTO orchestration_workflows VALUES (?,?,?,?,?,?,?)',
                (state.id, actor, conversation, state.version, state.revision,
                 state.created_at, state.model_dump_json()))
        return state

    def initialize(self, state, proposal, events):
        proposal = validate_plan(proposal, '\n'.join(s['content'] for s in state.source_messages),
                                 source_messages=state.source_messages)
        def operation(current):
            if current.tasks:
                raise VersionConflict('Plan already initialized')
            current.goal = proposal.goal
            current.tasks = {task.id: TaskRecord(spec=task) for task in proposal.tasks}
            current.events.extend(events)
            current.status = 'ready'
        return self.mutate(state.id, state.actor, state.conversation, operation,
                           plan_version=state.version, run_id=state.active_run)

    def expire_waits(self, identity, actor, conversation):
        with self.connect() as db:
            self._begin(db)
            state = self._load(db, identity, actor, conversation)
            before = state.model_dump_json()
            now = utc_now().isoformat()
            if state.expires_at <= now:
                state.status = 'expired'
            for task in state.tasks.values():
                if task.status in {TaskStatus.WAITING_INPUT, TaskStatus.WAITING_CONFIRMATION} and (
                        state.status == 'expired' or task.expires_at and task.expires_at <= now):
                    task.transition(TaskStatus.EXPIRED, at=now, reason='wait_expired')
            if state.model_dump_json() != before:
                self._write(db, state)
            self._expire_invalid_actions(db, state)
            return state

    def latest(self, actor, conversation):
        with self.connect() as db:
            row = db.execute('SELECT payload FROM orchestration_workflows WHERE actor=? '
                'AND conversation=? ORDER BY created_at DESC LIMIT 1', (actor, conversation)).fetchone()
            return WorkflowState.model_validate_json(row['payload']) if row else None

    @staticmethod
    def _sync_actions_in(db, state):
        table = db.execute("SELECT 1 FROM sqlite_master WHERE name='pending_actions'").fetchone()
        if table is None:
            return
        for task in state.tasks.values():
            if not task.action_id or task.status not in {TaskStatus.WAITING_CONFIRMATION, TaskStatus.RECONCILING}:
                continue
            row = db.execute('SELECT payload FROM pending_actions WHERE id=? AND actor=? AND conversation=?',
                (task.action_id, state.actor, state.conversation)).fetchone()
            if row is None:
                continue
            action = json.loads(row['payload'])
            if any(value != (state.actor if key == 'patient_id' else task.arguments.get(key))
                   for key, value in action['arguments'].items()):
                continue
            status = action['status']
            target = {'completed': TaskStatus.SUCCEEDED, 'failed': TaskStatus.FAILED,
                      'expired': TaskStatus.EXPIRED, 'rejected': TaskStatus.CANCELLED,
                      'executing': TaskStatus.RECONCILING, 'reconciling': TaskStatus.RECONCILING}.get(status)
            if target is None or target == task.status:
                continue
            task.transition(target, at=utc_now().isoformat(), reason='business_action_reconciliation')
            task.result = {**action, 'evidence': [], 'simulation': True}
            task.expires_at = None
            task.failure = (Failure(ErrorCode.INVALID_PARAMETERS, 'business_result', task.spec.id).payload()
                            if target == TaskStatus.FAILED else None)
            state.events.append({'event': 'business_action_reconciled', 'task_id': task.spec.id,
                'action_id': action['id'], 'status': status, 'at': utc_now().isoformat()})

    def sync_actions(self, identity, actor, conversation):
        with self.connect() as db:
            self._begin(db)
            state = self._load(db, identity, actor, conversation)
            self._sync_actions_in(db, state)
            from careguide.orchestration.replies import workflow_status
            state.status = workflow_status(state)
            self._write(db, state)
            return state

    @staticmethod
    def _expire_invalid_actions(db, state):
        if db.execute("SELECT 1 FROM sqlite_master WHERE name='pending_actions'").fetchone() is None:
            return
        rows = db.execute("SELECT payload FROM pending_actions WHERE actor=? AND conversation=? "
                          "AND status='pending_confirmation'", (state.actor, state.conversation)).fetchall()
        for row in rows:
            action = json.loads(row['payload'])
            binding = action.get('workflow_binding')
            if not binding or binding.get('workflow_id') != state.id:
                continue
            task = state.tasks.get(binding.get('task_id'))
            if (task and task.status == TaskStatus.RUNNING and task.revision == binding.get('task_revision')
                    and (task.run_id or state.active_run) == binding.get('run_id')
                    and state.expires_at > utc_now().isoformat()
                    and action['expires_at'] > utc_now().isoformat()):
                # Preparation has returned from the business service but its
                # worker may still be settling the shared budget before commit.
                continue
            if ActionStore._binding_valid(db, binding, state.actor, state.conversation,
                    action['type'], action['arguments'], phase='confirm', action_id=action['id']):
                continue
            action['status'] = 'expired'
            action['events'].append({'status': 'expired', 'at': utc_now().isoformat(),
                                      'reason': 'workflow_task_changed'})
            db.execute('UPDATE pending_actions SET status=?,payload=? WHERE id=?',
                       ('expired', json.dumps(action, ensure_ascii=False), action['id']))

    def save(self, state: WorkflowState, *, expected_revision):
        with self.connect() as db:
            self._begin(db)
            current = self._load(db, state.id, state.actor, state.conversation)
            if current.revision != expected_revision or current.version != state.version:
                raise VersionConflict('Workflow changed while executing')
            self._write(db, state)
        return state

    def mutate(self, identity, actor, conversation, operation, *, plan_version, run_id, invalidate_actions=False):
        """Run a short, server-owned mutation while holding the write lock."""
        with self.connect() as db:
            self._begin(db)
            state = self._load(db, identity, actor, conversation)
            if state.version != plan_version or state.active_run != run_id:
                raise VersionConflict('Workflow changed while executing')
            if state.expires_at <= utc_now().isoformat():
                raise WorkflowExpired('Workflow expired')
            operation(state)
            self._write(db, state)
            if invalidate_actions:
                self._expire_invalid_actions(db, state)
            return state

    def begin_run(self, identity, actor, conversation, *, expected_version, lease_seconds=45):
        with self.connect() as db:
            self._begin(db)
            state = self._load(db, identity, actor, conversation)
            if state.version != expected_version:
                raise VersionConflict('Stale plan version')
            if state.expires_at <= utc_now().isoformat():
                raise WorkflowExpired('Workflow expired')
            if (state.active_run and state.lease_expires_at
                    and state.lease_expires_at > utc_now().isoformat()):
                raise VersionConflict('Workflow execution already active')
            # An expired lease cannot authorize further commits or prepare calls.
            # Keep costs/reservations charged: a timeout does not stop the remote
            # operation. Reads/prepare may resume; confirmed writes never rerun.
            for task in state.tasks.values():
                if task.status == TaskStatus.RUNNING and (
                        not task.lease_expires_at or task.lease_expires_at <= utc_now().isoformat()):
                    task.transition(TaskStatus.READY, at=utc_now().isoformat(), reason='expired_run_lease')
                    task.run_id, task.run_plan_version, task.lease_expires_at = None, None, None
            state.active_run = uuid.uuid4().hex
            state.lease_expires_at = (utc_now()+timedelta(seconds=lease_seconds)).isoformat()
            self._write(db, state)
            return state

    def end_run(self, identity, actor, conversation, run_id, *, status):
        with self.connect() as db:
            self._begin(db)
            state = self._load(db, identity, actor, conversation)
            if state.active_run != run_id:
                return state
            state.active_run, state.lease_expires_at = None, None
            state.status = status
            self._write(db, state)
            return state

    def answer(self, identity, actor, conversation, task_id, values, *,
               plan_version, task_revision, event_id=None):
        """Bind one user's reply to the exact current question and candidate set."""
        with self.connect() as db:
            self._begin(db)
            state = self._load(db, identity, actor, conversation)
            task = state.tasks.get(task_id)
            if state.version != plan_version or task is None or task.revision != task_revision:
                raise VersionConflict('Reply belongs to an old plan or task revision')
            if state.expires_at <= utc_now().isoformat() or (
                    task.expires_at and task.expires_at <= utc_now().isoformat()):
                raise WorkflowExpired('Waiting task expired; start a fresh query')
            fingerprint = hashlib.sha256(json.dumps([task_id, plan_version,
                task_revision, values], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            if event_id:
                previous = next((event for event in state.events if event.get('input_event_id') == event_id), None)
                if previous:
                    if previous['fingerprint'] != fingerprint:
                        raise VersionConflict('Input event reused with different values')
                    return state
            if task.status != TaskStatus.WAITING_INPUT:
                raise VersionConflict('Task is not waiting for input')
            fields = ACTION_ARGUMENTS[task.spec.action].model_fields
            if not values or set(values)-set(task.missing_fields):
                raise ValueError('Reply must address the selected task missing fields')
            event_id = event_id or uuid.uuid4().hex
            for key, value in values.items():
                TypeAdapter(fields[key].rebuild_annotation()).validate_python(value, strict=True)
                if value in (None, ''):
                    raise ValueError('An empty answer is not confirmation')
                reference = task.spec.inputs.get(key)
                kind = 'user_reply'
                if reference is not None:
                    if reference.mode != 'selected':
                        raise ValueError('This input is not a user-selection binding')
                    parent = state.tasks[reference.task_id]
                    if (parent.status != TaskStatus.SUCCEEDED or parent.result is None
                            or (reference.result_version and reference.result_version != parent.revision)
                            or (parent.result_valid_until and parent.result_valid_until <= utc_now().isoformat())):
                        raise WorkflowExpired('Candidate results expired or changed')
                    candidates = parent.result
                    for path in reference.path:
                        candidates = candidates[path]
                    if value not in [item.get(reference.item_field) for item in candidates]:
                        raise ValueError('Selection is not in the current candidate set')
                    kind = 'user_selection'
                task.arguments[key] = value
                task.origins[key] = {'kind': kind, 'confirmed': True,
                                     'message_id': event_id, 'at': utc_now().isoformat()}
            task.missing_fields = [key for key in task.missing_fields if key not in values]
            if not task.missing_fields:
                task.transition(TaskStatus.READY, at=utc_now().isoformat(), reason='user_reply')
                task.failure, task.expires_at, task.options = None, None, []
            state.source_messages.append({'id': event_id, 'content': json.dumps(values, ensure_ascii=False),
                                          'at': utc_now().isoformat()})
            state.events.append({'event': 'user_reply', 'task_id': task_id,
                'input_event_id': event_id, 'fingerprint': fingerprint,
                'version': state.version, 'task_revision': task.revision})
            self._write(db, state)
            return state

    def commit_result(self, identity, actor, conversation, task_id, result, status,
                      *, plan_version, task_revision, run_id, failure=None, missing_fields=None,
                      options=None, action_id=None, arguments=None, origins=None,
                      observed_at=None, valid_until=None, traces=None, wait_seconds=300, worker_result=None):
        with self.connect() as db:
            self._begin(db)
            state = self._load(db, identity, actor, conversation)
            task = state.tasks.get(task_id)
            dispatch_owned = task is not None and (
                task.run_id == run_id and task.run_plan_version == plan_version
                if task.run_id else state.version == plan_version and state.active_run == run_id)
            lease_expiry = (task.lease_expires_at if task and task.run_id else state.lease_expires_at)
            if (not dispatch_owned or task.revision != task_revision or task.status != TaskStatus.RUNNING
                    or state.expires_at <= utc_now().isoformat()
                    or (lease_expiry and lease_expiry <= utc_now().isoformat())):
                state.archived_results.append({'task_id': task_id, 'plan_version': plan_version,
                    'task_revision': task_revision, 'result': result, 'discarded': True})
                state.events.append({'event': 'late_result_discarded', 'task_id': task_id,
                                     'plan_version': plan_version, 'at': utc_now().isoformat()})
                self._write(db, state)
                return False
            task.result, task.failure = result, failure
            task.worker_result = worker_result
            task.missing_fields, task.options = missing_fields or [], options or []
            task.action_id = action_id
            if arguments is not None:
                task.arguments = arguments
            if origins is not None:
                task.origins = origins
            task.result_observed_at, task.result_valid_until = observed_at, valid_until
            if TaskStatus(status) in {TaskStatus.WAITING_INPUT, TaskStatus.WAITING_CONFIRMATION}:
                task.expires_at = min(state.expires_at,
                    result.get('expires_at') if TaskStatus(status) == TaskStatus.WAITING_CONFIRMATION
                    and result.get('expires_at') else (utc_now()+timedelta(seconds=wait_seconds)).isoformat())
            task.transition(TaskStatus(status), at=utc_now().isoformat())
            state.events.append({'event': 'task_result', 'task_id': task_id,
                'version': state.version, 'revision': task.revision, 'status': task.status.value,
                'run_id': run_id, 'calls': traces or []})
            from careguide.orchestration.replies import workflow_status
            state.status = workflow_status(state)
            self._write(db, state)
            return True

    def apply_change(self, identity, actor, conversation, change: PlanChange, *, user_text,
                     authorized_entities=None, source_at=None):
        with self.connect() as db:
            self._begin(db)
            state = self._load(db, identity, actor, conversation)
            if state.version != change.base_version:
                raise VersionConflict('Stale base plan version')
            if state.expires_at <= utc_now().isoformat():
                raise WorkflowExpired('Workflow expired')
            self._sync_actions_in(db, state)
            updates = {task.id: task for task in change.update}
            additions = {task.id: task for task in change.add}
            message_id, message_at = uuid.uuid4().hex, source_at or utc_now().isoformat()
            date_anchor = datetime.fromisoformat(message_at).astimezone(timezone(timedelta(hours=8))).date()
            changed_parameters = {}
            cancelled = set(change.cancel)
            if cancelled and not re.search(r'取消|不用|不查|停止|放弃|删除任务|删除分支', user_text):
                raise PermissionError('Cancelling a user goal requires explicit authorization')
            if len(updates) != len(change.update) or len(additions) != len(change.add):
                raise ValueError('Duplicate changed task IDs')
            if set(updates) & cancelled or set(additions) & (set(state.tasks) | cancelled):
                raise ValueError('Conflicting task changes')
            if (set(updates) | cancelled) - state.tasks.keys():
                raise ValueError('Unknown changed task')
            # A completed write remains an immutable business fact. Changing
            # or cancelling it requires a new prepare/confirm action.
            for task_id in set(updates) | cancelled:
                task = state.tasks[task_id]
                if task.status in {TaskStatus.CANCELLED, TaskStatus.EXPIRED}:
                    raise ValueError('Inactive task requires a new task ID')
                if task.spec.action in PREPARE_ACTIONS and task.status in {TaskStatus.SUCCEEDED, TaskStatus.RECONCILING}:
                    raise PermissionError('Completed write requires a new confirmed action')
                if task_id in updates:
                    proposed = updates[task_id]
                    for key, reference in task.spec.inputs.items():
                        if key not in CRITICAL_FIELDS:
                            continue
                        replacement = proposed.inputs.get(key)
                        if replacement and replacement.mode == 'each' and reference.mode != 'each' and not fanout_authorized(
                                proposed.action, key, user_text):
                            raise PermissionError('Expanding a confirmed selection requires user authorization')
                        if replacement is None:
                            after = proposed.arguments.get(key)
                            same_confirmed_value = (after is not None and after == task.arguments.get(key)
                                                    and task.origins.get(key, {}).get('confirmed'))
                            clear = {'practitioner_id': r'不限医生|所有医生|全部医生',
                                     'department': r'不限科室|所有科室|全部科室',
                                     'date': r'不限日期|所有日期|任何日期'}.get(key)
                            if not same_confirmed_value and not (after is not None and explicit_value(
                                    key, after, user_text, authorized_entities, today=date_anchor)) and not (
                                        after is None and clear and re.search(clear, user_text)):
                                raise PermissionError('Removing an input scope requires user authorization')
                    for key in CRITICAL_FIELDS & (task.spec.arguments.keys() | proposed.arguments.keys()):
                        before, after = task.spec.arguments.get(key), proposed.arguments.get(key)
                        if before == after or key in proposed.inputs:
                            continue
                        clear_scope = {
                            'date': r'不限日期|所有日期|任何日期',
                            'department': r'不限科室|所有科室|全部科室',
                            'practitioner_id': r'不限医生|所有医生|全部医生',
                        }.get(key)
                        explicitly_cleared = after is None and clear_scope and bool(re.search(clear_scope, user_text))
                        if not explicitly_cleared and not explicit_value(
                                key, after, user_text, authorized_entities, today=date_anchor):
                            raise PermissionError('Scope change requires explicit user authorization')
                        changed_parameters.setdefault(task_id, {})[key] = after
            for task_id in changed_parameters:
                # A valid scope change is tied to the new user message, even
                # when the model copied the previous request fragment.
                updates[task_id] = updates[task_id].model_copy(update={'fragment': user_text}, deep=True)
            affected = {key for key, spec in updates.items()
                        if spec.model_dump() != state.tasks[key].spec.model_dump()} | cancelled
            while True:
                descendants = {key for key, task in state.tasks.items()
                    if set(task.spec.depends_on) & affected}
                expanded = affected | descendants
                if expanded == affected:
                    break
                affected = expanded
            cancel_closure = set(cancelled)
            while True:
                expanded = cancel_closure | {key for key, task in state.tasks.items()
                    if set(task.spec.depends_on) & cancel_closure and key not in updates}
                if expanded == cancel_closure:
                    break
                cancel_closure = expanded
            immutable = {key for key in affected if state.tasks[key].spec.action in PREPARE_ACTIONS
                         and state.tasks[key].status in {TaskStatus.SUCCEEDED, TaskStatus.RECONCILING}}
            for key in immutable:
                task = state.tasks[key]
                state.archived_results.append({'task_id': key, 'task_revision': task.revision,
                    'record': task.model_dump(mode='json'), 'reason': 'confirmed_write_preserved'})
                task.spec = task.spec.model_copy(update={'depends_on': [], 'inputs': {},
                                                        'arguments': dict(task.arguments)}, deep=True)
            affected -= immutable
            cancel_closure -= immutable
            # Unaffected in-flight work retains its task revision and dispatch
            # ownership. Only affected old results are rejected; a plan version
            # change alone must not duplicate an unrelated running query.
            specs = {key: updates.get(key, task.spec) for key, task in state.tasks.items()
                     if key not in cancel_closure and task.status not in {TaskStatus.CANCELLED, TaskStatus.EXPIRED}}
            specs.update(additions)
            source = '\n'.join(message['content'] for message in state.source_messages)+'\n'+user_text
            if specs:
                validate_plan(PlanProposal(goal=state.goal, tasks=list(specs.values())), source,
                    source_messages=state.source_messages+[
                        {'id': message_id, 'content': user_text, 'at': message_at}])
            if not affected and not additions:
                raise ValueError('No progress: change repeats the current plan')
            next_revisions = {key: task.revision+int(key in affected)
                              for key, task in state.tasks.items()}
            next_revisions.update({key: 1 for key in additions})
            for task_id in affected:
                old = state.tasks[task_id]
                state.archived_results.append({'task_id': task_id, 'plan_version': state.version,
                    'task_revision': old.revision, 'record': old.model_dump(mode='json')})
                if task_id in cancel_closure:
                    if TaskStatus.CANCELLED in TRANSITIONS.get(old.status, set()):
                        old.transition(TaskStatus.CANCELLED, at=utc_now().isoformat(), reason=change.reason)
                    else:
                        state.tasks[task_id] = TaskRecord(spec=old.spec,
                            revision=old.revision+1, status=TaskStatus.CANCELLED,
                            result=old.result, history=[*old.history,
                                {'event': 'cancelled', 'at': utc_now().isoformat(), 'reason': change.reason}])
                else:
                    spec = specs[task_id].model_copy(deep=True)
                    for reference in spec.inputs.values():
                        reference.result_version = next_revisions[reference.task_id]
                    state.tasks[task_id] = TaskRecord(spec=spec, revision=old.revision+1,
                        history=[{'event': 'invalidated', 'previous_revision': old.revision,
                                  'reason': change.reason, 'at': utc_now().isoformat()}])
                    record = state.tasks[task_id]
                    for key, value in old.arguments.items():
                        previous_origin = old.origins.get(key, {})
                        reference = spec.inputs.get(key)
                        preserved_binding = (reference is not None and key in old.spec.inputs
                            and reference.model_dump(exclude={'result_version'}) == old.spec.inputs[key].model_dump(exclude={'result_version'})
                            and reference.task_id not in affected)
                        if not previous_origin.get('confirmed') or not (
                                preserved_binding or key not in spec.inputs and (
                                    spec.arguments.get(key) == value or key not in old.spec.inputs
                                    and old.spec.arguments.get(key) == spec.arguments.get(key))):
                            continue
                        record.arguments[key] = value
                        record.origins[key] = {**previous_origin,
                            'kind': previous_origin['kind'] if preserved_binding else 'retained_scope',
                            'retained_from': {'plan_version': state.version, 'task_revision': old.revision}}
                    for key, value in changed_parameters.get(task_id, {}).items():
                        if value is not None:
                            record.arguments[key] = value
                            record.origins[key] = {'kind': 'user_change', 'confirmed': True,
                                'source_message_id': message_id, 'at': message_at}
            for task_id, spec in additions.items():
                spec = spec.model_copy(deep=True)
                for reference in spec.inputs.values():
                    reference.result_version = next_revisions[reference.task_id]
                state.tasks[task_id] = TaskRecord(spec=spec)
            state.version += 1
            state.active_run = None
            state.lease_expires_at = None
            state.status = 'ready' if specs else 'cancelled'
            state.source_messages.append({'id': message_id, 'content': user_text, 'at': message_at})
            state.events.append({'event': 'plan_changed', 'base_version': change.base_version,
                'version': state.version, 'affected': sorted(affected), 'reason': change.reason})
            self._write(db, state)
            self._expire_invalid_actions(db, state)
            return state

    def apply_decision(self, identity, actor, conversation, decision, *, user_text,
                       sources=(), source_at=None, authorized_entities=None):
        """Apply a whole input/change package in one short transaction, or none."""
        with self.connect() as db:
            self._begin(db)
            token = self._active_connection.set(db)
            try:
                state = self._load(db, identity, actor, conversation)
                if state.version != decision.base_version or decision.workflow_id != identity:
                    raise VersionConflict('Stale decision')
                if state.expires_at <= utc_now().isoformat():
                    raise WorkflowExpired('Workflow expired')
                known = {s['id'] for s in state.source_messages}
                state.source_messages.extend(dict(s) for s in sources if s['id'] not in known)
                self._write(db, state)
                for index, answer in enumerate(decision.answers):
                    state = self.answer(identity, actor, conversation, answer.task_id, answer.values,
                        plan_version=decision.base_version, task_revision=answer.task_revision,
                        event_id=f"{sources[-1]['id'] if sources else state.id}:answer:{index}")
                change = decision.change
                if decision.kind == 'cancel':
                    change = PlanChange(base_version=decision.base_version,
                                        reason='用户明确取消', cancel=decision.cancel)
                if change is not None:
                    state = self.apply_change(identity, actor, conversation, change,
                        user_text=user_text, source_at=source_at, authorized_entities=authorized_entities)
                return state
            finally:
                self._active_connection.reset(token)
