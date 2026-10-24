"""Work-conserving task DAG scheduling inside the outer LangGraph workflow."""
from __future__ import annotations

import asyncio
from datetime import timedelta
import time

from pydantic import ValidationError

from careguide.errors import ErrorCode, Failure
from careguide.orchestration.bindings import InvalidResultReference
from careguide.orchestration.budget import BudgetExhausted
from careguide.orchestration.contracts import PREPARE_ACTIONS, TaskRecord, TaskStatus
from careguide.orchestration.store import VersionConflict, WorkflowExpired, utc_now
from careguide.orchestration.worker import TaskOutcome


class DAGScheduler:
    def __init__(self, store, worker, *, max_concurrency=4, wait_seconds=300):
        self.store, self.worker = store, worker
        self.max_concurrency, self.wait_seconds = max_concurrency, wait_seconds
        self.pending = set()

    def _mutate(self, state, operation, *, invalidate_actions=False):
        return self.store.mutate(state.id, state.actor, state.conversation, operation,
            plan_version=state.version, run_id=state.active_run, invalidate_actions=invalidate_actions)

    def _advance(self, state):
        if not any(task.status in {TaskStatus.WAITING_DEPENDENCIES, TaskStatus.READY}
                   for task in state.tasks.values()):
            return state, set()
        refresh = set()
        def operation(current):
            for task in current.tasks.values():
                if task.status not in {TaskStatus.WAITING_DEPENDENCIES, TaskStatus.READY}:
                    continue
                parents = [current.tasks[key] for key in task.spec.depends_on]
                if any(parent.status in {TaskStatus.FAILED, TaskStatus.BLOCKED,
                       TaskStatus.CANCELLED, TaskStatus.EXPIRED} for parent in parents):
                    task.failure = Failure(ErrorCode.DEPENDENCY_BLOCKED,
                        'dependency_blocked', task.spec.id).payload()
                    task.transition(TaskStatus.BLOCKED, at=utc_now().isoformat())
                    continue
                if any(parent.status != TaskStatus.SUCCEEDED for parent in parents):
                    continue
                try:
                    bound = self.worker.prepare(task, current)
                except InvalidResultReference as exc:
                    if 'expired' in str(exc) or 'revision changed' in str(exc):
                        refresh.add(exc.source_task)
                        continue
                    task.failure = Failure(ErrorCode.INVALID_PARAMETERS,
                        'input_binding', task.spec.id).payload()
                    task.transition(TaskStatus.BLOCKED, at=utc_now().isoformat())
                    continue
                except (ValidationError, ValueError):
                    task.failure = Failure(ErrorCode.INVALID_PARAMETERS,
                        'input_binding', task.spec.id).payload()
                    task.transition(TaskStatus.BLOCKED, at=utc_now().isoformat())
                    continue
                except PermissionError:
                    task.failure = Failure(ErrorCode.PERMISSION_DENIED,
                        'input_binding', task.spec.id).payload()
                    task.transition(TaskStatus.BLOCKED, at=utc_now().isoformat())
                    continue
                if task.status != TaskStatus.READY:
                    task.transition(TaskStatus.READY, at=utc_now().isoformat())
                if bound.missing_fields:
                    task.missing_fields, task.options = bound.missing_fields, bound.options
                    task.failure = Failure(ErrorCode.MISSING_INPUT, 'input_binding', task.spec.id,
                                           missing_fields=tuple(bound.missing_fields)).payload()
                    task.expires_at = min(current.expires_at,
                        (utc_now()+timedelta(seconds=self.wait_seconds)).isoformat())
                    for option in bound.options:
                        source_expiry = current.tasks[option['source_task']].result_valid_until
                        if source_expiry:
                            task.expires_at = min(task.expires_at, source_expiry)
                    task.transition(TaskStatus.WAITING_INPUT, at=utc_now().isoformat())
        state = self._mutate(state, operation)
        return state, refresh

    def _refresh(self, state, source_id):
        def operation(current):
            affected = {source_id}
            while True:
                expanded = affected | {key for key, task in current.tasks.items()
                    if set(task.spec.depends_on) & affected}
                if expanded == affected:
                    break
                affected = expanded
            revisions = {key: task.revision+int(key in affected) for key, task in current.tasks.items()}
            immutable = {key for key in affected if current.tasks[key].spec.action in PREPARE_ACTIONS
                         and current.tasks[key].status in {TaskStatus.SUCCEEDED, TaskStatus.RECONCILING}}
            for key in immutable:
                task = current.tasks[key]
                task.spec = task.spec.model_copy(update={'depends_on': [], 'inputs': {},
                                                        'arguments': dict(task.arguments)}, deep=True)
                revisions[key] = task.revision
            affected -= immutable
            for task_id in affected:
                old = current.tasks[task_id]
                current.archived_results.append({'task_id': task_id,
                    'task_revision': old.revision, 'record': old.model_dump(mode='json'),
                    'reason': 'result_revalidation'})
                spec = old.spec.model_copy(deep=True)
                for reference in spec.inputs.values():
                    reference.result_version = revisions[reference.task_id]
                current.tasks[task_id] = TaskRecord(spec=spec, revision=revisions[task_id])
            current.events.append({'event': 'source_refresh', 'source_task': source_id,
                                   'affected': sorted(affected)})
        return self._mutate(state, operation, invalidate_actions=True)

    def _start(self, state, task_id):
        def operation(current):
            task = current.tasks[task_id]
            if task.status != TaskStatus.READY:
                raise VersionConflict('Task is no longer ready')
            if task.attempts >= task.spec.max_attempts:
                task.failure = Failure(ErrorCode.BUDGET_EXHAUSTED,
                    'task_attempt_limit', task.spec.id).payload()
                task.transition(TaskStatus.FAILED, at=utc_now().isoformat())
                return
            task.attempts += 1
            task.run_id = current.active_run
            task.run_plan_version = current.version
            task.lease_expires_at = current.lease_expires_at
            task.failure = None
            task.transition(TaskStatus.RUNNING, at=utc_now().isoformat())
            current.events.append({'event': 'task_started', 'task_id': task_id,
                'version': current.version, 'revision': task.revision, 'at': utc_now().isoformat()})
        return self._mutate(state, operation)

    def _commit(self, state, task, outcome):
        return self.store.commit_result(state.id, state.actor, state.conversation,
            task.spec.id, outcome.data, outcome.status, plan_version=state.version,
            task_revision=task.revision, run_id=state.active_run, failure=outcome.failure,
            missing_fields=outcome.missing_fields, options=outcome.options,
            action_id=outcome.action_id, arguments=task.arguments, origins=task.origins,
            observed_at=outcome.observed_at, valid_until=outcome.valid_until,
            traces=outcome.traces, wait_seconds=self.wait_seconds)

    def _cancel_pending(self, state, running):
        for future, task in list(running.items()):
            future.cancel()
            self.pending.add(future)
            def late(done, record=task):
                self.pending.discard(done)
                if not done.cancelled() and done.exception() is None:
                    self._commit(state, record, done.result())
            future.add_done_callback(late)

    def _detach_pending(self, state, current, running):
        """Carry valid dispatches across versions; invalidate only changed tasks."""
        affected = {}
        for future, record in running.items():
            task = current.tasks.get(record.spec.id)
            if task is None or task.revision != record.revision or task.status != TaskStatus.RUNNING:
                affected[future] = record
                continue
            self.pending.add(future)
            def commit(done, record=record):
                self.pending.discard(done)
                if not done.cancelled() and done.exception() is None:
                    self._commit(state, record, done.result())
            future.add_done_callback(commit)
        self._cancel_pending(state, affected)

    def _expire_carried(self, state):
        if not any(task.status == TaskStatus.RUNNING and task.run_id != state.active_run
                   and (not task.lease_expires_at or task.lease_expires_at <= utc_now().isoformat())
                   for task in state.tasks.values()):
            return state
        def operation(current):
            for task in current.tasks.values():
                if task.status == TaskStatus.RUNNING and task.run_id != current.active_run and (
                        not task.lease_expires_at or task.lease_expires_at <= utc_now().isoformat()):
                    task.failure = Failure(ErrorCode.TOOL_TIMEOUT,
                        'prior_dispatch_expired', task.spec.id).payload()
                    task.transition(TaskStatus.FAILED, at=utc_now().isoformat())
        return self._mutate(state, operation)

    async def run(self, state, budget, *, deadline=None):
        deadline = deadline or time.monotonic()+budget.limits.request_seconds
        running = {}
        try:
            while True:
                current = self.store.get(state.id, state.actor, state.conversation)
                if current.version != state.version or current.active_run != state.active_run:
                    self._detach_pending(state, current, running)
                    return current
                current = self._expire_carried(current)
                current, refresh = self._advance(current)
                for source in refresh:
                    try:
                        ticket = budget.reserve('retry')
                        budget.finish(ticket, 0)
                    except BudgetExhausted:
                        def stop_ref(current, source=source):
                            for task in current.tasks.values():
                                if source in task.spec.depends_on and task.status in {
                                        TaskStatus.WAITING_DEPENDENCIES, TaskStatus.READY}:
                                    task.failure = Failure(ErrorCode.BUDGET_EXHAUSTED,
                                        'input_revalidation', task.spec.id).payload()
                                    task.transition(TaskStatus.BLOCKED, at=utc_now().isoformat())
                        current = self._mutate(current, stop_ref)
                    else:
                        current = self._refresh(current, source)
                        stale = {future: record for future, record in running.items()
                                 if current.tasks[record.spec.id].revision != record.revision}
                        self._cancel_pending(state, stale)
                        for future in stale:
                            running.pop(future)
                if refresh:
                    continue
                ready = [task for task in current.tasks.values() if task.status == TaskStatus.READY]
                for task in ready[:max(0, self.max_concurrency-len(running))]:
                    current = self._start(current, task.spec.id)
                    record = current.tasks[task.spec.id]
                    if record.status != TaskStatus.RUNNING:
                        continue
                    future = asyncio.create_task(self.worker.run(record, current, budget))
                    running[future] = record
                if not running:
                    if any(task.status == TaskStatus.RUNNING for task in current.tasks.values()):
                        if time.monotonic() >= deadline:
                            return current
                        await asyncio.sleep(.01)
                        continue
                    if any(task.status == TaskStatus.WAITING_DEPENDENCIES and not task.spec.depends_on
                           for task in current.tasks.values()):
                        continue
                    return self.store.get(state.id, state.actor, state.conversation)
                remaining = max(0, deadline-time.monotonic())
                done, _ = await asyncio.wait(running, timeout=remaining,
                                             return_when=asyncio.FIRST_COMPLETED)
                if not done:
                    for task in running.values():
                        self._commit(state, task, TaskOutcome(TaskStatus.FAILED,
                            failure=Failure(ErrorCode.TOOL_TIMEOUT, 'request_deadline',
                                            task.spec.id).payload()))
                    self._cancel_pending(state, running)
                    return self.store.get(state.id, state.actor, state.conversation)
                for future in done:
                    task = running.pop(future)
                    try:
                        outcome = future.result()
                    except BudgetExhausted:
                        outcome = TaskOutcome(TaskStatus.FAILED, failure=Failure(
                            ErrorCode.BUDGET_EXHAUSTED, 'task_dispatch', task.spec.id).payload())
                    except Exception:
                        outcome = TaskOutcome(TaskStatus.FAILED, failure=Failure(
                            ErrorCode.INTERNAL_ERROR, 'task_execution', task.spec.id).payload())
                    if not self._commit(state, task, outcome):
                        continue
                    if (outcome.status == TaskStatus.FAILED and outcome.failure
                            and outcome.failure.get('retryable') and task.attempts < task.spec.max_attempts):
                        try:
                            ticket = budget.reserve('retry')
                            budget.finish(ticket, 0)
                        except BudgetExhausted:
                            continue
                        def retry(current, task_id=task.spec.id):
                            current.tasks[task_id].transition(TaskStatus.READY,
                                at=utc_now().isoformat(), reason='bounded_transient_retry')
                        self._mutate(state, retry)
        except asyncio.CancelledError:
            for task in running.values():
                self._commit(state, task, TaskOutcome(TaskStatus.FAILED,
                    failure=Failure(ErrorCode.TOOL_TIMEOUT, 'request_cancelled', task.spec.id).payload()))
            self._cancel_pending(state, running)
            raise
        except (VersionConflict, WorkflowExpired):
            self._cancel_pending(state, running)
            return self.store.get(state.id, state.actor, state.conversation)
