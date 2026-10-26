"""Orchestrator entry: local understanding/composition, then one dynamic decision."""
from __future__ import annotations

import re
import asyncio
import time
import uuid

from careguide.domain import IntentCategory
from careguide.errors import ErrorCode, Failure
from careguide.intent_recognizer import IntentRecognizer, IntentResult
from careguide.qic_intent_recognizer import QICIntentRecognizer
from careguide.orchestration.budget import BudgetExhausted, use_budget
from careguide.orchestration.contracts import PlanChange, TaskStatus
from careguide.orchestration.decisions import RequestDecision, TaskAnswer
from careguide.orchestration.planner import PlanningFailure
from careguide.orchestration.runtime import OrchestrationRuntime, preference_snapshot
from careguide.orchestration.store import VersionConflict, WorkflowExpired, utc_now
from careguide.orchestration.understanding import LocalUnderstanding, compose_plan
from careguide.orchestration.understanding_planner import UnderstandingPlanner, build_context
from careguide.streaming import emit


class UnifiedRuntime(OrchestrationRuntime):
    def __init__(self, engine, path, *, understanding=None, planner=None, **kwargs):
        super().__init__(engine, path,
                         planner=planner or UnderstandingPlanner(engine.client, engine.model), **kwargs)
        self.understanding = understanding or LocalUnderstanding(engine.registry.corpus)
        from careguide.orchestration.artifacts import ResultArtifacts
        self.result_artifacts = ResultArtifacts(self.store.path)
        self.worker.brief_mode = True
        self.worker.artifacts = self.result_artifacts

    def _save_context(self, state, context, metadata):
        def save(value):
            known = {source['id'] for source in value.source_messages}
            value.source_messages.extend(s for s in context['user_sources'] if s['id'] not in known)
            value.events.append({'event': 'request_understood', **metadata,
                                 'message_id': context['current_message']['id']})
        return self.store.mutate(state.id, state.actor, state.conversation, save,
                                 plan_version=state.version, run_id=state.active_run)

    async def _respond(self, state, budget, message, started, **kwargs):
        from careguide.orchestration.answering import generate_answer
        deadline = started+budget.limits.request_seconds
        with use_budget(budget, deadline=deadline):
            emit('stage', stage='answering', message='正在根据查询结果生成回答')
            answer, diagnostic = await generate_answer(self.engine.client, self.engine.model, state, budget, message)
        latest = self.store.get(state.id, state.actor, state.conversation)
        if (latest.version, latest.revision) != (state.version, state.revision):
            from careguide.streaming import emit_answer
            if answer is not None:
                await emit_answer('answer_reset', answer_id=diagnostic['answer_id'], reason='workflow_changed')
            answer, state = None, latest
            diagnostic = {**diagnostic, 'status': 'stale_result_fallback'}
            kwargs['failures'] = ()
        body = diagnostic.pop('body', None) if answer is not None else None
        result = self._reply(state, budget, message, started, answer=answer, synthesis=diagnostic, **kwargs)
        if body is not None:
            result['answer'] = {'body': body, 'sources': diagnostic['sources']}
        result['answer_generation'] = diagnostic
        return result

    async def process(self, message, user_id, conv_id, history=None, *, workflow_event=None):
        started, at, message_id = time.monotonic(), utc_now().isoformat(), uuid.uuid4().hex
        history = history or []
        state = (self.store.get(workflow_event['workflow_id'], user_id, conv_id)
                 if workflow_event else self.store.latest(user_id, conv_id))
        urgent = IntentRecognizer._pattern(message).get('urgent')
        boundary = QICIntentRecognizer._boundary(message)
        if urgent or boundary == 'urgent_handoff':
            if state:
                self._preempt(state)
            return await self.engine.process(message, user_id, conv_id, history, understood=IntentResult(
                IntentCategory.EMERGENCY, .95, [IntentCategory.EMERGENCY], 'urgent_handoff',
                urgency='CRITICAL', boundary='urgent_handoff'))
        if boundary:
            return await self.engine.process(message, user_id, conv_id, history,
                understood=IntentResult(None, 0, [], boundary, boundary=boundary))
        if re.fullmatch(r'(?:请)?转人工(?:协助|咨询|客服)?|(?:查询)?CG-[A-F0-9]{12}', message.strip(), re.I):
            # Preserve the existing local consultation-record workflow. This
            # records a request; it does not contact a human or external system.
            return await self.engine.process(message, user_id, conv_id, history,
                understood=IntentResult(None, 0, [], 'needs_clarification'))
        if state and workflow_event and state.version != workflow_event['plan_version']:
            raise VersionConflict('Old plan version')
        if state:
            state = self.store.expire_waits(state.id, user_id, conv_id)
            if state.status == 'expired':
                if workflow_event:
                    raise WorkflowExpired('Workflow expired')
                state = None
        decision, events, metadata, understood = None, [], {}, None
        if workflow_event and workflow_event['kind'] != 'change':
            kind = workflow_event['kind']
            budget = self.ledger(state)
            if kind == 'answer':
                state = self.store.answer(state.id, user_id, conv_id, workflow_event['task_id'],
                    workflow_event['values'], plan_version=workflow_event['plan_version'],
                    task_revision=workflow_event['task_revision'], event_id=workflow_event.get('event_id'))
            elif kind == 'cancel':
                task = state.tasks.get(workflow_event['task_id'])
                if task is None or task.revision != workflow_event['task_revision']:
                    raise VersionConflict('Old task revision')
                state = self.store.apply_change(state.id, user_id, conv_id, PlanChange(
                    base_version=state.version, reason='用户明确取消', cancel=[task.spec.id]),
                    user_text='取消任务 '+task.spec.id)
                state = self.store.end_run(state.id, user_id, conv_id, None, status='cancelled' if all(
                    t.status == TaskStatus.CANCELLED for t in state.tasks.values()) else 'ready')
                return self._reply(state, budget, message, started)
            context = build_context(message, message_id, at, history, state)
            signals = {'request_context': context, 'decision_source': 'structured'}
            return await self._execute_workflow(state, budget, message, user_id, conv_id, history,
                                                 started, at, signals, 'execute')
        # Literal answers to a single persisted question are unambiguous. No
        # classifier or planning call is needed for these narrow forms.
        if state and not workflow_event:
            waiting = [t for t in state.tasks.values() if t.status == TaskStatus.WAITING_INPUT]
            if len(waiting) == 1 and len(waiting[0].missing_fields) == 1:
                task, value = waiting[0], message.strip()
                key = task.missing_fields[0]
                choices = [str(item.get(task.spec.inputs[key].item_field))
                           for option in task.options for item in option.get('values', [])
                           if key in task.spec.inputs]
                if value in choices or key == 'specification' and re.fullmatch(
                        r'[\d.]+\s*(?:mg|g|ml|μg|ug|毫克|克|毫升)(?::[\d.]+\s*(?:mg|g))?', value, re.I):
                    return await self.process(message, user_id, conv_id, history, workflow_event={
                        'kind': 'answer', 'workflow_id': state.id, 'plan_version': state.version,
                        'task_id': task.spec.id, 'task_revision': task.revision,
                        'values': {key: value}, 'event_id': message_id})
        understood = await self.understanding.understand(message, message_id=message_id, at=at,
                                                        history=history, classify=False)
        proposal, metadata = compose_plan(message, understood)
        if workflow_event:
            proposal = None  # explicit changes must target their existing plan
        if proposal and (state is None or message != state.goal):
            state = None  # independent new query; preserve old workflow/actions
        fresh = state is None
        if fresh:
            state = self.store.create_pending(user_id, conv_id, message, source_at=at,
                                              source_message_id=message_id)
            budget = self.ledger(state, create=True)
        else:
            budget = self.ledger(state)
        with use_budget(budget, deadline=started+budget.limits.request_seconds):
            try:
                async with asyncio.timeout(max(.001, started+budget.limits.request_seconds-time.monotonic())):
                    await self.understanding.enrich(message, understood, budget=budget, scope=(user_id, conv_id),
                                                   allow_remote=self.engine.client is not None)
            except TimeoutError:
                understood.diagnostics.append('understanding_request_deadline')
        metadata['local_understanding'] = understood.model_dump(mode='json')
        context = build_context(message, message_id, at, history, state, understood)
        if fresh and proposal:
            decision = RequestDecision(kind='new', plan=proposal)
        elif not workflow_event and message == state.goal and state.tasks:
            decision = RequestDecision(kind='run', workflow_id=state.id, base_version=state.version)
        else:
            if self.engine.client is None:
                decision = RequestDecision(kind='clarify', question=(
                    '请明确本次查询对象、条件及对应任务；当前未配置模型，无法可靠规划这个请求。'))
            else:
                try:
                    emit('stage', stage='planning', message='正在生成任务计划')
                    with use_budget(budget, deadline=started+budget.limits.request_seconds):
                        decision, events = await self.planner.decide(context, budget=budget)
                except PlanningFailure as exc:
                    return self._reply(state, budget, message, started, failures=[exc.failure],
                                       signals={'plan_source': 'llm', 'planning_events': exc.events})
                metadata = {**metadata, 'plan_source': 'llm', 'recognition_source': understood.recognition_source}
        if decision.kind == 'boundary':
            if decision.boundary == 'urgent_handoff':
                self._preempt(state)
            return await self.engine.process(message, user_id, conv_id, history, understood=IntentResult(
                IntentCategory.EMERGENCY if decision.boundary == 'urgent_handoff' else None, 0, [],
                decision.boundary, boundary=decision.boundary))
        if decision.kind == 'clarify':
            state = self._save_context(state, context, {**metadata, 'decision': 'clarify'})
            if not state.tasks:
                state.status = 'waiting_user'
                state = self.store.save(state, expected_revision=state.revision)
            return self._reply(state, budget, message, started,
                               note='\n\n'+decision.question, signals=metadata)
        if decision.kind == 'new':
            if state.tasks:
                # A model decision about an active workflow is charged to its
                # existing ledger, even if it discovers an independent request.
                prior_budget = state.budget_id or state.id
                state = self.store.create_pending(user_id, conv_id, message, source_at=at,
                                                  source_message_id=message_id)
                state.budget_id = prior_budget
                state = self.store.save(state, expected_revision=state.revision)
            state = self._save_context(state, context, metadata)
            state.preference_snapshot = preference_snapshot(history)
            state = self.store.save(state, expected_revision=state.revision)
            state = self.store.initialize(state, decision.plan, events)
        elif decision.kind in {'answer', 'change', 'cancel'}:
            if decision.kind == 'change':
                try:
                    async with budget.call('replan'):
                        pass
                except BudgetExhausted:
                    return self._reply(state, budget, message, started, failures=[Failure(
                        ErrorCode.BUDGET_EXHAUSTED, 'user_replanning').payload()])
            if decision.change and decision.change.add and budget.summary()['closing']:
                return self._reply(state, budget, message, started, failures=[Failure(
                    ErrorCode.BUDGET_EXHAUSTED, 'closing_new_tasks').payload()])
            state = self.store.apply_decision(state.id, user_id, conv_id, decision, user_text=message,
                sources=context['user_sources'], source_at=at,
                authorized_entities=self.worker.entities._explicit_entities(message, []))
            if decision.kind == 'cancel':
                return self._reply(state, budget, message, started, signals=metadata)
        context = build_context(message, message_id, at, history, state, understood)
        signals = {**metadata, 'request_context': context}
        return await self._execute_workflow(state, budget, message, user_id, conv_id, history,
                                             started, at, signals, 'execute')
