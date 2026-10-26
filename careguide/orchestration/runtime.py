"""LangGraph outer control flow over the existing scoped agent runtime.

WorkflowStore is authoritative for task commits; graph checkpoints contain only
control flow and references. Replaying a graph node never replays completed work.
"""
from __future__ import annotations

import asyncio
import hashlib
import re
import time
import uuid
from datetime import datetime, timedelta
from typing import TypedDict
from types import SimpleNamespace

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from careguide.errors import ErrorCode, Failure
from careguide.intent_recognizer import IntentRecognizer, IntentResult
from careguide.domain import IntentCategory
from careguide.orchestration.budget import BudgetExhausted, BudgetLedger, BudgetLimits, use_budget
from careguide.orchestration.contracts import (ACTION_ARGUMENTS, HOSPITAL_ACTIONS, PREPARE_ACTIONS,
    TRANSITIONS, PlanChange, PlanProposal, TaskRecord, TaskSpec, TaskStatus)
from careguide.orchestration.planner import ModelPlanner, PlanningFailure
from careguide.orchestration.replies import compose_workflow, workflow_status
from careguide.orchestration.scheduler import DAGScheduler
from careguide.orchestration.store import VersionConflict, WorkflowExpired, WorkflowStore, utc_now
from careguide.orchestration.worker import TaskWorker
from careguide.trace_store import TRACE_VERSION, validate_trace_completeness
from careguide.task_context import TaskContext


class Flow(TypedDict, total=False):
    operation: str
    failures: list[dict]
    replan_rounds: int
    next_step: str


def needs_plan(text, records=()):
    """A conservative raw-text gate independent of a classifier's single label."""
    names = {record.drug_name for record in records if record.drug_name and record.drug_name in text}
    categories = int(bool(re.search(r'医生|医师|号源|预约|挂号', text))) + int(bool(names) or bool(
        re.search(r'说明书|禁忌|药品|相互作用', text))) + int(bool(re.search(r'就诊准备|科室引导|头痛就诊', text)))
    two_drugs = re.search(r'(?:片|胶囊|喷雾剂|滴鼻液|布洛芬|酮替芬)[^，。；]{0,35}'
                         r'(?:和|与|、)[^，。；]{0,35}(?:片|胶囊|喷雾剂|滴鼻液|布洛芬|酮替芬)', text)
    return bool(two_drugs) or len(names) > 1 or categories > 1 or bool(re.search(
        r'同时|另外|以及|顺便|顺带|顺手|此外|还想|还要|也想|也查|再查|再看|还有|一并|然后|再帮|并且|分别|两个|两种|[；;\n]|'
        r'(?:医生|科室).*(?:号源|预约)|(?:说明书|药品).*和.*(?:说明书|药品)', text))


def changes_goal(text):
    return bool(re.search(r'改成|改为|改查|换成|换号源|更换|取消|不用|另外|增加|再加|不查', text))


def standalone_request(text):
    """Recognize explicit new requests without guessing what a short reply means."""
    if re.search(r'这个|那个|上述|刚才|这家|那家|该药|该医生', text):
        return False
    if re.fullmatch(r'(?:请问)?医院(?:地址|介绍)[？?。]?', text):
        return True
    return bool(re.search(r'查询|查一下|查查|帮我查|请查|看看|^查', text)
                and re.search(r'医生|医师|号源|预约|挂号|药|说明书|禁忌|科室|就诊|医院', text))


def verified_simple_request(text):
    """Only recognized single-query forms bypass planning; uncertainty plans."""
    cleaned = re.sub(r'(?:禁忌|注意事项|用法|用量|剂量)(?:和|及|、)(?:禁忌|注意事项|用法|用量|剂量)', '', text)
    if len(text) > 160 or re.search(r'和|与|及|、|同时|另外|顺|还有|再|也|[；;\n]', cleaned):
        return bool(re.fullmatch(r'[^，。；;\n]{1,100}(?:相互作用|合用|一起吃)[？?。]?', text))
    patterns = [r'(?:医生|医师|号源|可预约时间|说明书|禁忌|注意事项|用法|用量|相互作用)[？?。]?',
        r'号源[，,]?(?:不要|不用|避开)(?:上午|下午|晚上)[？?。]?',
        r'(?:材料|流程|渠道|就诊准备|就医信息|预约状态)[？?。]?',
        r'(?:挂什么科|看什么科|看哪个科|医院地址|医院介绍|就诊前需要什么准备)[？?。]?',
        r'(?:SLOT-|APT-)[A-Z0-9-]+[？?。]?',
        r'^头痛就诊前(?:要|需要)?记录(?:哪些|什么)信息[？?。]?',
        r'^(?:反复)?头痛[，,]?就诊前怎么记录[？?。]?']
    queries = re.findall(r'查询|查一下|查查|问一下|看看|查', text)
    return len(queries) <= 1 and any(re.search(pattern+'$', text, re.I) for pattern in patterns)


def preference_snapshot(history):
    """Read only server-attached, source-validated durable preference metadata."""
    from careguide.profile_memory import Candidate, Extraction, eligible_sources, validate_candidates
    for turn in reversed(history):
        preferences = turn.get('durable_preferences')
        if not isinstance(preferences, dict):
            continue
        field = preferences.get('preferred_time')
        if not isinstance(field, dict):
            continue
        try:
            expires = datetime.fromisoformat(field['expires_at'])
            if expires <= utc_now():
                continue
            source = SimpleNamespace(role='user', content=field['source_quote'],
                                     metadata={'message_id': field['source_message_id']})
            candidate = Candidate(key='preferred_time', value=field['value'],
                source_message_id=field['source_message_id'], source_quote=field['source_quote'])
            validate_candidates(Extraction(candidates=[candidate]), eligible_sources([source]))
            return {'preferred_time': dict(field)}
        except (KeyError, TypeError, ValueError, PermissionError):
            continue
    return {}


class OrchestrationRuntime:
    def __init__(self, engine, path, *, limits=None, planner=None, wait_seconds=300):
        self.engine = engine
        self.store = WorkflowStore(path)
        # AsyncSqliteSaver commits across await points. Sharing its writer
        # file with synchronous business transactions can deadlock the event
        # loop until busy_timeout. Keep graph checkpoints on the same volume,
        # in a separate file; task/action atomicity stays in the business DB.
        self.checkpoint_path = self.store.path.with_name(
            self.store.path.stem+'.checkpoints'+self.store.path.suffix)
        self.limits = limits or BudgetLimits()
        self.planner = planner or ModelPlanner(engine.client, engine.model)
        self.worker = TaskWorker(engine)
        self.scheduler = DAGScheduler(self.store, self.worker, wait_seconds=wait_seconds)

    def __getattr__(self, name):
        return getattr(self.engine, name)

    def ledger(self, state, *, create=False):
        return BudgetLedger(self.store.path, state.budget_id or state.id, state.actor, state.conversation, self.limits,
                            require_existing=not create)

    def _reply(self, state, budget, message, started, *, signals=None, failures=(), run_ids=(), note='', synthesis=None, answer=None):
        result = compose_workflow(state, budget, extra_failures=failures, intent_signals=signals)
        result['response'] = (answer if answer is not None else result['response']) + note
        if synthesis is not None:
            result['synthesis'] = synthesis
        result['latency_ms'] = (time.monotonic()-started)*1000
        from careguide.timing import timing_snapshot
        result['timings'] = timing_snapshot()
        result['integrations'] = {
            'knowledge': {'backend': getattr(self.knowledge_backend, 'name', 'embedded'),
                          'status': 'used' if result['evidence'] else 'not_used'},
            'hospital': {'backend': getattr(self.appointment_service.backend, 'name', 'disabled'),
                         'status': 'used' if any(task.spec.action in HOSPITAL_ACTIONS
                                                for task in state.tasks.values()) else 'not_used',
                         'simulation': True}}
        if any(event.get('event') == 'direct_route_adopted' for event in state.events):
            result['execution_mode'] = 'direct_task_workflow'
        finished = utc_now().isoformat()
        calls = [call for event in state.events if event.get('run_id') in run_ids
                 for call in event.get('calls', [])]
        trace = {'trace_version': TRACE_VERSION, 'request_id': result['request_id'],
            'conversation_id': state.conversation, 'user_id': state.actor, 'timestamp': finished,
            'input': message, 'started_at': (utc_now()-timedelta(milliseconds=result['latency_ms'])).isoformat(),
            'finished_at': finished,
            'total_latency_ms': result['latency_ms'], 'status': result['workflow_status'],
            'task_understanding': {'task_type': 'validated_task_plan', 'entities': {},
                'missing_fields': {key: task.missing_fields for key, task in state.tasks.items()},
                'workflow_status_after_understanding': result['workflow_status'], 'intent_signals': signals or {}},
            'routing': {'primary_agent': result['primary_agent'], 'supporting_agents': result['supporting_agents'],
                'routing_reason': 'Full-text model plan validated by code', 'routing_confidence': None},
            'execution': {'execution_waves': [], 'tool_calls': calls,
                'workflow': state.model_dump(mode='json'), 'budget': budget.summary(),
                'synthesis': synthesis, 'timings': result['timings']},
            'evidence': result['evidence'], 'side_effects': [
                {'type': 'synthetic_action_prepared', 'id': call['output']['id']}
                for call in calls if call['output'].get('status') == 'pending_confirmation'],
            'final': {'workflow_status': result['workflow_status'], 'final_response': result['response'],
                'agent_types': result['agent_types'], 'primary_agent': result['primary_agent'],
                'supporting_agents': result['supporting_agents'], 'total_tool_calls': len(calls),
                'total_latency_ms': result['latency_ms']}}
        if not validate_trace_completeness(trace)['complete']:
            raise RuntimeError('Workflow trace contract incomplete')
        try:
            self.traces.append(trace)
        except Exception:
            result['trace_status'] = 'unavailable'
        return result

    def _preempt(self, state):
        with self.store.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            current = self.store._load(db, state.id, state.actor, state.conversation)
            for task in current.tasks.values():
                if TaskStatus.CANCELLED in TRANSITIONS.get(task.status, set()):
                    task.transition(TaskStatus.CANCELLED, at=utc_now().isoformat(), reason='urgent_preemption')
            current.active_run, current.lease_expires_at = None, None
            current.status = 'urgent_handoff'
            current.events.append({'event': 'urgent_preemption', 'at': utc_now().isoformat()})
            self.store._write(db, current)
            self.store._expire_invalid_actions(db, current)

    async def process(self, message, user_id, conv_id, history=None, *, workflow_event=None):
        started = time.monotonic()
        received_at = utc_now().isoformat()
        history = history or []
        state = (self.store.get(workflow_event['workflow_id'], user_id, conv_id)
                 if workflow_event else self.store.latest(user_id, conv_id))
        # Existing emergency rules run before planning or accepting a reply.
        # The full recognizer remains responsible for contextual boundary signals.
        urgent = IntentRecognizer._pattern(message).get('urgent')
        checker = getattr(self.engine.recognizer, '_boundary', None) or getattr(self.engine.recognizer, '_pattern_v3', None)
        if checker:
            boundary = checker(message)
            urgent = urgent or boundary == 'urgent_handoff' or isinstance(boundary, dict) and boundary.get('disposition') == 'urgent_handoff'
        if urgent:
            if state:
                self._preempt(state)
            understood = IntentResult(IntentCategory.EMERGENCY, .95, [IntentCategory.EMERGENCY],
                'urgent_handoff', urgency='CRITICAL', boundary='urgent_handoff')
            return await self.engine.process(message, user_id, conv_id, history, understood=understood)
        if workflow_event and state.version != workflow_event['plan_version']:
            raise VersionConflict('Reply belongs to an old plan version')
        if state and not workflow_event and state.status not in {
                'waiting_user', 'pending_confirmation', 'reconciling', 'executing', 'ready', 'planning'}:
            if (not changes_goal(message) and message != state.goal) or state.status in {'expired', 'urgent_handoff'}:
                state = None
        if (state and not workflow_event and message != state.goal
                and not changes_goal(message) and standalone_request(message)):
            # A pending question does not own every later message in the chat.
            # Keep its durable state and confirmation intact; explicit events
            # can still resume it after this independent request is handled.
            state = None
        fresh = state is None
        preferences = preference_snapshot(history) if fresh else {}
        understood = None
        if state is None and not needs_plan(message, self.registry.corpus.records):
            # Preserve direct routing and its single recognition pass. Missing
            # inputs promote a deterministic one-task route into durable state,
            # without paying for a Planner merely to ask for a known field.
            direct_id = uuid.uuid4().hex
            direct_budget = BudgetLedger(self.store.path, direct_id, user_id, conv_id, self.limits)
            with use_budget(direct_budget, deadline=started+direct_budget.limits.request_seconds):
                understood = await self.engine.recognizer.recognize(message, history)
            agents = self.engine.route_decision(understood)
            uncertain = understood.disposition == 'route' and not verified_simple_request(message)
            spec = None
            if not uncertain and understood.disposition == 'route' and len(agents) == 1 and agents[0].value != 'escalation':
                context = TaskContext.from_intent_result(understood)
                business = understood.business_action or understood.fine_intent
                action = next((key for key, value in HOSPITAL_ACTIONS.items() if value[1] == business), None)
                call = ({'name': action, 'input': context.entities} if action else
                        self.engine._fallback_call(agents[0], message, context))
                action = call['name']
                fields = ACTION_ARGUMENTS[action].model_fields
                arguments = {key: value for key, value in call['input'].items() if key in fields}
                spec = TaskSpec(id='direct', fragment=message, action=action, agent=agents[0],
                    arguments=arguments, completion='prepared' if action in PREPARE_ACTIONS else
                    'items' if action in HOSPITAL_ACTIONS else 'source_evidence')
                proposal = PlanProposal(goal=message, tasks=[spec])
                probe = TaskRecord(spec=spec)
                missing = self.worker.prepare(probe, SimpleNamespace(tasks={}, preference_snapshot=preferences,
                    source_messages=[{'content': message, 'at': received_at}], created_at=received_at)).missing_fields
                temporal_scope = action == 'list_available_slots' and any(
                    key in probe.arguments for key in ('time_period', 'date'))
                if not missing and action not in PREPARE_ACTIONS and not temporal_scope:
                    spec = None
                else:
                    spec = spec.model_copy(update={'arguments': dict(probe.arguments)}, deep=True)
                    proposal = PlanProposal(goal=message, tasks=[spec])
            if spec is None and not uncertain:
                with use_budget(direct_budget, deadline=started+direct_budget.limits.request_seconds):
                    result = await self.engine.process(message, user_id, conv_id, history, understood=understood)
                result['budget'] = direct_budget.summary()
                return result
            if uncertain:
                state = self.store.create_pending(user_id, conv_id, message, identity=direct_id, source_at=received_at)
            else:
                state = self.store.create(user_id, conv_id, proposal, source_text=message,
                                          identity=direct_id, source_at=received_at)
                state.events.append({'event': 'direct_route_adopted', 'action': spec.action,
                                     'model_planner_used': False})
                state = self.store.save(state, expected_revision=state.revision)
        if fresh:
            if state is None:
                state = self.store.create_pending(user_id, conv_id, message, source_at=received_at)
        else:
            state = self.store.expire_waits(state.id, user_id, conv_id)
            if state.status == 'expired':
                if workflow_event:
                    raise WorkflowExpired('Workflow expired; start a new request')
                return self._reply(state, self.ledger(state), message, started)
        if fresh and preferences:
            state.preference_snapshot = preferences
            state = self.store.save(state, expected_revision=state.revision)
        budget = self.ledger(state, create=fresh)
        if not workflow_event and re.fullmatch(r'(?:不要|不用|取消)(?:当前|这个|本次|模拟)?预约[。！!\s]*', message):
            pending = [task for task in state.tasks.values() if task.spec.action in PREPARE_ACTIONS
                       and task.status in {TaskStatus.WAITING_INPUT, TaskStatus.WAITING_CONFIRMATION}]
            if len(pending) == 1:
                workflow_event = dict(kind='cancel', task_id=pending[0].spec.id,
                                      task_revision=pending[0].revision)
        if workflow_event and workflow_event['kind'] == 'cancel':
            task = state.tasks.get(workflow_event['task_id'])
            if task is None or task.revision != workflow_event['task_revision']:
                raise VersionConflict('Cancellation belongs to an old task revision')
            state = self.store.apply_change(state.id, user_id, conv_id, PlanChange(
                base_version=state.version, reason='用户通过对应任务明确取消', cancel=[task.spec.id]),
                user_text='取消任务 '+task.spec.id)
            # Cancelling an unexecuted branch needs neither a model call nor
            # additional budget. In particular it must work after budget expiry.
            state = self.store.end_run(state.id, user_id, conv_id, None, status=workflow_status(state))
            return self._reply(state, budget, message, started)
        operation = 'plan' if not state.tasks else 'execute'
        if workflow_event:
            kind = workflow_event['kind']
            if kind == 'answer':
                state = self.store.answer(state.id, user_id, conv_id,
                    workflow_event['task_id'], workflow_event['values'],
                    plan_version=workflow_event['plan_version'],
                    task_revision=workflow_event['task_revision'], event_id=workflow_event.get('event_id'))
            elif kind == 'change':
                operation = 'change'
        elif not fresh:
            waiting = [task for task in state.tasks.values() if task.status == TaskStatus.WAITING_INPUT]
            if changes_goal(message):
                operation = 'change'
            elif len(waiting) == 1 and len(waiting[0].missing_fields) == 1:
                task, value = waiting[0], message.strip()
                field = task.missing_fields[0]
                # Only bind an unambiguous literal to the one persisted question.
                option_values = [str(item.get(task.spec.inputs[field].item_field))
                    for option in task.options for item in option.get('values', [])
                    if field in task.spec.inputs]
                if ((field == 'specification' and re.fullmatch(r'[\d.]+\s*(?:mg|g|ml|μg|ug|毫克|克|毫升)(?::[\d.]+\s*(?:mg|g))?', value, re.I))
                        or value in option_values):
                    state = self.store.answer(state.id, user_id, conv_id, task.spec.id, {field: value},
                        plan_version=state.version, task_revision=task.revision)
                else:
                    return self._reply(state, budget, message, started,
                        note='\n\n尚未识别这条消息与当前任务的关系。请在对应任务中补充信息；'
                             '若要发起新查询，请写明查询对象，例如“查询医院地址”。')
            elif waiting:
                return self._reply(state, budget, message, started,
                    note='\n\n请在对应任务中补充信息，明确本次回答属于哪项任务。')
            elif message != state.goal and message.strip() not in {
                    '继续', '继续任务', '恢复任务', '查看进度', '查询进度'}:
                return self._reply(state, budget, message, started,
                    note='\n\n尚未识别这条消息与当前任务的关系。请使用对应任务的确认或恢复入口；'
                         '若要发起新查询，请写明查询对象。')
        try:
            if understood is None:
                with use_budget(budget, deadline=started+budget.limits.request_seconds):
                    understood = await self.engine.recognizer.recognize(message, history)
            signals = understood.to_dict()
        except BudgetExhausted:
            return self._reply(state, budget, message, started, failures=[Failure(
                ErrorCode.BUDGET_EXHAUSTED, 'intent_signals').payload()])
        if understood.disposition == 'urgent_handoff':
            self._preempt(state)
            return await self.engine.process(message, user_id, conv_id, history, understood=understood)
        return await self._execute_workflow(state, budget, message, user_id, conv_id, history,
                                            started, received_at, signals, operation)

    async def _respond(self, *args, **kwargs):
        return self._reply(*args, **kwargs)

    async def _execute_workflow(self, state, budget, message, user_id, conv_id, history,
                                started, received_at, signals, operation):
        from careguide.streaming import emit, workflow_progress
        workflow_progress(state, stage='plan_ready')
        if operation != 'change':
            state = self.store.begin_run(state.id, user_id, conv_id, expected_version=state.version,
                                         lease_seconds=budget.limits.request_seconds+2)
        # A user change may invalidate an in-flight read. It does not take over
        # the old lease or release it on a rejected proposal; apply_change is
        # the atomic version boundary and acquires a new lease only on success.
        run_id = None if operation == 'change' else state.active_run
        request_runs = [run_id] if run_id else []
        identity = state.id
        deadline = started+budget.limits.request_seconds
        def current():
            return self.store.get(identity, user_id, conv_id)
        async def plan_node(flow):
            emit('stage', stage='planning', message='正在生成任务计划')
            existing = current()
            if existing.tasks:
                return {}
            try:
                decision = await self.planner.plan(existing.source_messages[0]['content'], signals,
                    existing=existing.model_dump(mode='json'), budget=budget)
                self.store.initialize(existing, decision.proposal, decision.events)
                return {'failures': []}
            except PlanningFailure as exc:
                return {'failures': [exc.failure]}
        async def execute(flow):
            if not flow.get('failures'):
                await self.scheduler.run(current(), budget, deadline=deadline)
            return {}
        async def replan(flow):
            nonlocal run_id
            emit('stage', stage='replanning', message='正在调整任务计划')
            existing = current()
            try:
                event = {'kind': 'user_change' if flow['operation'] == 'change' else 'execution_result',
                         'received_at': received_at,
                         'message': message, 'tasks': existing.model_dump(mode='json')['tasks']}
                decision = await self.planner.replan(message, signals,
                    existing=existing.model_dump(mode='json'), event=event, budget=budget)
                if decision.proposal.add and budget.summary()['closing']:
                    raise PlanningFailure(Failure(ErrorCode.BUDGET_EXHAUSTED,
                        'closing_new_tasks').payload(), decision.events)
                entities = self.worker.entities._explicit_entities(message, [])
                updated = self.store.apply_change(identity, user_id, conv_id,
                    decision.proposal, user_text=message, authorized_entities=entities, source_at=received_at)
                updated = self.store.begin_run(identity, user_id, conv_id, expected_version=updated.version,
                    lease_seconds=max(1, deadline-time.monotonic()+2))
                run_id = updated.active_run
                request_runs.append(run_id)
                return {'failures': [], 'replan_rounds': flow.get('replan_rounds', 0)+1,
                        'operation': 'execute'}
            except PlanningFailure as exc:
                return {'failures': [exc.failure]}
            except (ValueError, PermissionError) as exc:
                code = ErrorCode.PERMISSION_DENIED if isinstance(exc, PermissionError) else ErrorCode.INVALID_PARAMETERS
                return {'failures': [Failure(code, 'local_replan_validation').payload()]}
        def check(flow):
            existing = current()
            if flow.get('failures'):
                step = 'finish'
            elif any(task.status in {TaskStatus.WAITING_INPUT, TaskStatus.WAITING_CONFIRMATION,
                    TaskStatus.RECONCILING} for task in existing.tasks.values()):
                step = 'wait'
            elif (flow.get('replan_rounds', 0) < budget.limits.replan
                    and not budget.summary()['closing'] and time.monotonic() < deadline
                    and any(task.status == TaskStatus.FAILED and task.failure and task.failure['code'] in {
                        'not_covered', 'tool_unavailable', 'tool_timeout'} for task in existing.tasks.values())):
                step = 'replan'
            else:
                step = 'finish'
            return {'next_step': step}
        def wait(flow):
            existing = current()
            interrupt({'workflow_id': identity, 'version': existing.version,
                'waiting_tasks': [{'id': task.spec.id, 'revision': task.revision,
                    'status': task.status.value, 'missing_fields': task.missing_fields}
                    for task in existing.tasks.values() if task.status in {
                        TaskStatus.WAITING_INPUT, TaskStatus.WAITING_CONFIRMATION, TaskStatus.RECONCILING}]})
            return {'failures': [], 'operation': 'execute'}
        builder = StateGraph(Flow)
        builder.add_node('plan', plan_node)
        builder.add_node('execute', execute)
        builder.add_node('replan', replan)
        builder.add_node('check', check)
        builder.add_node('wait', wait)
        builder.add_node('finish', lambda flow: {})
        builder.add_conditional_edges(START, lambda flow: {'plan': 'plan', 'change': 'replan'}.get(
            flow['operation'], 'execute'), ['plan', 'replan', 'execute'])
        builder.add_edge('plan', 'execute')
        builder.add_edge('replan', 'execute')
        builder.add_edge('execute', 'check')
        builder.add_conditional_edges('check', lambda flow: flow['next_step'], ['wait', 'replan', 'finish'])
        builder.add_edge('wait', 'execute')
        builder.add_edge('finish', END)
        # A changed plan gets a distinct checkpoint stream. The task store still
        # atomically rejects stale versions, even if an older HTTP call is alive.
        suffix = uuid.uuid4().hex if operation == 'change' else ''
        thread = hashlib.sha256(f'{user_id}:{conv_id}:{identity}:{state.version}:{suffix}'.encode()).hexdigest()
        config = {'configurable': {'thread_id': thread}, 'recursion_limit': 40}
        flow = {'operation': operation, 'failures': [], 'replan_rounds': 0}
        try:
            async with AsyncSqliteSaver.from_conn_string(str(self.checkpoint_path)) as saver:
                graph = builder.compile(checkpointer=saver)
                snapshot = await graph.aget_state(config)
                invocation = Command(resume={'accepted': True}) if snapshot.interrupts else flow
                async with asyncio.timeout(max(.001, deadline-time.monotonic())):
                    with use_budget(budget, deadline=deadline):
                        flow = await graph.ainvoke(invocation, config)
        except TimeoutError:
            flow['failures'] = [Failure(ErrorCode.TOOL_TIMEOUT, 'workflow_deadline').payload()]
        except Exception as exc:
            flow['failures'] = [Failure(ErrorCode.INTERNAL_ERROR, 'workflow_control').payload()]
            latest = current()
            if latest.active_run == run_id:
                def record_error(value):
                    value.events.append({'event': 'workflow_control_failure',
                                         'diagnostic_type': type(exc).__name__})
                self.store.mutate(identity, user_id, conv_id, record_error,
                                  plan_version=latest.version, run_id=run_id)
        finally:
            latest = current()
            status = workflow_status(latest, extra_failures=flow.get('failures', []))
            state = self.store.end_run(identity, user_id, conv_id, run_id, status=status)
        return await self._respond(state, budget, message, started, failures=flow.get('failures', []),
                           signals=signals, run_ids=request_runs)

    async def confirm_action(self, identity, actor, conversation):
        action = self.appointment_service.actions.get(identity, actor)
        if action is None or action['conversation_id'] != conversation:
            raise LookupError('Action not found')
        binding = action.get('workflow_binding')
        if not binding:
            return await self.appointment_service.confirm(identity, actor, conversation)
        state = self.store.get(binding['workflow_id'], actor, conversation)
        budget = self.ledger(state)
        if action['status'] == 'completed':
            return action
        with use_budget(budget):
            result = await self.appointment_service.confirm(identity, actor, conversation)
        state = self.store.sync_actions(state.id, actor, conversation)
        return {**result, 'workflow': compose_workflow(state, budget)['workflow']}

    def reject_action(self, identity, actor, conversation):
        result = self.appointment_service.reject(identity, actor, conversation)
        binding = result.get('workflow_binding')
        if binding:
            state = self.store.sync_actions(binding['workflow_id'], actor, conversation)
            result = {**result, 'workflow': compose_workflow(state, self.ledger(state))['workflow']}
        return result
