import asyncio
import json
import os

from fastapi.testclient import TestClient
import pytest

from careguide import runtime as factory
from careguide.api import app
from careguide.corpus import DEFAULT_CORPUS
from careguide.orchestration.contracts import TaskStatus
from careguide.orchestration.decisions import RequestDecision, TaskAnswer
from careguide.orchestration.store import VersionConflict
from careguide.orchestration.unified import UnifiedRuntime
from careguide.orchestration.understanding_planner import UnderstandingPlanner
from scripts.reproduce_orchestration import build
from tests.test_chat_runtime import ScriptedClient, tool_reply
from tests.test_model_planner import reply


def unified(tmp_path):
    old, _ = build(tmp_path, [])
    return UnifiedRuntime(old.engine, old.store.path)


def test_local_composition_is_durable_and_brief_isolated(tmp_path):
    instance = unified(tmp_path)
    observed = []
    original = instance.engine._run_agent
    async def capture(agent, message, history, context):
        observed.append((message, history, context))
        return await original(agent, message, history, context)
    instance.engine._run_agent = capture
    async def run():
        result = await instance.process('查消化内科医生，同时查询精氨酸布洛芬片0.2g说明书', 'u', 'c')
        assert result['workflow_status'] == 'completed'
        assert result['budget']['used']['model'] == 0
        assert result['budget']['used']['tool'] == 2
        assert len(result['workflow']['tasks']) == 2
        assert result['synthesis']['status'] == 'not_needed'
        assert len(observed) == 1
        message, history, context = observed[0]
        brief = json.loads(message)
        assert '消化内科' not in message
        assert history == []
        assert brief['known_facts']['arguments']['drug_name'] == '精氨酸布洛芬片'
        assert context.task_type == 'medication_query'
        assert context.task_brief['context_policy'] == 'brief'
        assert all(task['worker_result'] for task in result['workflow']['tasks'])
        trace = instance.traces.get(result['request_id'])
        assert trace['task_understanding']['intent_signals']['plan_source'] == 'compose'
        sources = {s['id'] for s in instance.store.get(result['workflow']['id'], 'u', 'c').source_messages}
        goals = trace['task_understanding']['intent_signals']['local_understanding']['goals']
        assert all(source in sources for goal in goals for source in goal['sources'].values())
    asyncio.run(run())


def test_template_selection_confirmation_and_resume_do_not_need_planner(tmp_path):
    instance = unified(tmp_path)
    async def run():
        first = await instance.process('查消化内科医生的号源并预约', 'u', 'c')
        assert first['workflow_status'] == 'waiting_user'
        identity = first['workflow']['id']
        assert first['budget']['used']['model'] == 0
        def event(task, field, value):
            return dict(kind='answer', workflow_id=identity, plan_version=1,
                        task_id=task, task_revision=1, values={field: value})
        await instance.process('PRA-GASTRO-01', 'u', 'c', workflow_event=event('slots', 'practitioner_id', 'PRA-GASTRO-01'))
        prepared = await instance.process('SLOT-GASTRO-001', 'u', 'c', workflow_event=event('booking', 'slot_id', 'SLOT-GASTRO-001'))
        assert prepared['workflow_status'] == 'pending_confirmation'
        assert prepared['budget']['used']['model'] == 0
        action = prepared['action_requests'][0]
        confirmed = await instance.confirm_action(action['id'], 'u', 'c')
        assert confirmed['status'] == 'completed'
        assert (await instance.confirm_action(action['id'], 'u', 'c'))['status'] == 'completed'
        resumed = await instance.process('继续', 'u', 'c', workflow_event=dict(
            kind='run', workflow_id=identity, plan_version=1))
        assert resumed['workflow_status'] == 'completed'
        assert len(instance.appointment_service.backend.appointments) == 1
    asyncio.run(run())


def test_missing_spec_new_query_then_reopen_and_old_answer(tmp_path):
    instance = unified(tmp_path)
    async def run():
        first = await instance.process('查询精氨酸布洛芬片说明书', 'u', 'c')
        assert first['workflow_status'] == 'waiting_user'
        identity = first['workflow']['id']
        before = instance.store.get(identity, 'u', 'c').model_dump(mode='json')
        new = await instance.process('查心血管内科医生', 'u', 'c')
        assert new['workflow_status'] == 'completed'
        assert new['workflow']['id'] != identity
        assert instance.store.get(identity, 'u', 'c').model_dump(mode='json') == before
        reopened = UnifiedRuntime(instance.engine, instance.store.path)
        done = await reopened.process('0.2g', 'u', 'c', workflow_event=dict(kind='answer',
            workflow_id=identity, plan_version=1, task_id='goal1', task_revision=1,
            values={'specification': '0.2g'}, event_id='restore'))
        assert done['workflow_status'] == 'completed'
        assert done['budget']['used']['tool'] == 1
    asyncio.run(run())


def test_input_and_change_package_rolls_back_completely_on_invalid_addition(tmp_path):
    instance = unified(tmp_path)
    first = asyncio.run(instance.process('查询精氨酸布洛芬片说明书', 'u', 'c'))
    identity = first['workflow']['id']
    state = instance.store.get(identity, 'u', 'c')
    before = state.model_dump(mode='json')
    decision = RequestDecision.model_validate(dict(kind='change', workflow_id=identity, base_version=1,
        answers=[dict(task_id='goal1', task_revision=1, values={'specification': '0.2g'})],
        change=dict(base_version=1, reason='test', add=[dict(id='bad', fragment='查医生',
            action='unavailable_action', agent='appointment', completion='items')])))
    with pytest.raises(PermissionError):
        instance.store.apply_decision(identity, 'u', 'c', decision, user_text='0.2g，另外查医生')
    assert instance.store.get(identity, 'u', 'c').model_dump(mode='json') == before


def test_dynamic_mixed_answer_and_addition_with_real_worker_and_safe_synthesis_fallback(tmp_path):
    instance = unified(tmp_path)
    async def run():
        first = await instance.process('查询精氨酸布洛芬片说明书', 'u', 'c')
        identity = first['workflow']['id']
        decision = dict(kind='change', workflow_id=identity, base_version=1,
            answers=[dict(task_id='goal1', task_revision=1, values={'specification': '0.2g'})],
            change=dict(base_version=1, reason='用户补规格并新增查询', add=[dict(id='doctor',
                fragment='查心血管内科医生', action='search_doctors', agent='appointment',
                arguments={'department': '心血管内科'}, completion='items')]))
        args = {'drug_name': '精氨酸布洛芬片', 'dosage_form': '片剂', 'specification': '0.2g', 'section': 'all'}
        client = ScriptedClient([reply(decision), tool_reply('lookup_drug_label', args),
                                 {'content': 'done'}, reply({'sections': [dict(
                                     task_id='goal1', evidence_id='invented', excerpt='预约成功')]})])
        instance.engine.client = client
        instance.planner = UnderstandingPlanner(client)
        done = await instance.process('规格0.2g，另外查心血管内科医生', 'u', 'c')
        assert done['workflow_status'] == 'completed'
        assert done['workflow']['version'] == 2
        assert len(done['workflow']['tasks']) == 2
        assert done['budget']['used']['replan'] == 1
        assert done['synthesis']['status'] == 'deterministic_fallback'
        assert done['evidence']
        assert '预约成功' not in done['response']
        assert len(client.calls) == 4
        planning_input = json.loads(client.calls[0]['messages'][-1]['content'])
        assert planning_input['existing_workflow']['tasks']['goal1']['missing_fields'] == ['specification']
        assert planning_input['workers']['medication']['mission']
        trace = instance.traces.get(done['request_id'])
        assert trace['final']['final_response'] == done['response']
        assert trace['execution']['budget'] == done['budget']
    asyncio.run(run())


def test_default_factory_wires_existing_classifier_and_does_not_call_legacy_recognizer(monkeypatch, tmp_path):
    for key in tuple(os.environ):
        if key.startswith('CAREGUIDE_'):
            monkeypatch.delenv(key)
    monkeypatch.setenv('CAREGUIDE_BUSINESS_DB', str(tmp_path/'factory.sqlite3'))
    monkeypatch.setenv('CAREGUIDE_CORPUS_DIR', str(DEFAULT_CORPUS))
    monkeypatch.setenv('CAREGUIDE_HOSPITAL_BACKEND', 'memory')
    from careguide.orchestration.understanding import LocalClassifier
    calls = []
    def predict(self, text):
        calls.append((self.path, text))
        return {'label': '其他', 'scores': {'其他': 1}, 'margin': 1, 'truncated': False}
    monkeypatch.setattr(LocalClassifier, 'predict', predict)
    factory.get_chat_runtime.cache_clear()
    factory.get_registry.cache_clear()
    try:
        instance = factory.get_chat_runtime()[0]
        assert isinstance(instance, UnifiedRuntime)
        async def forbidden(*args, **kwargs):
            raise AssertionError('Legacy recognition must not run')
        instance.engine.recognizer.recognize = forbidden
        with TestClient(app) as http:
            result = http.post('/chat', json={'message': '有没有其他建议', 'user_id': 'u', 'conv_id': 'c'})
            assert result.status_code == 200
            assert result.json()['workflow_status'] == 'waiting_user'
            assert len(calls) == 1
            assert 'careguide-intent-roberta-v1' in str(calls[0][0])
            trace = instance.traces.get(result.json()['request_id'])
            local = trace['task_understanding']['intent_signals']['local_understanding']
            assert local['candidates'][-1]['source'] == 'classifier'
            foreign = http.get('/workflows/'+result.json()['workflow']['id'], params={'user_id': 'other', 'conv_id': 'c'})
            assert foreign.status_code == 404
    finally:
        factory.get_chat_runtime.cache_clear()
        factory.get_registry.cache_clear()


def test_model_discovered_new_request_keeps_charged_budget_and_old_wait(tmp_path):
    instance = unified(tmp_path)
    async def run():
        old = await instance.process('查询精氨酸布洛芬片说明书', 'u', 'c')
        old_state = instance.store.get(old['workflow']['id'], 'u', 'c')
        before = old_state.model_dump(mode='json')
        text = '帮我找心内科的医生'
        decision = dict(kind='new', plan=dict(goal=text, tasks=[dict(id='doctor', fragment=text,
            action='search_doctors', agent='appointment', arguments={'department': '心血管内科'}, completion='items')]))
        client = ScriptedClient([reply(decision)])
        instance.engine.client = client
        instance.planner = UnderstandingPlanner(client)
        result = await instance.process(text, 'u', 'c')
        assert result['workflow_status'] == 'completed'
        assert result['workflow']['id'] != old_state.id
        assert result['budget']['used']['model'] == 1
        assert instance.ledger(old_state).summary()['used']['model'] == 1
        assert instance.store.get(old_state.id, 'u', 'c').model_dump(mode='json') == before
        assert len(client.calls) == 1
    asyncio.run(run())


def test_structured_cancel_works_after_model_exhaustion_without_understanding(tmp_path):
    instance = unified(tmp_path)
    async def run():
        first = await instance.process('查询精氨酸布洛芬片说明书', 'u', 'c')
        state = instance.store.get(first['workflow']['id'], 'u', 'c')
        ledger = instance.ledger(state)
        for _ in range(ledger.limits.model):
            async with ledger.call('model'):
                pass
        async def forbidden(*args, **kwargs):
            raise AssertionError('Structured event must bypass local understanding')
        instance.understanding.understand = forbidden
        result = await instance.process('取消', 'u', 'c', workflow_event=dict(kind='cancel',
            workflow_id=state.id, plan_version=1, task_id='goal1', task_revision=1))
        assert result['workflow']['tasks'][0]['status'] == 'cancelled'
        assert result['budget']['used']['model'] == ledger.limits.model
        assert result['budget']['used']['tool'] == 0
    asyncio.run(run())


def test_unified_emergency_preempts_wait_before_understanding_or_tools(tmp_path):
    instance = unified(tmp_path)
    async def run():
        first = await instance.process('查询精氨酸布洛芬片说明书', 'u', 'c')
        async def forbidden(*args, **kwargs):
            raise AssertionError('Emergency must preempt ordinary understanding')
        instance.understanding.understand = forbidden
        result = await instance.process('突然胸痛呼吸困难', 'u', 'c')
        assert result['workflow_status'] == 'urgent_handoff'
        assert result['tools_used'] == []
        state = instance.store.get(first['workflow']['id'], 'u', 'c')
        assert state.tasks['goal1'].status == TaskStatus.CANCELLED
        assert instance.ledger(state).summary()['used']['tool'] == 0
    asyncio.run(run())


def test_mode_fallback_and_direct_confirmation_preserve_aliased_budget(tmp_path):
    from careguide.orchestration.runtime import OrchestrationRuntime
    instance = unified(tmp_path)
    async def run():
        anchor = await instance.process('查询精氨酸布洛芬片说明书', 'u', 'c')
        first = await instance.process('查消化内科医生的号源并预约', 'u', 'c')
        identity = first['workflow']['id']
        state = instance.store.get(identity, 'u', 'c')
        state.budget_id = anchor['workflow']['id']
        instance.store.save(state, expected_revision=state.revision)
        for task, field, value in [('slots', 'practitioner_id', 'PRA-GASTRO-01'),
                                   ('booking', 'slot_id', 'SLOT-GASTRO-001')]:
            prepared = await instance.process(value, 'u', 'c', workflow_event=dict(kind='answer',
                workflow_id=identity, plan_version=1, task_id=task, task_revision=1, values={field: value}))
        action = prepared['action_requests'][0]
        # Exercise AppointmentService's own fallback, outside UnifiedRuntime's
        # budget context, as in direct mode.
        confirmed = await instance.appointment_service.confirm(action['id'], 'u', 'c')
        assert confirmed['status'] == 'completed'
        legacy = OrchestrationRuntime(instance.engine, instance.store.path)
        assert legacy.ledger(state).summary()['used']['tool'] == 3
        assert instance.ledger(state).summary()['used']['tool'] == 3
        assert (await instance.appointment_service.confirm(action['id'], 'u', 'c'))['status'] == 'completed'
        assert len(instance.appointment_service.backend.appointments) == 1
    asyncio.run(run())
