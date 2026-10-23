import asyncio
import json
import os
import traceback
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from careguide import runtime
from careguide.api import app
from careguide.corpus import DEFAULT_CORPUS
from careguide.domain import IntentCategory
from careguide.intent_recognizer import IntentRecognizer
from careguide.trace_store import validate_trace_completeness
from careguide.orchestration.budget import BudgetExhausted, BudgetLimits
from careguide.intent_recognizer import IntentResult
import pytest
from tests.test_chat_runtime import FixedRecognizer, ScriptedClient, result, tool_reply
from tests.test_model_planner import TEXT, reply, valid_plan


def configured(monkeypatch, tmp_path, client):
    for key in tuple(os.environ):
        if key.startswith('CAREGUIDE_'):
            monkeypatch.delenv(key)
    monkeypatch.setenv('CAREGUIDE_BUSINESS_DB', str(tmp_path/'runtime.sqlite3'))
    monkeypatch.setenv('CAREGUIDE_HOSPITAL_BACKEND', 'memory')
    monkeypatch.setenv('CAREGUIDE_LLM_API_KEY', 'offline-scripted-test')
    monkeypatch.setenv('CAREGUIDE_CORPUS_DIR', str(DEFAULT_CORPUS))
    client.close = AsyncMock()
    monkeypatch.setattr('careguide.llm_utils.create_llm_client', lambda *args: client)
    runtime.get_chat_runtime.cache_clear()
    runtime.get_registry.cache_clear()
    engine, memory, monitor = runtime.get_chat_runtime()
    # Force the exact single-label limitation: Planner must still see both tasks.
    engine.engine.recognizer = FixedRecognizer(result(IntentCategory.APPOINTMENT))
    process = engine.process
    async def diagnosed(*args, **kwargs):
        try:
            return await process(*args, **kwargs)
        except Exception:
            # Only scripted inputs exist in this test; expose the original
            # internal traceback when the public API correctly sanitizes it.
            traceback.print_exc()
            raise
    engine.process = diagnosed
    return engine, memory, monitor


def test_production_factory_and_http_reach_real_model_planner_and_scoped_worker(monkeypatch, tmp_path):
    plan = valid_plan()
    client = ScriptedClient([reply(plan), tool_reply('lookup_drug_label', plan['tasks'][1]['arguments']),
                             {'content': 'Do not use this narrative as medical advice.'}])
    engine, _, _ = configured(monkeypatch, tmp_path, client)
    try:
        with TestClient(app, raise_server_exceptions=False) as http:
            response = http.post('/chat', json={'message': TEXT, 'user_id': 'alice', 'conv_id': 'c'})
            assert response.status_code == 200, response.text
            data = response.json()
            assert data['execution_mode'] == 'validated_task_plan'
            assert data['workflow_status'] == 'completed'
            assert len(data['workflow']['tasks']) == 2
            assert data['budget']['used']['model'] == 3
            assert data['budget']['used']['tool'] == 2
            payload = json.loads(client.calls[0]['messages'][-1]['content'])
            assert payload['user_request'] == TEXT
            assert payload['intent_signals']['intents'] == ['appointment']
            assert data['evidence']
            assert 'Do not use' not in data['response']
            trace = http.get('/trace/tool/'+data['request_id']).json()['trace']
            assert validate_trace_completeness(trace)['complete']
            assert trace['execution']['workflow']['id'] == data['workflow']['id']
            assert len(trace['execution']['tool_calls']) == 2
            assert trace['execution']['budget']['tokens']['calls_without_usage'] == 3
    finally:
        runtime.get_chat_runtime.cache_clear()
        runtime.get_registry.cache_clear()


def test_http_wait_restore_budget_and_reject_stale_or_other_owner_event(monkeypatch, tmp_path):
    text = TEXT.replace('0.2g', '')
    plan = valid_plan()
    plan['goal'] = text
    plan['tasks'][1]['fragment'] = plan['tasks'][1]['fragment'].replace('0.2g', '')
    del plan['tasks'][1]['arguments']['specification']
    args = {**plan['tasks'][1]['arguments'], 'specification': '0.2g'}
    client = ScriptedClient([reply(plan), tool_reply('lookup_drug_label', args), {'content': 'done'}])
    engine, _, _ = configured(monkeypatch, tmp_path, client)
    try:
        with TestClient(app, raise_server_exceptions=False) as http:
            first = http.post('/chat', json={'message': text, 'user_id': 'alice', 'conv_id': 'c'})
            assert first.status_code == 200, first.text
            waiting = first.json()
            assert waiting['workflow_status'] == 'waiting_user'
            assert waiting['budget']['used']['tool'] == 1
            workflow_id = waiting['workflow']['id']
            event = dict(kind='answer', workflow_id=workflow_id, plan_version=1,
                         task_id='drug', task_revision=1, values={'specification': '0.2g'}, event_id='spec-answer')
            body = {'message': '0.2g', 'user_id': 'alice', 'conv_id': 'c', 'workflow_event': event}
            foreign = http.post('/chat', json={**body, 'user_id': 'bob'})
            assert foreign.status_code == 404
            stale = http.post('/chat', json={**body, 'workflow_event': {**event, 'plan_version': 2}})
            assert stale.status_code == 409
            resumed = http.post('/chat', json=body)
            assert resumed.status_code == 200, resumed.text
            data = resumed.json()
            assert data['workflow_status'] == 'completed'
            assert data['budget']['used']['tool'] == 2
            assert data['budget']['used']['model'] == 3
            assert next(task for task in data['workflow']['tasks'] if task['id'] == 'doctors')['status'] == 'succeeded'
            state = engine.store.get(workflow_id, 'alice', 'c')
            assert state.tasks['doctors'].attempts == 1
            assert len(client.calls) == 3
    finally:
        runtime.get_chat_runtime.cache_clear()
        runtime.get_registry.cache_clear()


def test_simple_request_keeps_direct_path_without_planner(monkeypatch, tmp_path):
    client = ScriptedClient([])
    engine, _, _ = configured(monkeypatch, tmp_path, client)
    engine.engine.recognizer = IntentRecognizer(embedding_mode='disabled')
    # Direct role fallback is deterministic when its model is explicitly absent.
    engine.engine.client = None
    data = asyncio.run(engine.process('查消化内科医生', 'alice', 'simple'))
    assert 'workflow' not in data
    assert engine.store.latest('alice', 'simple') is None
    assert client.calls == []
    runtime.get_chat_runtime.cache_clear()
    runtime.get_registry.cache_clear()


def test_http_local_replan_preserves_unaffected_wait_and_rejects_old_answer(monkeypatch, tmp_path):
    text = TEXT.replace('0.2g', '')
    plan = valid_plan()
    plan['goal'] = text
    plan['tasks'][1]['fragment'] = plan['tasks'][1]['fragment'].replace('0.2g', '')
    del plan['tasks'][1]['arguments']['specification']
    changed = {**plan['tasks'][0], 'fragment': '改查心血管内科医生',
               'arguments': {'department': '心血管内科'}}
    patch = {'base_version': 1, 'reason': '用户修改医生科室', 'update': [changed]}
    client = ScriptedClient([reply(plan), reply(patch)])
    engine, _, _ = configured(monkeypatch, tmp_path, client)
    try:
        with TestClient(app) as http:
            first = http.post('/chat', json={'message': text, 'user_id': 'alice', 'conv_id': 'c'}).json()
            identity = first['workflow']['id']
            updated = http.post('/chat', json={'message': '改查心血管内科医生', 'user_id': 'alice', 'conv_id': 'c',
                'workflow_event': {'kind': 'change', 'workflow_id': identity, 'plan_version': 1}})
            assert updated.status_code == 200, updated.text
            data = updated.json()
            assert data['workflow']['version'] == 2
            tasks = {task['id']: task for task in data['workflow']['tasks']}
            assert tasks['drug']['revision'] == 1
            assert tasks['drug']['status'] == 'waiting_input'
            assert tasks['doctors']['revision'] == 2
            assert tasks['doctors']['result']['items'][0]['department'] == '心血管内科'
            assert data['budget']['used']['replan'] == 1
            assert data['budget']['used']['tool'] == 2
            old_answer = http.post('/chat', json={'message': '0.2g', 'user_id': 'alice', 'conv_id': 'c',
                'workflow_event': {'kind': 'answer', 'workflow_id': identity, 'plan_version': 1,
                    'task_id': 'drug', 'task_revision': 1, 'values': {'specification': '0.2g'}}})
            assert old_answer.status_code == 409
            assert len(client.calls) == 2
    finally:
        runtime.get_chat_runtime.cache_clear()
        runtime.get_registry.cache_clear()


def test_http_dependency_selection_prepare_confirm_and_resume_never_duplicate_booking(monkeypatch, tmp_path):
    clause = '查消化内科医生的号源并预约'
    text = clause+'，同时看精氨酸布洛芬片0.2g说明书'
    plan = valid_plan()
    plan['goal'] = text
    plan['tasks'][0]['fragment'] = clause
    plan['tasks'].extend([
        dict(id='slots', fragment=clause, agent='appointment', action='list_available_slots',
            depends_on=['doctors'], inputs={'practitioner_id': dict(task_id='doctors', path=['items'],
                mode='selected', item_field='id')}, completion='items'),
        dict(id='booking', fragment=clause, agent='appointment', action='prepare_appointment',
            depends_on=['slots'], inputs={'slot_id': dict(task_id='slots', path=['items'],
                mode='selected', item_field='id')}, completion='prepared')])
    client = ScriptedClient([reply(plan), tool_reply('lookup_drug_label', plan['tasks'][1]['arguments']),
                             {'content': 'done'}])
    engine, _, _ = configured(monkeypatch, tmp_path, client)
    try:
        with TestClient(app) as http:
            response = http.post('/chat', json={'message': text, 'user_id': 'alice', 'conv_id': 'c'})
            assert response.status_code == 200, response.text
            first = response.json()
            identity = first['workflow']['id']
            def answer(task, field, value):
                response = http.post('/chat', json={'message': value, 'user_id': 'alice', 'conv_id': 'c',
                    'workflow_event': dict(kind='answer', workflow_id=identity, plan_version=1,
                        task_id=task, task_revision=1, values={field: value})})
                assert response.status_code == 200, response.text
                return response.json()
            assert first['workflow_status'] == 'waiting_user'
            selected = answer('slots', 'practitioner_id', 'PRA-GASTRO-01')
            assert selected['workflow_status'] == 'waiting_user'
            prepared = answer('booking', 'slot_id', 'SLOT-GASTRO-001')
            assert prepared['workflow_status'] == 'pending_confirmation'
            assert engine.appointment_service.backend.appointments == {}
            action_id = prepared['action_requests'][0]['id']
            confirmed = http.post('/actions/'+action_id+'/confirm', json={'user_id': 'alice', 'conv_id': 'c'})
            assert confirmed.status_code == 200, confirmed.text
            assert confirmed.json()['status'] == 'completed'
            replay = http.post('/actions/'+action_id+'/confirm', json={'user_id': 'alice', 'conv_id': 'c'})
            assert replay.json()['status'] == 'completed'
            resumed = http.post('/chat', json={'message': '继续', 'user_id': 'alice', 'conv_id': 'c',
                'workflow_event': dict(kind='run', workflow_id=identity, plan_version=1)})
            assert resumed.status_code == 200, resumed.text
            assert resumed.json()['workflow_status'] == 'completed'
            state = engine.store.get(identity, 'alice', 'c')
            assert all(task.attempts == 1 for task in state.tasks.values())
            assert len(engine.appointment_service.backend.appointments) == 1
            assert len(client.calls) == 3
            assert resumed.json()['budget']['used']['tool'] == 5
    finally:
        runtime.get_chat_runtime.cache_clear()
        runtime.get_registry.cache_clear()


def test_simple_missing_parameter_promotes_direct_route_without_planner_and_survives_reopen(monkeypatch, tmp_path):
    client = ScriptedClient([])
    engine, _, _ = configured(monkeypatch, tmp_path, client)
    engine.engine.client = None
    engine.engine.recognizer = IntentRecognizer(embedding_mode='disabled')
    async def run():
        first = await engine.process('查询精氨酸布洛芬片说明书', 'alice', 'simple-wait')
        assert first['execution_mode'] == 'direct_task_workflow'
        assert first['workflow_status'] == 'waiting_user'
        assert first['budget']['used']['model'] == first['budget']['used']['tool'] == 0
        from careguide.orchestration.runtime import OrchestrationRuntime
        reopened = OrchestrationRuntime(engine.engine, engine.store.path)
        final = await reopened.process('0.2g', 'alice', 'simple-wait')
        assert final['workflow_status'] == 'completed'
        assert final['workflow']['id'] == first['workflow']['id']
        assert final['budget']['used']['tool'] == 1
        assert client.calls == []
    try:
        asyncio.run(run())
    finally:
        runtime.get_chat_runtime.cache_clear()
        runtime.get_registry.cache_clear()


def test_direct_booking_confirmation_keeps_budget_even_without_outer_runtime_wrapper(monkeypatch, tmp_path):
    client = ScriptedClient([])
    engine, _, _ = configured(monkeypatch, tmp_path, client)
    engine.engine.client = None
    engine.limits = BudgetLimits(model=0, tool=1)
    engine.engine.recognizer = FixedRecognizer(IntentResult(IntentCategory.APPOINTMENT, .99,
        [IntentCategory.APPOINTMENT], 'route', entities={'slot_id': 'SLOT-GASTRO-001'},
        fine_intent='appointment_create', business_action='appointment_create'))
    async def run():
        first = await engine.process('帮我预约模拟号源SLOT-GASTRO-001', 'alice', 'simple-booking')
        assert first['execution_mode'] == 'direct_task_workflow'
        assert first['workflow_status'] == 'pending_confirmation'
        assert first['budget']['used']['tool'] == 1
        action_id = first['action_requests'][0]['id']
        # Calling the existing service directly models a rollback to direct
        # outer routing. The binding must still enforce persisted quota.
        with pytest.raises(BudgetExhausted):
            await engine.appointment_service.confirm(action_id, 'alice', 'simple-booking')
        assert engine.appointment_service.actions.get(action_id)['status'] == 'pending_confirmation'
        assert engine.appointment_service.backend.appointments == {}
        cancelled = await engine.process('不用预约', 'alice', 'simple-booking')
        assert cancelled['workflow_status'] == 'cancelled'
        assert engine.appointment_service.actions.get(action_id)['status'] == 'expired'
        assert client.calls == []
    try:
        asyncio.run(run())
    finally:
        runtime.get_chat_runtime.cache_clear()
        runtime.get_registry.cache_clear()
