import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient

from careguide.api import app
from careguide.domain import IntentCategory
from careguide.hospital_workflow import FHIRHospitalBackend, InMemoryHospitalBackend
from careguide.intent_recognizer import IntentResult
from careguide.runtime import get_chat_runtime
from careguide.task_context import PersistentMemory
from careguide.conversation_memory import MemoryContext
from scripts.reproduce_orchestration import build
from tests.test_chat_runtime import FixedRecognizer
from tests.test_profile_memory import candidate, manager, message


def test_persistent_morning_is_reusable_but_current_afternoon_query_takes_priority(tmp_path):
    async def run():
        memory = manager(tmp_path, [message('以后我偏好上午')], [{'candidates': [candidate()]}])
        actor = 'example-user'
        assert (await memory.update_profile(actor, 'preference-conversation'))['status'] == 'updated'
        profile = await memory._get_profile(actor)
        memory.get_context = AsyncMock(return_value=MemoryContext([], [], profile, ''))
        history = await PersistentMemory(memory).history(actor, 'query-conversation')
        backend = InMemoryHospitalBackend(slots=[
            dict(id='SLOT-AM', practitioner_id='PRA-GASTRO-01', department='消化内科', status='free',
                 start='2030-01-02T09:00:00+08:00', end='2030-01-02T09:15:00+08:00'),
            dict(id='SLOT-PM', practitioner_id='PRA-GASTRO-01', department='消化内科', status='free',
                 start='2030-01-02T15:00:00+08:00', end='2030-01-02T15:15:00+08:00')])
        runtime, provider = build(tmp_path, [], backend=backend)
        runtime.engine.recognizer = FixedRecognizer(IntentResult(IntentCategory.APPOINTMENT, .99,
            [IntentCategory.APPOINTMENT], 'route', entities={'department': '消化内科'},
            fine_intent='slot_search', business_action='slot_search'))
        first = await runtime.process('查询消化内科号源', actor, 'default-preference', history)
        first_task = first['workflow']['tasks'][0]
        assert [item['id'] for item in first_task['result']['items']] == ['SLOT-AM']
        saved = runtime.store.get(first['workflow']['id'], actor, 'default-preference')
        assert saved.tasks['direct'].origins['time_period']['kind'] == 'long_term_preference'
        second = await runtime.process('这次查询消化内科下午号源', actor, 'current-override', history)
        second_task = second['workflow']['tasks'][0]
        assert [item['id'] for item in second_task['result']['items']] == ['SLOT-PM']
        saved = runtime.store.get(second['workflow']['id'], actor, 'current-override')
        assert saved.tasks['direct'].origins['time_period']['kind'] == 'user_text'
        assert (await memory._get_profile(actor))['preferred_time']['value'] == 'morning'
        negative = await runtime.process('查询消化内科号源，不要上午', actor, 'negative-preference', history)
        assert negative['workflow_status'] == 'waiting_user'
        assert negative['workflow']['tasks'][0]['missing_fields'] == ['time_period']
        assert negative['budget']['used']['tool'] == 0
        chosen = await runtime.process('下午', actor, 'negative-preference', history,
            workflow_event=dict(kind='answer', workflow_id=negative['workflow']['id'], plan_version=1,
                task_id='direct', task_revision=1, values={'time_period': 'afternoon'}))
        assert chosen['workflow_status'] == 'completed'
        assert [item['id'] for item in chosen['workflow']['tasks'][0]['result']['items']] == ['SLOT-PM']
        assert provider.calls == []
    asyncio.run(run())


def test_relative_date_keeps_original_message_calendar_after_midnight(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    from careguide.orchestration.contracts import PlanProposal
    runtime, _ = build(tmp_path, [])
    fragment = '查消化内科明天号源'
    state = runtime.store.create('u', 'c', PlanProposal(goal=fragment, tasks=[dict(id='slots',
        fragment=fragment, action='list_available_slots', agent='appointment',
        arguments={'department': '消化内科', 'date': '2030-01-03'}, completion='items')]),
        source_text=fragment, source_at='2030-01-01T15:55:00+00:00')
    class AfterMidnight(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2030, 1, 1, 16, 5, tzinfo=timezone.utc).astimezone(tz)
    monkeypatch.setattr('careguide.orchestration.worker.datetime', AfterMidnight)
    bound = runtime.worker.prepare(state.tasks['slots'], state)
    assert bound.arguments['date'] == '2030-01-02'
    assert bound.missing_fields == []


def test_fhir_date_and_period_filter_use_shanghai_calendar_for_utc_slots():
    async def run():
        calls = []
        def handle(request):
            calls.append(request)
            return httpx.Response(200, json={'resourceType': 'Bundle', 'entry': [
                {'resource': {'resourceType': 'Slot', 'id': 'morning', 'status': 'free',
                    'start': '2030-01-01T23:00:00Z', 'end': '2030-01-01T23:15:00Z'}},
                {'resource': {'resourceType': 'Slot', 'id': 'evening', 'status': 'free',
                    'start': '2030-01-02T12:00:00Z', 'end': '2030-01-02T12:15:00Z'}}]})
        backend = FHIRHospitalBackend('http://synthetic.test/fhir')
        await backend.client.aclose()
        backend.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        result = await backend.list_available_slots(date='2030-01-02', time_period='morning')
        assert [item['id'] for item in result] == ['morning']
        assert calls[0].url.params['start'] == 'ge2030-01-02T00:00:00+08:00'
        await backend.close()
    asyncio.run(run())


@pytest.mark.parametrize('stall', ['history', 'add'])
def test_stalled_memory_cannot_hide_completed_business_reply(monkeypatch, stall):
    async def blocked(*args, **kwargs):
        await asyncio.Event().wait()
    memory = SimpleNamespace(history=AsyncMock(return_value=[]), add=AsyncMock(return_value=None))
    setattr(memory, stall, blocked)
    business = {'response': '已完成且保留业务结果', 'workflow_status': 'completed', 'medical_answering_enabled': False}
    engine = SimpleNamespace(process=AsyncMock(return_value=dict(business)))
    monkeypatch.setattr('careguide.api._MEMORY_READ_TIMEOUT_S', .01)
    monkeypatch.setattr('careguide.api._MEMORY_WRITE_TIMEOUT_S', .01)
    app.dependency_overrides[get_chat_runtime] = lambda: (engine, memory, None)
    try:
        response = TestClient(app).post('/chat', json={'message': '查询资料', 'user_id': 'memory-test', 'conv_id': stall})
        assert response.status_code == 200
        assert response.json()['response'] == business['response']
        assert response.json()['workflow_status'] == 'completed'
        assert response.json()['memory_status'] == 'unavailable'
        assert engine.process.await_count == 1
    finally:
        app.dependency_overrides.clear()
