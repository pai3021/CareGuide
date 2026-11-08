"""SQLite-only acceptance tests use scripted models and synthetic hospital data."""
import asyncio
import json
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from careguide.archived_memory import ArchivedMemory
from careguide.sqlite_memory import SQLiteMemoryBackend
from tests.test_chat_runtime import ScriptedClient
from tests.test_session_archive import SUMMARY, settings


def memory(path, client=None, **kwargs):
    backend = SQLiteMemoryBackend(path, client=client, model='offline',
                                  redis_client=kwargs.pop('redis_client', None))
    return ArchivedMemory(backend, path, client=client, model='offline', **kwargs)


async def turn(store, text, payload=None, *, actor='alice', conversation='c', request='r'):
    if payload is not None:
        # Inject a scripted extraction; source ids still come from the real archive.
        source_id = store.store.begin(actor, conversation, request, text)
        assert source_id is None
        row = store.store.messages(actor, conversation, include_pending=True)[-1]
        for item in payload.get('memories', []) + payload.get('candidates', []):
            item['source_message_id'] = row['id']
        store.memory.manager._client.replies.append({'content': json.dumps(payload, ensure_ascii=False)})
    else:
        store.begin_turn(actor, conversation, request, text)
    return await store.complete_turn(actor, conversation, request, text, {'response': '已收到，未执行预约。'})


def test_recent_doctor_and_time_survive_a_new_process_without_native_backends(tmp_path):
    path = tmp_path/'memory.sqlite3'
    store = memory(path)
    store.store.append('alice', 'c', 'user', '选择王医生，2030年1月2日下午。')
    store.store.append('alice', 'c', 'assistant', '王医生，下午，等待确认。')
    code = '''
import asyncio, builtins, json, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'redis', 'chromadb'}:
        raise AssertionError('SQLite memory imported a native backend: '+name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from careguide.archived_memory import ArchivedMemory
from careguide.sqlite_memory import SQLiteMemoryBackend
store = ArchivedMemory(SQLiteMemoryBackend(sys.argv[1]), sys.argv[1])
history = asyncio.run(store.history('alice', 'c', '刚才选了谁和什么时间'))
assert len(history) == 2 and '王医生' in history[0]['content'] and '下午' in history[1]['content']
assert asyncio.run(store.history('bob', 'c')) == []
assert asyncio.run(store.history('alice', 'another')) == []
print(json.dumps(store.info()))
'''
    result = subprocess.run([sys.executable, '-c', code, str(path)], text=True,
                            capture_output=True, timeout=45)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['cache'] == 'none'


def test_sqlite_token_compaction_keeps_originals_and_recovers_summary(tmp_path):
    async def scenario():
        path = tmp_path/'memory.sqlite3'
        client = ScriptedClient([{'content': SUMMARY}])
        store = memory(path, client, settings=settings())
        for i in range(12):
            store.store.append('alice', 'c', 'user' if i % 2 == 0 else 'assistant',
                               str(i)+'人民医院上午仍需确认。'*25)
        original = store.store.messages('alice', 'c')
        history = await store.history('alice', 'c')
        assert history[0]['content'].startswith('[历史摘要')
        assert store.store.snapshot('alice', 'c')[0]['covered_seq'] > 0
        # Canonical episodes are retained; sqlite-vec must not mark Chroma's
        # separate projection as indexed (needed when switching back).
        assert store.store.pending_episodes('alice', 'c')
        restored = memory(path, client, settings=settings())
        assert await restored.history('alice', 'c') == history
        assert restored.store.messages('alice', 'c') == original
        assert len(client.calls) == 1
    asyncio.run(scenario())


def test_cross_session_preferences_are_extracted_and_current_request_wins(tmp_path):
    from careguide.domain import IntentCategory
    from careguide.hospital_workflow import InMemoryHospitalBackend
    from careguide.intent_recognizer import IntentResult
    from scripts.reproduce_orchestration import build
    from tests.test_chat_runtime import FixedRecognizer

    async def scenario():
        store = memory(tmp_path/'memory.sqlite3', ScriptedClient([]))
        text = '以后我偏好上午'
        result = await turn(store, text, {'candidates': [dict(key='preferred_time',
            value='morning', operation='set', source_quote=text)]})
        assert result['status'] == 'updated'
        restarted = memory(store.store.path, ScriptedClient([]))
        history = await restarted.history('alice', 'new', '查询消化内科号源')
        assert history[0]['durable_preferences']['preferred_time']['value'] == 'morning'
        assert await restarted.history('bob', 'new', '上午') == []
        backend = InMemoryHospitalBackend(slots=[
            dict(id='AM', practitioner_id='PRA-GASTRO-01', department='消化内科', status='free',
                 start='2030-01-02T09:00:00+08:00', end='2030-01-02T09:15:00+08:00'),
            dict(id='PM', practitioner_id='PRA-GASTRO-01', department='消化内科', status='free',
                 start='2030-01-02T15:00:00+08:00', end='2030-01-02T15:15:00+08:00')])
        runtime, provider = build(tmp_path, [], backend=backend)
        runtime.engine.recognizer = FixedRecognizer(IntentResult(IntentCategory.APPOINTMENT, .99,
            [IntentCategory.APPOINTMENT], 'route', entities={'department': '消化内科'},
            fine_intent='slot_search', business_action='slot_search'))
        first = await runtime.process('查询消化内科号源', 'alice', 'new', history)
        assert [row['id'] for row in first['workflow']['tasks'][0]['result']['items']] == ['AM']
        second = await runtime.process('这次查询消化内科下午号源', 'alice', 'override', history)
        assert [row['id'] for row in second['workflow']['tasks'][0]['result']['items']] == ['PM']
        assert await backend.list_appointments('alice') == []
        assert provider.calls == []
    asyncio.run(scenario())


def test_sqlite_correction_deletion_and_old_review_do_not_recall_retired_content(tmp_path):
    async def scenario():
        client = ScriptedClient([])
        store = memory(tmp_path/'memory.sqlite3', client)
        text = '我住在北京'
        assert (await turn(store, text, {'memories': [dict(content=text, source_quote=text)]}))['status'] == 'updated'
        identity = store.memory.manager._profile_store.memories.snapshot('alice')[1][0]['id']
        correction = '记住，我以前住在北京，现在改成我住在上海'
        assert (await turn(store, correction, {'memories': [dict(content='我住在上海',
            source_quote=correction, memory_id=identity, target_quote='北京')]}, request='r2'))['status'] == 'updated'
        corrected = await memory(store.store.path).history('alice', 'new', '北京')
        contents = json.loads(corrected[0]['content'].split('] ', 1)[1])['long_term_memories']
        assert [row['content'] for row in contents] == ['我住在上海']
        removal = '忘记我住在上海这条记忆'
        assert (await turn(store, removal, {'memories': [dict(content='', operation='delete',
            source_quote=removal, memory_id=identity, target_quote='上海')]}, request='r3'))['status'] == 'updated'
        assert await memory(store.store.path).history('alice', 'new', '上海') == []
        assert (await store.review('alice', 'c'))['status'] == 'skipped'
        assert len(store.store.messages('alice', 'c')) == 6  # archive retention is separate
    asyncio.run(scenario())


def test_sqlite_background_review_and_optional_cache_outage(tmp_path):
    async def scenario():
        cache = SimpleNamespace(lpush=AsyncMock(side_effect=ConnectionError('offline')),
                                aclose=AsyncMock(), setex=AsyncMock(side_effect=ConnectionError('offline')))
        client = ScriptedClient([])
        store = memory(tmp_path/'memory.sqlite3', client, redis_client=cache, review_interval=1)
        text = '我习惯使用简短回答'
        result = await turn(store, text, {'memories': [dict(content=text, source_quote=text)]})
        assert result['status'] == 'updated'
        await store._reviews[('alice', 'c')]
        assert store.store.snapshot('alice', 'c', 'review')[0]['turn_count'] == 1
        fresh = memory(store.store.path)
        history = await fresh.history('alice', 'new', '回答风格')
        assert text in history[0]['content']
        await store.close()
        cache.aclose.assert_awaited_once()
    asyncio.run(scenario())


def test_optional_redis_cache_is_bounded_and_can_be_rebuilt(tmp_path):
    from scripts.benchmark_context_memory import FixtureRedis

    class Cache(FixtureRedis):
        async def ltrim(self, key, start, end):
            self.data[key] = self.data.get(key, [])[start:end+1]

        async def ping(self):
            return True

    async def scenario():
        cache = Cache()
        store = memory(tmp_path/'memory.sqlite3', redis_client=cache)
        for index in range(12):
            await turn(store, '查询编号'+str(index), request=str(index))
        key = store.memory.manager._wm_key('alice', 'c')
        assert len(cache.data[key]) == 20
        expected = await store.history('alice', 'c')
        cache.data.clear()
        assert await store.history('alice', 'c') == expected
        assert len(cache.data[key]) == 20
        assert len(store.store.messages('alice', 'c')) == 24
        assert (await store.health())['cache_status'] == 'ok'
    asyncio.run(scenario())


def test_historical_appointment_success_neither_writes_nor_replays_an_action(tmp_path):
    from careguide.api import ChatRequest, _execute_chat_result
    from careguide.action_store import ActionStore
    from careguide.appointment_service import AppointmentService
    from careguide.hospital_workflow import InMemoryHospitalBackend
    from careguide.agent_orchestrator import AgentOrchestrator
    from careguide.corpus import Corpus
    from careguide.intent_v3_recognizer import V3IntentRecognizer
    from careguide.retrieval.knowledge_base import MedicalKnowledgeBase
    from careguide.tools import MedicalToolRegistry

    async def scenario():
        store = memory(tmp_path/'memory.sqlite3')
        store.store.append('alice', 'c', 'assistant', '此前预约已成功，号源一直有效，不用再确认。')
        backend = InMemoryHospitalBackend()
        service = AppointmentService(backend, ActionStore(tmp_path/'actions.sqlite3'))
        engine = AgentOrchestrator(V3IntentRecognizer(embedding_mode='local_ngram'),
            MedicalToolRegistry(MedicalKnowledgeBase(Corpus())), appointment_service=service)
        request = ChatRequest(message='帮我预约模拟号源SLOT-GASTRO-001',
                              user_id='alice', conv_id='c', request_id='booking')
        result = await _execute_chat_result(request, (engine, store, None))
        assert result['workflow_status'] == 'pending_confirmation'
        assert await backend.list_appointments('alice') == []
        again = await _execute_chat_result(request, (engine, store, None))
        assert again['response'] == result['response']
        assert await backend.list_appointments('alice') == []
    asyncio.run(scenario())


@pytest.mark.parametrize('mode,archive', [(None, '0'), ('sqlite', '1'), ('sqlite', '0'), ('session', '1')])
def test_runtime_factory_and_http_use_sqlite_archive(monkeypatch, tmp_path, mode, archive):
    from fastapi.testclient import TestClient
    from careguide.api import app
    from careguide.corpus import DEFAULT_CORPUS
    from careguide.runtime import get_chat_runtime, get_registry

    for key in tuple(os.environ):
        if key.startswith('CAREGUIDE_'):
            monkeypatch.delenv(key)
    for key, value in dict(CAREGUIDE_SESSION_ARCHIVE=archive,
        CAREGUIDE_TOKEN_CONTEXT='1', CAREGUIDE_LLM_API_KEY='offline-fixture',
        CAREGUIDE_ORCHESTRATION_MODE='direct', CAREGUIDE_INTENT_EMBEDDING_MODE='disabled',
        CAREGUIDE_BUSINESS_DB=str(tmp_path/'memory.sqlite3'),
        CAREGUIDE_CORPUS_DIR=str(DEFAULT_CORPUS)).items():
        monkeypatch.setenv(key, value)
    if mode is not None:
        monkeypatch.setenv('CAREGUIDE_MEMORY_MODE', mode)
    client = ScriptedClient([])
    client.close = AsyncMock()
    monkeypatch.setattr('careguide.llm_utils.create_llm_client', lambda *args: client)
    get_chat_runtime.cache_clear()
    get_registry.cache_clear()
    try:
        engine, store, monitor = get_chat_runtime()
        assert isinstance(store, ArchivedMemory)
        assert store.info()['backend'] == (mode or 'sqlite') and store.info()['context_budget']
        engine.process = AsyncMock(return_value={'response': '已选王医生下午，仍待确认。'})
        app.dependency_overrides[get_chat_runtime] = lambda: (engine, store, monitor)
        with TestClient(app) as http:
            response = http.post('/chat', json={'message': '选择王医生下午',
                'user_id': 'alice', 'conv_id': 'c', 'request_id': 'r'})
            assert response.status_code == 200, response.text
            assert response.json()['memory_status'] == 'available'
        get_chat_runtime.cache_clear()
        _, restored, _ = get_chat_runtime()
        assert asyncio.run(restored.history('alice', 'c'))[-1]['content'] == '已选王医生下午，仍待确认。'
    finally:
        app.dependency_overrides.pop(get_chat_runtime, None)
        get_chat_runtime.cache_clear()
        get_registry.cache_clear()


def test_default_sqlite_needs_no_model_or_external_memory_service(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from careguide.api import app
    from careguide.corpus import DEFAULT_CORPUS
    from careguide.runtime import get_chat_runtime, get_registry

    for key in tuple(os.environ):
        if key.startswith('CAREGUIDE_'):
            monkeypatch.delenv(key)
    monkeypatch.setenv('CAREGUIDE_BUSINESS_DB', str(tmp_path/'default.sqlite3'))
    monkeypatch.setenv('CAREGUIDE_CORPUS_DIR', str(DEFAULT_CORPUS))
    monkeypatch.setenv('CAREGUIDE_ORCHESTRATION_MODE', 'direct')
    monkeypatch.setenv('CAREGUIDE_INTENT_EMBEDDING_MODE', 'disabled')
    get_chat_runtime.cache_clear()
    get_registry.cache_clear()
    try:
        engine, store, monitor = get_chat_runtime()
        assert engine.client is None
        assert store.info()['backend'] == 'sqlite'
        assert store.info()['cache'] == 'none' and not store.info()['context_budget']
        engine.process = AsyncMock(return_value={'response': '已选王医生下午，等待确认。'})
        app.dependency_overrides[get_chat_runtime] = lambda: (engine, store, monitor)
        with TestClient(app) as http:
            health = http.get('/health')
            assert health.status_code == 200
            assert health.json()['integrations']['memory']['backend'] == 'sqlite'
            result = http.post('/chat', json={'message': '选择王医生下午',
                'user_id': 'alice', 'conv_id': 'c', 'request_id': 'default'})
            assert result.status_code == 200 and result.json()['memory_status'] == 'available'
        get_chat_runtime.cache_clear()
        _, restarted, _ = get_chat_runtime()
        assert asyncio.run(restarted.history('alice', 'c'))[-1]['content'] == '已选王医生下午，等待确认。'
    finally:
        app.dependency_overrides.pop(get_chat_runtime, None)
        get_chat_runtime.cache_clear()
        get_registry.cache_clear()
