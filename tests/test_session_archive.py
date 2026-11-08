import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from careguide.archived_memory import ArchivedMemory
from careguide.session_store import SessionStore, TurnConflict
from careguide.task_context import SessionMemory
from careguide.context_memory import ContextSettings, estimate_tokens
from tests.test_chat_runtime import ScriptedClient


SUMMARY = '目标：继续预约。约束：上午，人民医院。关键实体：用户本人。已确认决策：尚未挂号。未解决问题：待确认。来源：历史用户原话。'


def settings(**overrides):
    return ContextSettings(**dict(window=8192, input_cap=2400, output_reserve=256,
        margin=128, summary_tokens=256, **overrides))


def test_archive_idempotency_pending_failure_and_scope(tmp_path):
    store = SessionStore(tmp_path/'archive.sqlite3')
    assert store.begin('alice', 'c', 'r', '上午挂号') is None
    assert store.messages('alice', 'c') == []
    with pytest.raises(TurnConflict):
        store.begin('alice', 'c', 'r', '下午挂号')
    store.complete('alice', 'c', 'r', {'response': '待确认'})
    assert store.begin('alice', 'c', 'r', '上午挂号') == {'response': '待确认'}
    assert len(SessionStore(store.path).messages('alice', 'c')) == 2
    assert store.messages('bob', 'c') == []
    store.begin('alice', 'c', 'failed', '发生错误的输入')
    store.fail('alice', 'c', 'failed')
    assert store.messages('alice', 'c')[-1]['status'] == 'failed'
    with pytest.raises(TurnConflict):
        store.begin('alice', 'c', 'failed', '发生错误的输入')
    store.delete('bob', 'c')
    assert len(store.messages('alice', 'c')) == 3


def test_restart_restores_archive_without_executing_tools(tmp_path):
    async def scenario():
        path = tmp_path/'archive.sqlite3'
        memory = ArchivedMemory(SessionMemory(), path)
        memory.begin_turn('alice', 'c', 'r', '请找人民医院上午的号')
        assert await memory.history('alice', 'c') == []  # current user is passed separately
        await memory.complete_turn('alice', 'c', 'r', '请找人民医院上午的号', {'response': '需要你确认'})
        restarted = ArchivedMemory(SessionMemory(), path)
        assert await restarted.history('alice', 'c') == [
            {'role': 'user', 'content': '请找人民医院上午的号'}, {'role': 'assistant', 'content': '需要你确认'}]
        assert await restarted.history('bob', 'c') == []
        assert restarted.memory.sessions[('alice', 'c')][-1]['content'] == '需要你确认'
    asyncio.run(scenario())


def test_rollup_boundary_survives_restart_and_raw_messages_are_complete(tmp_path):
    async def scenario():
        path = tmp_path/'archive.sqlite3'
        client = ScriptedClient([{'content': SUMMARY}]*5)
        memory = ArchivedMemory(SessionMemory(), path, client=client, model='offline', settings=settings())
        for i in range(12):
            memory.store.append('alice', 'c', 'user' if i % 2 == 0 else 'assistant', f'{i}:人民医院上午待确认。'+'历史信息。'*55)
        before = memory.store.messages('alice', 'c')
        history = await memory.history('alice', 'c')
        snapshot, revision = memory.store.snapshot('alice', 'c')
        assert revision == 1 and snapshot['covered_seq'] > 0
        assert history[0]['content'].startswith('[历史摘要')
        assert '人民医院' in history[0]['content'] and '待确认' in history[0]['content']
        restarted = ArchivedMemory(SessionMemory(), path, client=client, model='offline', settings=settings())
        restored = await restarted.history('alice', 'c')
        assert restored == history
        assert memory.store.messages('alice', 'c') == before
        assert len(client.calls) == 1
    asyncio.run(scenario())


def test_request_archives_original_tool_round_before_pruning(tmp_path):
    async def scenario():
        memory = ArchivedMemory(SessionMemory(), tmp_path/'a.sqlite3', client=None,
            model='offline', settings=settings())
        tool = {'role': 'tool', 'tool_call_id': 'a', 'content': '完整来源'*1000}
        with memory.request_context('alice', 'c', 'r') as policy:
            policy.archive([{'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': 'a', 'type': 'function', 'function': {'name': 'lookup', 'arguments': '{}'}}]}, tool])
        rows = memory.store.messages('alice', 'c')
        assert len(rows) == 2
        assert rows[-1]['content'] == tool['content']
        assert await memory.history('alice', 'c') == []  # tools are not fake chat turns
    asyncio.run(scenario())


def test_api_retries_return_archive_without_reexecuting_business_action(tmp_path):
    from careguide.api import ChatRequest, _execute_chat_result
    async def scenario():
        engine = SimpleNamespace(process=AsyncMock(return_value={
            'response': '待确认，未执行预约', 'request_id': 'existing-trace'}))
        memory = ArchivedMemory(SessionMemory(), tmp_path/'a.sqlite3')
        request = ChatRequest(message='挂号', user_id='alice', conv_id='c', request_id='same')
        first = await _execute_chat_result(request, (engine, memory, None))
        second = await _execute_chat_result(request, (engine, memory, None))
        assert first['response'] == second['response']
        assert engine.process.await_count == 1
        assert first['request_id'] == 'existing-trace'
        assert first['archive_request_id'] == 'same'
        assert len(memory.store.messages('alice', 'c')) == 2
    asyncio.run(scenario())


def test_episode_outage_retries_projection_without_losing_raw_archive(tmp_path):
    from scripts.benchmark_context_memory import FixtureRedis, fixture_manager, FixtureModel
    from careguide.task_context import PersistentMemory
    async def scenario():
        model = FixtureModel()
        manager = fixture_manager(tmp_path/'profile.sqlite3', model, FixtureRedis())
        manager._store_episodic = AsyncMock(return_value=False)
        memory = ArchivedMemory(PersistentMemory(manager), tmp_path/'archive.sqlite3', client=model,
                                model='offline', settings=settings())
        for i in range(12):
            memory.store.append('alice', 'c', 'user' if i % 2 == 0 else 'assistant',
                                str(i)+'人民医院上午仍需确认。'*25)
        await memory.history('alice', 'c')
        assert memory.store.pending_episodes('alice', 'c')
        first_id = manager._store_episodic.call_args.kwargs['doc_id']
        manager._store_episodic.return_value = True
        await memory.history('alice', 'c')
        assert not memory.store.pending_episodes('alice', 'c')
        assert manager._store_episodic.call_args.kwargs['doc_id'] == first_id
        assert len(memory.store.messages('alice', 'c')) == 12
    asyncio.run(scenario())


def test_failed_compression_cooldown_survives_adapter_restart(tmp_path):
    async def scenario():
        model = ScriptedClient([TimeoutError('fixture')])
        memory = ArchivedMemory(SessionMemory(), tmp_path/'archive.sqlite3', client=model,
            model='offline', settings=settings(trigger_ratio=.4, target_ratio=.2))
        for i in range(4):
            memory.store.append('alice', 'c', 'user' if i % 2 == 0 else 'assistant', str(i)+'旧信息。'*60)
        history = await memory.history('alice', 'c')
        restarted = ArchivedMemory(SessionMemory(), memory.store.path, client=model,
                                   model='offline', settings=memory.settings)
        assert await restarted.history('alice', 'c') == history
        assert len(model.calls) == 1
        assert memory.store.snapshot('alice', 'c', 'compression')[0]['cooldown_until'] > 0
    asyncio.run(scenario())


def test_api_failure_keeps_received_input_without_claiming_success(tmp_path):
    from careguide.api import ChatRequest, _execute_chat_result
    from fastapi import HTTPException
    async def scenario():
        engine = SimpleNamespace(process=AsyncMock(side_effect=RuntimeError('fixture')))
        memory = ArchivedMemory(SessionMemory(), tmp_path/'a.sqlite3')
        with pytest.raises(HTTPException):
            await _execute_chat_result(ChatRequest(message='挂号', user_id='alice', conv_id='c'),
                                       (engine, memory, None))
        rows = memory.store.messages('alice', 'c')
        assert len(rows) == 1 and rows[0]['status'] == 'failed'
    asyncio.run(scenario())


def test_archive_adapter_with_real_workflow_runtime_preserves_trace_and_tool_rounds(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from careguide.api import app
    from careguide.runtime import get_chat_runtime, get_registry
    from careguide.trace_store import validate_trace_completeness
    from tests.test_orchestration_api import configured
    from tests.test_model_planner import valid_plan, reply, TEXT
    from tests.test_chat_runtime import tool_reply
    client = ScriptedClient([reply(valid_plan()),
        tool_reply('lookup_drug_label', valid_plan()['tasks'][1]['arguments']),
        {'content': 'Only sourced information.'}])
    engine, original, monitor = configured(monkeypatch, tmp_path, client)
    # The factory now already archives by default. Test one adapter over its
    # backend rather than nesting archives and writing every turn twice.
    if isinstance(original, ArchivedMemory):
        original = original.memory
    memory = ArchivedMemory(original, tmp_path/'runtime.sqlite3', client=client, model='offline',
                            settings=ContextSettings())
    app.dependency_overrides[get_chat_runtime] = lambda: (engine, memory, monitor)
    try:
        with TestClient(app) as http:
            request = {'message': TEXT, 'user_id': 'alice', 'conv_id': 'c', 'request_id': 'archive-id'}
            first = http.post('/chat', json=request)
            assert first.status_code == 200, first.text
            count = len(client.calls)
            second = http.post('/chat', json=request)
            assert second.status_code == 200, second.text
            assert len(client.calls) == count
            result = first.json()
            assert result['archive_request_id'] == 'archive-id'
            assert result['workflow_status'] == 'completed'
            assert validate_trace_completeness(engine.traces.get(result['request_id']))['complete']
            rows = memory.store.messages('alice', 'c')
            assert [r['role'] for r in rows if r['role'] != 'tool_context'] == ['user', 'assistant']
            assert any(r['role'] == 'tool_context' for r in rows)
    finally:
        app.dependency_overrides.pop(get_chat_runtime, None)
        get_chat_runtime.cache_clear()
        get_registry.cache_clear()
