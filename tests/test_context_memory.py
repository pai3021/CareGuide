import asyncio
import copy

import pytest

from careguide.context_memory import (ContextPolicy, ContextBudgetExceeded, estimate_tokens,
                                     use_context_policy, groups)
from careguide.llm_utils import create_chat_completion
from tests.test_session_archive import SUMMARY, settings
from tests.test_chat_runtime import ScriptedClient


def history():
    return [{'role': 'system', 'content': '不可执行未确认的预约。'}] + [
        {'role': 'user' if i % 2 == 0 else 'assistant', 'content': str(i)+'旧历史。'*60}
        for i in range(12)] + [{'role': 'user', 'content': '继续人民医院上午的预约，尚未确认'}]


def test_four_phase_summary_keeps_current_request_and_system_and_calls_once():
    async def scenario():
        client = ScriptedClient([{'content': SUMMARY}])
        policy = ContextPolicy(client, 'offline', settings())
        messages = history()
        original = copy.deepcopy(messages)
        prepared = await policy.prepare(messages)
        assert messages == original
        assert prepared[0] == original[0] and prepared[-1] == original[-1]
        assert any('未解决问题' in m['content'] for m in prepared)
        assert len(client.calls) == 1
        assert policy.events[-1]['after_tokens'] < policy.events[-1]['before_tokens']
    asyncio.run(scenario())


def test_summary_failure_preserves_input_and_cooldown_limits_calls():
    async def scenario():
        client = ScriptedClient([TimeoutError('fixture')])
        policy = ContextPolicy(client, 'offline', settings())
        messages = history()[:6]+history()[-1:]
        # Force a soft-pressure compression while remaining below the hard cap.
        policy.settings = settings(trigger_ratio=.4, target_ratio=.2)
        assert estimate_tokens({'messages': messages, 'tools': []}) < policy.settings.limit()
        assert await policy.prepare(messages) == messages
        assert await policy.prepare(messages) == messages
        assert len(client.calls) == 1
        assert policy.events[-1]['status'] == 'preserved'
    asyncio.run(scenario())


def test_oversized_current_input_fails_without_sending_or_deleting_it():
    async def scenario():
        client = ScriptedClient([])
        policy = ContextPolicy(client, 'offline', settings())
        messages = [{'role': 'user', 'content': '当前不可丢弃的请求'*3000}]
        with pytest.raises(ContextBudgetExceeded):
            await policy.prepare(messages)
        assert len(messages[0]['content']) > 10000
        assert client.calls == []
    asyncio.run(scenario())


def test_tool_group_is_atomic_and_pending_result_is_protected():
    async def scenario():
        policy = ContextPolicy(ScriptedClient([]), 'offline', settings())
        parent = {'role': 'assistant', 'content': None, 'tool_calls': [
            {'id': 'a', 'type': 'function', 'function': {'name': 'lookup', 'arguments': '{}'}}]}
        result = {'role': 'tool', 'tool_call_id': 'a', 'content': '未消费的新证据'*3000}
        messages = [{'role': 'user', 'content': '查号源'}, parent, result]
        assert len(groups(messages)) == 2
        with pytest.raises(ContextBudgetExceeded):
            await policy.prepare(messages)
        assert messages[-1] == result
        with pytest.raises(ContextBudgetExceeded):
            groups([result])
    asyncio.run(scenario())


def test_usage_anchor_invalidation_and_appended_delta():
    policy = ContextPolicy(None, 'model-a', settings())
    messages = [{'role': 'user', 'content': '你好'}]
    policy.observe(messages, [], 'model-a', {'usage': {'prompt_tokens': 100}})
    appended = messages + [{'role': 'assistant', 'content': '你好呀'}]
    count, method = policy.count(appended, [], 'model-a')
    assert count == 100 + estimate_tokens(appended[-1])
    assert method.startswith('provider_anchor')
    assert policy.count(appended, [{'name': 'new-tool'}], 'model-a')[1] == 'estimate_utf8'
    assert policy.count(appended, [], 'model-b')[1] == 'estimate_utf8'
    assert policy.count([{'role': 'user', 'content': 'changed'}], [], 'model-a')[1] == 'estimate_utf8'


def test_full_request_hook_accounts_for_system_and_tools_and_no_recursive_summary():
    async def scenario():
        client = ScriptedClient([{'content': SUMMARY}, {'content': '继续确认'}])
        policy = ContextPolicy(client, 'offline', settings())
        with use_context_policy(policy):
            await create_chat_completion(client, model='offline', messages=history()[1:],
                system='固定系统指令', tools=[{'name': 'lookup', 'description': '查询',
                    'input_schema': {'type': 'object', 'properties': {}}}], max_tokens=256)
        assert len(client.calls) == 2
        assert client.calls[-1]['messages'][0]['role'] == 'system'
        assert client.calls[-1]['tools'][0]['function']['name'] == 'lookup'
        assert len(policy.events) == 1 and policy.events[0]['summary_calls'] == 1
    asyncio.run(scenario())
