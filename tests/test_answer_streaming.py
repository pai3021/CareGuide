"""Real SDK streaming with offline transport; no paid provider requests."""
import asyncio
import json

import httpx
from openai import AsyncOpenAI
import pytest

from careguide.api import ChatRequest, chat_stream, _stream_tasks
from careguide.llm_utils import track_token_usage
from careguide.orchestration.answering import generate_answer, answer_context
from careguide.orchestration.budget import BudgetLimits, use_budget
from careguide.orchestration.unified import UnifiedRuntime
from careguide.streaming import progress_events
from careguide.task_context import SessionMemory
from scripts.reproduce_orchestration import build
from tests.test_chat_stream import decode


def frame(text=None, *, finish=None, reasoning=None, tools=None, usage=None):
    delta = {}
    if text is not None:
        delta['content'] = text
    if reasoning:
        delta['reasoning_content'] = reasoning
    if tools:
        delta['tool_calls'] = tools
    chunk = dict(id='offline', object='chat.completion.chunk', created=1, model='mimo-v2.6-flash',
        choices=[] if usage else [dict(index=0, delta=delta, finish_reason=finish)])
    if usage:
        chunk['usage'] = usage
    return ('data: '+json.dumps(chunk, ensure_ascii=False)+'\n\n').encode()


def sdk(chunks, calls):
    def respond(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, headers={'content-type': 'text/event-stream'},
                              content=b''.join(chunks)+b'data: [DONE]\n\n')
    return AsyncOpenAI(api_key='offline-test', max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))


async def prepared(tmp_path):
    old, _ = build(tmp_path, [])
    runtime = UnifiedRuntime(old.engine, old.store.path)
    result = await runtime.process('查消化内科医生', 'u', 'c')
    state = runtime.store.get(result['workflow']['id'], 'u', 'c')
    return runtime, state, runtime.ledger(state)


def test_sdk_streams_only_content_preserves_usage_and_no_nested_budget(tmp_path):
    async def run():
        runtime, state, budget = await prepared(tmp_path)
        calls, events = [], []
        chunks = [frame(reasoning='hidden reasoning must never reach frontend'),
                  frame('查询到了'), frame('消化内科医生。[S1]'), frame(finish='stop'),
                  frame(usage=dict(prompt_tokens=20, completion_tokens=10, total_tokens=30))]
        async with sdk(chunks, calls) as client:
            with progress_events(lambda event, data: events.append((event, data))), track_token_usage() as usage:
                answer, diagnostic = await generate_answer(client, 'mimo-v2.6-flash', state, budget, '查医生')
        assert answer.startswith('查询到了消化内科医生。[S1]')
        assert '任务状态（系统记录）' in answer
        assert diagnostic['status'] == 'generated'
        assert ''.join(d['text'] for e, d in events if e == 'delta') == '查询到了消化内科医生。[S1]'
        assert [d['offset'] for e, d in events if e == 'delta'] == [0, 4]
        assert 'hidden reasoning' not in json.dumps(events)
        assert calls[0]['stream'] is True and calls[0]['stream_options']['include_usage'] is True
        assert calls[0]['thinking'] == {'type': 'disabled'}
        assert 'tools' not in calls[0] and 'response_format' not in calls[0]
        assert budget.summary()['used']['model'] == 1
        assert budget.summary()['tokens']['total_tokens'] == usage.total_tokens == 30
        assert budget.summary()['reservations'] == {'settled': 2}  # one tool plus one model
        assert len(calls) == 1
    asyncio.run(run())


@pytest.mark.parametrize('chunks', [
    [frame('医生资料。[S999]'), frame(finish='stop')],
    [frame('医生资料没有引用。'), frame(finish='stop')],
    [frame('已为您预约。[S1]'), frame(finish='stop')],
    [frame('医生资料。[S1]')],
    [frame('医生资料。[S1]'), frame(finish='length')],
    [frame(tools=[dict(index=0, id='x', type='function', function=dict(name='prepare_appointment', arguments='{}'))])],
    [frame(finish='stop')],
])
def test_bad_answers_reset_draft_and_keep_business_results(tmp_path, chunks):
    async def run():
        runtime, state, budget = await prepared(tmp_path)
        calls, events = [], []
        async with sdk(chunks, calls) as client:
            with progress_events(lambda e, d: events.append((e, d))):
                answer, diagnostic = await generate_answer(client, 'mimo-v2.6-flash', state, budget, '查医生')
        assert answer is None and diagnostic['status'] == 'deterministic_fallback'
        assert events[-1][0] == 'answer_reset'
        assert budget.summary()['used']['model'] == len(calls) == 1
        assert runtime.store.get(state.id, 'u', 'c').tasks['goal1'].result['items']
    asyncio.run(run())


def test_missing_usage_is_not_invented_and_pending_has_no_answer_call(tmp_path):
    async def run():
        _, state, budget = await prepared(tmp_path)
        calls = []
        async with sdk([frame('医生资料。[S1]'), frame(finish='stop')], calls) as client:
            answer, _ = await generate_answer(client, 'script', state, budget, '查医生')
            assert answer
            assert budget.summary()['tokens']['calls_without_usage'] == 1
            state.status = 'waiting_user'
            answer, diagnostic = await generate_answer(client, 'script', state, budget, '继续')
            assert answer is None and diagnostic['status'] == 'not_needed' and len(calls) == 1
    asyncio.run(run())


def test_deadline_covers_entire_stream_and_failure_is_charged(tmp_path):
    class Slow(httpx.AsyncByteStream):
        closed = False
        async def __aiter__(self):
            yield frame('未完成的回答')
            await asyncio.Event().wait()
        async def aclose(self):
            self.closed = True
    async def run():
        import time
        _, state, budget = await prepared(tmp_path)
        wire, events = Slow(), []
        async with AsyncOpenAI(api_key='offline', max_retries=0, http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(lambda request: httpx.Response(200,
                    headers={'content-type': 'text/event-stream'}, stream=wire)))) as client:
            with use_budget(budget, deadline=time.monotonic()+.2), progress_events(lambda e, d: events.append((e, d))):
                answer, diagnostic = await generate_answer(client, 'script', state, budget, '查医生')
        assert answer is None and diagnostic['status'] == 'deterministic_fallback'
        assert events[-1][0] == 'answer_reset' and wire.closed
        assert budget.summary()['used']['model'] == 1
        assert budget.summary()['reservations'] == {'settled': 1, 'unknown': 1}
    asyncio.run(run())


def test_real_sse_delta_precedes_model_finish_final_memory_and_trace_agree(tmp_path):
    async def run():
        runtime, _, _ = await prepared(tmp_path)
        release = asyncio.Event()
        class Gated(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield frame('找到医生信息')
                await release.wait()
                yield frame('，供您查看。[S1]')
                yield frame(finish='stop')
                yield b'data: [DONE]\n\n'
        async with AsyncOpenAI(api_key='offline', max_retries=0, http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(lambda request: httpx.Response(200,
                    headers={'content-type': 'text/event-stream'}, stream=Gated())))) as client:
            runtime.engine.client, runtime.engine.model = client, 'mimo-v2.6-flash'
            memory = SessionMemory()
            response = await chat_stream(ChatRequest(message='查消化内科医生', user_id='u', conv_id='live'),
                                         (runtime, memory, None))
            events, iterator = [], response.body_iterator
            while not any(e == 'delta' for e, _, _ in events):
                events.extend(decode(await asyncio.wait_for(anext(iterator), 5)))
            assert not release.is_set() and not any(e == 'result' for e, _, _ in events)
            release.set()
            async for chunk in iterator:
                events.extend(decode(chunk))
            final = events[-1][1]
            assert events[-1][0] == 'result'
            assert final['answer_generation']['status'] == 'generated'
            assert final['response'].startswith('找到医生信息，供您查看。[S1]')
            assert final['budget']['used']['model'] == 1 and final['budget']['used']['tool'] == 1
            assert runtime.traces.get(final['request_id'])['final']['final_response'] == final['response']
            assert (await memory.history('u', 'live'))[-1]['content'] == final['response']
    asyncio.run(run())


def test_answer_backpressure_is_lossless_and_disconnect_unblocks_producer(tmp_path):
    from careguide.streaming import emit_answer
    async def run():
        runtime, _, _ = await prepared(tmp_path)
        process, reached, finished = runtime.process, asyncio.Event(), asyncio.Event()
        async def burst(*args, **kwargs):
            for index in range(100):
                await emit_answer('delta', answer_id='burst', offset=index, text='字', provisional=True)
            reached.set()
            result = await process(*args, **kwargs)
            finished.set()
            return result
        runtime.process = burst
        value = (runtime, SessionMemory(), None)
        response = await chat_stream(ChatRequest(message='查消化内科医生', conv_id='slow'), value)
        iterator = response.body_iterator
        await anext(iterator)
        await anext(iterator)  # resume past the session yield to start the producer
        await asyncio.sleep(.05)
        assert not reached.is_set()  # a slow receiver backpressures the content producer
        events = []
        async for chunk in iterator:
            events.extend(decode(chunk))
        assert len([e for e, _, _ in events if e == 'delta']) == 100
        reached.clear(); finished.clear()
        response = await chat_stream(ChatRequest(message='查消化内科医生', conv_id='disconnect'), value)
        iterator = response.body_iterator
        await anext(iterator)
        await anext(iterator)
        await asyncio.sleep(.05)
        await iterator.aclose()
        await asyncio.wait_for(finished.wait(), 5)
        await asyncio.gather(*tuple(_stream_tasks))
    asyncio.run(run())


def test_answer_reset_supersedes_full_queue_without_stalling_fallback(tmp_path):
    from careguide.streaming import emit, emit_answer
    async def run():
        runtime, _, _ = await prepared(tmp_path)
        process = runtime.process
        async def saturated(*args, **kwargs):
            for _ in range(100):
                emit('stage', message='正在处理')
            await emit_answer('answer_reset', answer_id='a', reason='timeout')
            return await process(*args, **kwargs)
        runtime.process = saturated
        response = await chat_stream(ChatRequest(message='查消化内科医生', conv_id='reset'),
                                     (runtime, SessionMemory(), None))
        events = []
        async for chunk in response.body_iterator:
            events.extend(decode(chunk))
        assert any(e == 'answer_reset' for e, _, _ in events)
        assert events[-1][0] == 'result'
    asyncio.run(run())


def test_concurrent_workflow_change_retracts_answer_and_trace_uses_current_state(tmp_path):
    from careguide.orchestration.contracts import TaskStatus
    async def run():
        import time
        runtime, state, budget = await prepared(tmp_path)
        events = []
        def changed(request):
            def cancel(value):
                value.tasks['goal1'].status = TaskStatus.CANCELLED
                value.status = 'cancelled'
            runtime.store.mutate(state.id, state.actor, state.conversation, cancel,
                                 plan_version=state.version, run_id=state.active_run)
            return httpx.Response(200, headers={'content-type': 'text/event-stream'},
                content=frame('旧的查询结果。[S1]')+frame(finish='stop')+b'data: [DONE]\n\n')
        async with AsyncOpenAI(api_key='offline', max_retries=0,
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(changed))) as client:
            runtime.engine.client, runtime.engine.model = client, 'mimo-v2.6-flash'
            with progress_events(lambda e, d: events.append((e, d))):
                result = await runtime._respond(state, budget, '查医生', time.monotonic())
        assert result['answer_generation']['status'] == 'stale_result_fallback'
        assert result['workflow_status'] == 'cancelled'
        assert '旧的查询结果' not in result['response']
        assert events[-1][0] == 'answer_reset'
        assert runtime.traces.get(result['request_id'])['final']['final_response'] == result['response']
    asyncio.run(run())


def test_exhausted_model_budget_skips_generation_without_losing_results(tmp_path):
    async def run():
        runtime, state, budget = await prepared(tmp_path)
        while budget.summary()['used']['model'] < budget.limits.model:
            budget.finish(budget.reserve('model'), 0)
        calls = []
        async with sdk([], calls) as client:
            answer, diagnostic = await generate_answer(client, 'mimo-v2.6-flash', state, budget, '查医生')
        assert answer is None and diagnostic['status'] == 'deterministic_budget_fallback'
        assert not calls and runtime.store.get(state.id, 'u', 'c').tasks['goal1'].result['items']
    asyncio.run(run())


def test_multi_task_answer_needs_sources_from_each_successful_branch(tmp_path):
    async def run():
        runtime, _, _ = await prepared(tmp_path)
        result = await runtime.process('查消化内科医生，同时查询精氨酸布洛芬片0.2g说明书', 'u', 'multi')
        state = runtime.store.get(result['workflow']['id'], 'u', 'multi')
        context = answer_context(state, '两个查询')
        references = {source['task_id']: source['id'] for source in context['verified_results']}
        assert len(references) == 2
        text = '医生结果可供查看。['+references['goal1']+'] 药品信息请结合说明书阅读。['+references['goal2']+']'
        calls = []
        async with sdk([frame(text), frame(finish='stop')], calls) as client:
            answer, diagnostic = await generate_answer(client, 'script', state, runtime.ledger(state), '两个查询')
        assert diagnostic['status'] == 'generated' and text in answer
        assert '资料来源' in answer and len(diagnostic['sources']) >= 2
    asyncio.run(run())
