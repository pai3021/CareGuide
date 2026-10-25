"""SSE uses the real chat pipeline and publishes progress before task completion."""
import asyncio
import json

import httpx
import pytest

from careguide.api import app, chat_stream, ChatRequest, _stream_tasks
from careguide.runtime import get_chat_runtime
from careguide.orchestration.unified import UnifiedRuntime
from careguide.task_context import SessionMemory
from scripts.reproduce_orchestration import build


@pytest.fixture
def runtime(tmp_path):
    old, _ = build(tmp_path, [])
    engine = UnifiedRuntime(old.engine, old.store.path)
    value = (engine, SessionMemory(), None)
    app.dependency_overrides[get_chat_runtime] = lambda: value
    yield value
    app.dependency_overrides.pop(get_chat_runtime, None)


def decode(text):
    output = []
    for block in text.split('\n\n'):
        fields = dict(line.split(': ', 1) for line in block.splitlines() if ': ' in line)
        if 'data' in fields:
            output.append((fields['event'], json.loads(fields['data']), int(fields['id'])))
    return output


def test_stream_and_json_use_same_workers_budgets_and_final_contract(runtime):
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            response = await client.post('/chat/stream', json={'message': '查消化内科医生', 'user_id': 'alice'})
            assert response.status_code == 200
            assert response.headers['content-type'].startswith('text/event-stream')
            assert response.headers['x-accel-buffering'] == 'no'
            events = decode(response.text)
            assert events[0][0] == 'session' and events[-1][0] == 'result'
            assert [e[2] for e in events] == list(range(len(events)))
            final = events[-1][1]
            assert events[0][1]['conversation_id'] == final['conversation_id']
            assert final['workflow_status'] == 'completed'
            assert final['budget']['used']['tool'] == 1
            assert any(t['status'] == 'running' for event, data, _ in events if event == 'workflow'
                       for t in data['workflow']['tasks'])
            assert any(t['status'] == 'succeeded' for event, data, _ in events if event == 'workflow'
                       for t in data['workflow']['tasks'])
            plain = (await client.post('/chat', json={'message': '查消化内科医生', 'user_id': 'bob'})).json()
            assert plain['response'] == final['response']
            assert plain['budget']['used'] == final['budget']['used']
            assert len(await runtime[1].history('alice', final['conversation_id'])) == 2
    asyncio.run(run())


def test_real_progress_precedes_worker_completion_and_disconnect_preserves_execution(runtime):
    async def run():
        engine = runtime[0]
        entered, release = asyncio.Event(), asyncio.Event()
        worker = engine.worker.run
        async def gated(*args):
            entered.set()
            await release.wait()
            return await worker(*args)
        engine.worker.run = gated
        response = await chat_stream(ChatRequest(message='查消化内科医生', user_id='u', conv_id='live'), runtime)
        iterator = response.body_iterator
        await anext(iterator)  # session is delivered before execution
        seen = []
        while not any(event == 'workflow' and any(t['status'] == 'running' for t in data['workflow']['tasks'])
                      for event, data, _ in seen):
            seen.extend(decode(await asyncio.wait_for(anext(iterator), 5)))
        await asyncio.wait_for(entered.wait(), 5)
        assert not release.is_set()
        assert not any(event == 'result' for event, _, _ in seen)
        # A dropped client must not cancel or submit the business task twice.
        await iterator.aclose()
        release.set()
        await asyncio.wait_for(asyncio.gather(*tuple(_stream_tasks)), 5)
        state = engine.store.latest('u', 'live')
        assert state.status == 'completed'
        assert state.tasks['goal1'].attempts == 1
        assert engine.ledger(state).summary()['used']['tool'] == 1
        assert len(await runtime[1].history('u', 'live')) == 2
    asyncio.run(run())


def test_error_frames_and_pre_header_validation_are_sanitized(runtime):
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            invalid = await client.post('/chat/stream', json={'message': '继续', 'workflow_event': {
                'kind': 'answer', 'workflow_id': 'w', 'plan_version': 1}})
            assert invalid.status_code == 422 and 'text/event-stream' not in invalid.headers['content-type']
            foreign = await client.post('/chat/stream', json={'message': '继续', 'user_id': 'other', 'conv_id': 'c',
                'workflow_event': {'kind': 'run', 'workflow_id': 'private', 'plan_version': 1}})
            terminal = decode(foreign.text)[-1]
            assert terminal[0] == 'error' and terminal[1]['status'] == 404
            async def broken(*args, **kwargs):
                raise RuntimeError('private secret exception detail')
            runtime[0].process = broken
            failed = await client.post('/chat/stream', json={'message': '查医生', 'conv_id': 'c'})
            assert decode(failed.text)[-1][1]['status'] == 503
            assert 'private secret' not in failed.text
    asyncio.run(run())


def test_stream_wait_resume_versions_and_confirmation_remain_authoritative(runtime):
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            async def send(text, event=None, user='u'):
                payload = {'message': text, 'user_id': user, 'conv_id': 'c'}
                if event:
                    payload['workflow_event'] = event
                return decode((await client.post('/chat/stream', json=payload)).text)[-1]
            _, first, _ = await send('查消化内科医生的号源并预约')
            identity = first['workflow']['id']
            def answer(task, field, value):
                return {'kind': 'answer', 'workflow_id': identity, 'plan_version': 1,
                    'task_id': task, 'task_revision': 1, 'values': {field: value}, 'event_id': task}
            event = answer('slots', 'practitioner_id', 'PRA-GASTRO-01')
            assert (await send('PRA-GASTRO-01', event, user='other'))[1]['status'] == 404
            assert (await send('PRA-GASTRO-01', {**event, 'plan_version': 2}))[1]['status'] == 409
            await send('PRA-GASTRO-01', event)
            _, prepared, _ = await send('SLOT-GASTRO-001', answer('booking', 'slot_id', 'SLOT-GASTRO-001'))
            action = prepared['action_requests'][0]
            assert prepared['workflow_status'] == 'pending_confirmation'
            for _ in range(2):
                confirmed = await client.post('/actions/'+action['id']+'/confirm', json={'user_id': 'u', 'conv_id': 'c'})
                assert confirmed.json()['status'] == 'completed'
            assert len(runtime[0].appointment_service.backend.appointments) == 1
    asyncio.run(run())


def test_concurrent_users_receive_only_their_own_progress(runtime):
    async def run():
        async def collect(user, text):
            response = await chat_stream(ChatRequest(message=text, user_id=user, conv_id=user), runtime)
            output = []
            async for chunk in response.body_iterator:
                output.extend(decode(chunk))
            return output
        left, right = await asyncio.gather(collect('left', '查消化内科医生'), collect('right', '查心内科医生'))
        for user, text, events in [('left', '查消化内科医生', left), ('right', '查心内科医生', right)]:
            assert events[0][1]['conversation_id'] == user
            final = events[-1][1]
            identity = final['workflow']['id']
            assert all(data['workflow']['id'] == identity and all(t['label'] == text for t in data['workflow']['tasks'])
                       for event, data, _ in events if event == 'workflow')
        assert left[-1][1]['workflow']['id'] != right[-1][1]['workflow']['id']
    asyncio.run(run())


def test_heartbeat_does_not_require_worker_completion(runtime, monkeypatch):
    import importlib
    api_module = importlib.import_module('careguide.api')
    async def run():
        release = asyncio.Event()
        process = runtime[0].process
        async def gated(*args, **kwargs):
            await release.wait()
            return await process(*args, **kwargs)
        runtime[0].process = gated
        response = await chat_stream(ChatRequest(message='查消化内科医生', user_id='u', conv_id='heart'), runtime)
        iterator = response.body_iterator
        assert decode(await anext(iterator))[0][0] == 'session'
        assert decode(await anext(iterator))[0][0] == 'stage'
        monkeypatch.setattr(api_module, '_STREAM_HEARTBEAT_SECONDS', .01)
        assert await asyncio.wait_for(anext(iterator), 1) == ': heartbeat\n\n'
        release.set()
        events = []
        async for chunk in iterator:
            events.extend(decode(chunk))
        assert events[-1][0] == 'result'
    asyncio.run(run())


def test_backpressure_can_drop_progress_but_never_terminal_result(runtime):
    from careguide.streaming import emit
    async def run():
        process = runtime[0].process
        async def burst(*args, **kwargs):
            for _ in range(200):
                emit('stage', stage='executing', message='正在执行')
            return await process(*args, **kwargs)
        runtime[0].process = burst
        response = await chat_stream(ChatRequest(message='查消化内科医生', conv_id='burst'), runtime)
        events = []
        async for chunk in response.body_iterator:
            events.extend(decode(chunk))
        assert len(events) < 200
        assert events[-1][0] == 'result' and events[-1][1]['workflow_status'] == 'completed'
    asyncio.run(run())


def test_asgi_disconnect_stops_delivery_without_cancelling_persisted_task(runtime):
    async def run():
        entered, release, disconnected = asyncio.Event(), asyncio.Event(), asyncio.Event()
        worker = runtime[0].worker.run
        async def gated(*args):
            entered.set()
            await release.wait()
            return await worker(*args)
        runtime[0].worker.run = gated
        payload = json.dumps({'message': '查消化内科医生', 'user_id': 'u', 'conv_id': 'asgi-disconnect'}).encode()
        received, bodies = False, []
        async def receive():
            nonlocal received
            if not received:
                received = True
                return {'type': 'http.request', 'body': payload, 'more_body': False}
            await disconnected.wait()
            return {'type': 'http.disconnect'}
        async def send(message):
            if message['type'] == 'http.response.start':
                assert message['status'] == 200
            if message['type'] == 'http.response.body':
                chunk = message.get('body', b'').decode()
                bodies.append(chunk)
                for event, data, _ in decode(chunk):
                    if event == 'workflow' and any(t['status'] == 'running' for t in data['workflow']['tasks']):
                        disconnected.set()
        scope = {'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.0'}, 'method': 'POST',
            'scheme': 'http', 'path': '/chat/stream', 'raw_path': b'/chat/stream', 'root_path': '',
            'query_string': b'', 'http_version': '1.1', 'headers': [(b'content-type', b'application/json')],
            'server': ('test', 80), 'client': ('127.0.0.1', 1234)}
        delivery = asyncio.create_task(app(scope, receive, send))
        await asyncio.wait_for(entered.wait(), 5)
        await asyncio.wait_for(delivery, 5)
        assert not any(event == 'result' for event, _, _ in decode(''.join(bodies)))
        release.set()
        await asyncio.wait_for(asyncio.gather(*tuple(_stream_tasks)), 5)
        state = runtime[0].store.latest('u', 'asgi-disconnect')
        assert state.status == 'completed' and state.tasks['goal1'].attempts == 1
        assert runtime[0].ledger(state).summary()['used']['tool'] == 1
    asyncio.run(run())
