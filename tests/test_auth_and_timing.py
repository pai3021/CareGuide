import asyncio
import hashlib
import json

import httpx
import pytest

from careguide.api import app
from careguide.runtime import get_chat_runtime
from careguide.task_context import SessionMemory
from careguide.orchestration.unified import UnifiedRuntime
from scripts.reproduce_orchestration import build


@pytest.fixture
def secured(tmp_path, monkeypatch):
    credentials = tmp_path/'access.json'
    values = {'alice': 'a'*40, 'bob': 'b'*40, 'admin': 'c'*40}
    credentials.write_text(json.dumps({'users': [{'user_id': user, 'role': 'admin' if user == 'admin' else 'user',
        'token_sha256': hashlib.sha256(token.encode()).hexdigest()} for user, token in values.items()]}), encoding='utf-8')
    monkeypatch.setenv('CAREGUIDE_AUTH_FILE', str(credentials))
    old, _ = build(tmp_path, [])
    engine = UnifiedRuntime(old.engine, old.store.path)
    from careguide.business_store import BusinessStore
    engine.engine.business_store = BusinessStore(tmp_path/'business.sqlite3')
    memory = SessionMemory()
    app.dependency_overrides[get_chat_runtime] = lambda: (engine, memory, None)
    yield engine, memory, values
    app.dependency_overrides.pop(get_chat_runtime, None)


def test_identity_binds_missing_actor_and_prevents_cross_user_reads_and_writes(secured):
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            assert (await client.get('/auth/me')).status_code == 401
            alice = {'Authorization': 'Bearer '+secured[2]['alice']}
            bob = {'Authorization': 'Bearer '+secured[2]['bob']}
            identity = (await client.get('/auth/me', headers=alice)).json()
            assert identity == {'mode': 'bearer', 'user_id': 'alice', 'role': 'user'}
            assert (await client.post('/chat', headers=alice, json={'message': '查医生', 'user_id': 'bob'})).status_code == 403
            result = (await client.post('/chat', headers=alice, json={'message': '查消化内科医生', 'conv_id': 'c'})).json()
            assert result['workflow_status'] == 'completed'
            assert len(await secured[1].history('alice', 'c')) == 2
            assert not await secured[1].history('demo-user', 'c')
            path = '/workflows/'+result['workflow']['id']
            assert (await client.get(path+'?user_id=alice&conv_id=c', headers=bob)).status_code == 403
            assert (await client.get(path+'?user_id=bob&conv_id=c', headers=bob)).status_code == 404
            trace = '/trace/tool/'+result['request_id']
            assert (await client.get(trace, headers=alice)).json()['found'] is True
            assert (await client.get(trace, headers=bob)).json()['found'] is False
            assert not (await client.get('/trace/tools', headers=bob)).json()['traces']
    asyncio.run(run())


def test_admin_routes_fail_before_handlers_and_credentials_never_echo(secured):
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            headers = {'Authorization': 'Bearer '+secured[2]['alice']}
            for path, method in [('/eval/run', 'POST'), ('/knowledge/submissions', 'GET'), ('/monitor', 'GET'), ('/skills/reload', 'POST')]:
                response = await client.request(method, path, headers=headers)
                assert response.status_code == 403 and secured[2]['alice'] not in response.text
            headers = {'Authorization': 'Bearer '+secured[2]['admin']}
            assert (await client.get('/auth/me', headers=headers)).json()['role'] == 'admin'
            assert (await client.get('/knowledge/submissions', headers=headers)).status_code == 200
    asyncio.run(run())


def test_bad_auth_configuration_fails_closed_and_demo_mode_is_explicit(monkeypatch):
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            monkeypatch.setenv('CAREGUIDE_REQUIRE_AUTH', 'true')
            monkeypatch.delenv('CAREGUIDE_AUTH_FILE', raising=False)
            assert (await client.get('/auth/me')).status_code == 503
            monkeypatch.setenv('CAREGUIDE_REQUIRE_AUTH', '1')
            assert (await client.get('/auth/me')).status_code == 503
            monkeypatch.setenv('CAREGUIDE_REQUIRE_AUTH', 'invalid')
            assert (await client.get('/auth/me')).status_code == 503
            monkeypatch.setenv('CAREGUIDE_REQUIRE_AUTH', 'false')
            assert (await client.get('/auth/me')).json()['mode'] == 'local_demo'
    asyncio.run(run())


def test_request_timings_are_scoped_and_include_memory_and_understanding(secured):
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            async def request(user):
                return (await client.post('/chat', json={'message': '查消化内科医生', 'conv_id': 'timing'},
                    headers={'Authorization': 'Bearer '+secured[2][user]})).json()
            results = await asyncio.gather(request('alice'), request('bob'))
            for result in results:
                timing = result['timings']
                assert timing['total_ms'] > 0 and timing['first_content_ms'] is None
                assert {'memory_read', 'memory_write', 'understanding', 'tool'}.issubset(timing['phases_ms'])
                assert all(value >= 0 for value in timing['phases_ms'].values())
                assert secured[0].traces.get(result['request_id'])['execution']['timings']
    asyncio.run(run())


def test_authenticated_sse_checks_identity_before_headers_and_binds_background_work(secured):
    from tests.test_chat_stream import decode
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            denied = await client.post('/chat/stream', json={'message': '查医生'})
            assert denied.status_code == 401 and 'text/event-stream' not in denied.headers['content-type']
            response = await client.post('/chat/stream', headers={'Authorization': 'Bearer '+secured[2]['alice']},
                json={'message': '查消化内科医生', 'conv_id': 'secure-stream'})
            final = decode(response.text)[-1][1]
            assert final['workflow_status'] == 'completed'
            assert len(await secured[1].history('alice', 'secure-stream')) == 2
            assert not await secured[1].history('bob', 'secure-stream')
            assert final['timings']['phases_ms']['tool'] >= 0
    asyncio.run(run())
