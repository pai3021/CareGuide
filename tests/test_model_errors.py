"""Offline SDK transport checks for user-visible provider failure categories."""
import asyncio
import json

import httpx
import pytest
from openai import AsyncOpenAI

from careguide.errors import model_failure
from careguide.orchestration.planner import ModelPlanner, PlanningFailure
from careguide.orchestration.understanding_planner import UnderstandingPlanner
from tests.test_answer_streaming import prepared


@pytest.mark.parametrize('planner_type', [ModelPlanner, UnderstandingPlanner])
@pytest.mark.parametrize('status,detail,expected', [
    (401, {'message': 'private-provider-secret'}, 'model_auth_failed'),
    (402, {'message': 'private-provider-secret'}, 'model_quota_exceeded'),
    (429, {'code': 'insufficient_quota'}, 'model_quota_exceeded'),
    (429, {'message': '余额不足'}, 'model_quota_exceeded'),
    (429, {'message': 'Too many requests'}, 'model_rate_limited'),
    (403, {'message': 'Forbidden'}, 'model_access_denied'),
    (504, {'message': 'Gateway timeout'}, 'model_timeout'),
    (503, {'message': 'Unavailable'}, 'model_unavailable'),
])
def test_planners_classify_actual_sdk_errors_without_leaking_or_retrying(planner_type, status, detail, expected):
    async def run():
        requests = []
        def handle(request):
            requests.append(request)
            return httpx.Response(status, json={'error': detail})
        async with AsyncOpenAI(api_key='offline-secret', max_retries=0,
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle))) as client:
            planner = planner_type(client)
            with pytest.raises(PlanningFailure) as caught:
                if planner_type is UnderstandingPlanner:
                    await planner.decide({})
                else:
                    await planner.plan('查询医生', {})
        assert caught.value.failure['code'] == expected
        assert len(requests) == 1
        assert 'private-provider-secret' not in json.dumps(caught.value.failure)
        assert 'offline-secret' not in json.dumps(caught.value.events)
        assert caught.value.failure['retryable'] == (expected in {
            'model_timeout', 'model_rate_limited', 'model_unavailable'})
    asyncio.run(run())


@pytest.mark.parametrize('kind,expected', [('timeout', 'model_timeout'), ('network', 'model_connection_failed')])
def test_sdk_transport_failures(kind, expected):
    async def run():
        def handle(request):
            error = httpx.ReadTimeout if kind == 'timeout' else httpx.ConnectError
            raise error('private-provider-secret', request=request)
        async with AsyncOpenAI(api_key='offline-secret', max_retries=0,
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle))) as client:
            with pytest.raises(PlanningFailure) as caught:
                await UnderstandingPlanner(client).decide({})
        assert caught.value.failure['code'] == expected
        assert 'private-provider-secret' not in json.dumps(caught.value.events)
    asyncio.run(run())


def test_program_timeout_and_unknown_error_are_distinct():
    assert model_failure(TimeoutError(), 'planning')['code'] == 'model_timeout'
    assert model_failure(RuntimeError('private'), 'planning')['code'] == 'model_unavailable'


def test_answer_failure_preserves_completed_results_and_displays_safe_notice(tmp_path):
    async def run():
        runtime, state, budget = await prepared(tmp_path)
        def handle(request):
            return httpx.Response(401, json={'error': {'message': 'private-provider-secret'}})
        async with AsyncOpenAI(api_key='offline-secret', max_retries=0,
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle))) as client:
            runtime.engine.client = client
            result = await runtime._respond(state, budget, '查医生', __import__('time').monotonic())
        assert result['workflow_status'] == 'completed'
        assert result['workflow']['tasks'][0]['status'] == 'succeeded'
        assert result['answer_generation']['failure']['code'] == 'model_auth_failed'
        assert '鉴权失败' in result['response']
        assert '已完成的查询结果仍按系统记录展示' in result['response']
        assert 'private-provider-secret' not in json.dumps(result)
    asyncio.run(run())
