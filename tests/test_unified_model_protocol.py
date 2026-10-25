import asyncio
import json

import httpx
from openai import AsyncOpenAI
import pytest

from careguide.orchestration.budget import BudgetLedger, BudgetLimits
from careguide.orchestration.planner import PlanningFailure
from careguide.orchestration.understanding_planner import UnderstandingPlanner, build_context


def test_real_sdk_uses_one_decision_request_with_roles_context_and_usage(tmp_path):
    requests = []
    def handler(request):
        requests.append(json.loads(request.content))
        content = {'kind': 'clarify', 'question': '您指的是哪一种药品？'}
        return httpx.Response(200, json={'id': 'offline', 'object': 'chat.completion',
            'created': 1, 'model': 'script', 'choices': [{'index': 0, 'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': json.dumps(content, ensure_ascii=False)}}],
            'usage': {'prompt_tokens': 30, 'completion_tokens': 10, 'total_tokens': 40}})
    async def run():
        async with AsyncOpenAI(api_key='offline-test', max_retries=0,
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))) as client:
            planner = UnderstandingPlanner(client, 'script')
            ledger = BudgetLedger(tmp_path/'budget.sqlite3', 'w', 'u', 'c')
            context = build_context('它能和刚才的药一起吃吗', 'm1', '2026-10-24T10:00:00+00:00', [
                {'role': 'user', 'content': '刚才提到了两个药品'},
                {'role': 'assistant', 'content': '历史回答仅供上下文'}])
            decision, events = await planner.decide(context, budget=ledger)
            assert decision.kind == 'clarify'
            assert len(requests) == 1
            payload = json.loads(requests[0]['messages'][-1]['content'])
            assert payload['current_message']['id'] == 'm1'
            assert payload['workers']['medication']['mission']
            assert 'lookup_drug_label' in payload['allowed_actions']
            assert len(payload['user_sources']) == 2  # assistant is never an authorization source
            assert requests[0]['response_format'] == {'type': 'json_object'}
            assert ledger.summary()['used']['model'] == 1
            assert ledger.summary()['tokens']['total_tokens'] == 40
    asyncio.run(run())


def test_bad_decisions_have_one_bounded_repair_and_no_tool_dispatch(tmp_path):
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={'id': 'offline', 'object': 'chat.completion', 'created': 1,
            'model': 'script', 'choices': [{'index': 0, 'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': '{broken'}}]})
    async def run():
        async with AsyncOpenAI(api_key='offline-test', max_retries=0,
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))) as client:
            ledger = BudgetLedger(tmp_path/'budget.sqlite3', 'w', 'u', 'c')
            planner = UnderstandingPlanner(client)
            with pytest.raises(PlanningFailure):
                await planner.decide(build_context('请帮我安排', 'm', '2026-10-24T10:00:00+00:00'), budget=ledger)
            assert len(requests) == 2
            assert ledger.summary()['used'] == {'model': 2, 'tool': 0, 'retry': 1, 'replan': 0}
    asyncio.run(run())
