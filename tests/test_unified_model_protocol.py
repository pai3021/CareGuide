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


def test_live_observed_completion_mismatch_gets_catalog_guided_bounded_repair(tmp_path):
    requests = []
    text = '帮我看看有哪些消化内科医生，只查询，不预约'
    def handler(request):
        body = json.loads(request.content)
        payload = json.loads(body['messages'][-1]['content'])
        requests.append(payload)
        completion = 'source_evidence' if len(requests) == 1 else payload['allowed_actions']['search_doctors']['completion']
        decision = {'kind': 'new', 'plan': {'goal': text, 'tasks': [dict(
            id='doctor', fragment='帮我看看有哪些消化内科医生', action='search_doctors',
            agent='appointment', arguments={'department': '消化内科'}, completion=completion)]}}
        return httpx.Response(200, json={'id': 'offline', 'object': 'chat.completion', 'created': 1,
            'model': 'script', 'choices': [{'index': 0, 'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': json.dumps(decision, ensure_ascii=False)}}]})
    async def run():
        async with AsyncOpenAI(api_key='offline', max_retries=0,
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))) as client:
            budget = BudgetLedger(tmp_path/'budget.sqlite3', 'w', 'u', 'c')
            decision, events = await UnderstandingPlanner(client).decide(
                build_context(text, 'm', '2026-10-25T10:00:00+00:00'), budget=budget)
            assert decision.plan.tasks[0].completion == 'items'
            assert [event['status'] for event in events] == ['rejected', 'validated']
            assert requests[0]['allowed_actions']['prepare_appointment']['completion'] == 'prepared'
            assert requests[0]['allowed_actions']['lookup_drug_label']['completion'] == 'source_evidence'
            assert 'completion' in requests[1]['validation_feedback']['hint']
            assert budget.summary()['used']['model'] == 2 and budget.summary()['used']['retry'] == 1
    asyncio.run(run())
