import asyncio
import json

import pytest
import httpx
from openai import AsyncOpenAI

from careguide.llm_utils import create_llm_client
from careguide.llm_utils import create_chat_completion
from careguide.orchestration.budget import BudgetLedger, BudgetLimits, use_budget
from careguide.orchestration.planner import ModelPlanner, PlanningFailure
from tests.test_chat_runtime import ScriptedClient


TEXT = '查消化内科医生，同时看精氨酸布洛芬片0.2g说明书'


def valid_plan():
    return {'goal': TEXT, 'tasks': [
        {'id': 'doctors', 'fragment': '查消化内科医生', 'agent': 'appointment',
         'action': 'search_doctors', 'arguments': {'department': '消化内科'},
         'completion': 'items'},
        {'id': 'drug', 'fragment': '看精氨酸布洛芬片0.2g说明书', 'agent': 'medication',
         'action': 'lookup_drug_label', 'arguments': {'drug_name': '精氨酸布洛芬片',
            'dosage_form': '片剂', 'specification': '0.2g'}, 'completion': 'source_evidence'}]}


def reply(value):
    return {'content': json.dumps(value, ensure_ascii=False)}


def test_real_planner_prompt_preserves_full_text_despite_single_intent():
    client = ScriptedClient([reply(valid_plan())])
    decision = asyncio.run(ModelPlanner(client).plan(TEXT, {'agent_route': 'appointment'}))
    assert len(decision.proposal.tasks) == 2
    payload = json.loads(client.calls[0]['messages'][-1]['content'])
    assert payload['user_request'] == TEXT
    assert payload['intent_signals']['agent_route'] == 'appointment'
    assert 'lookup_drug_label' in payload['allowed_actions']
    assert client.calls[0]['response_format'] == {'type': 'json_object'}


def test_invalid_dag_gets_one_bounded_repair(tmp_path):
    invalid = valid_plan()
    invalid['tasks'][0]['depends_on'] = ['drug']
    invalid['tasks'][1]['depends_on'] = ['doctors']
    client = ScriptedClient([reply(invalid), reply(valid_plan())])
    budget = BudgetLedger(tmp_path/'budget.sqlite3', 'workflow', 'u', 'c')
    result = asyncio.run(ModelPlanner(client).plan(TEXT, {}, budget=budget))
    assert len(result.events) == 2
    assert result.events[0]['failure']['code'] == 'invalid_parameters'
    assert result.events[1]['status'] == 'validated'
    assert budget.summary()['used']['model'] == 2
    assert budget.summary()['used']['retry'] == 1


def test_unauthorized_plan_stops_before_execution_without_retry():
    invalid = valid_plan()
    invalid['tasks'][0]['agent'] = 'medication'
    client = ScriptedClient([reply(invalid), reply(valid_plan())])
    with pytest.raises(PlanningFailure) as caught:
        asyncio.run(ModelPlanner(client).plan(TEXT, {}))
    assert caught.value.failure['code'] == 'permission_denied'
    assert len(client.calls) == 1


def test_invalid_json_cannot_turn_into_user_missing_fields_or_leak_payload():
    client = ScriptedClient([{'content': 'api_key=private-provider-content'}]*3)
    with pytest.raises(PlanningFailure) as caught:
        asyncio.run(ModelPlanner(client).plan(TEXT, {}))
    assert len(client.calls) == 2
    assert caught.value.failure['code'] == 'model_format_error'
    assert caught.value.failure['missing_fields'] == []
    assert 'private-provider-content' not in json.dumps(caught.value.events)


def test_planner_cannot_use_repair_to_reset_cumulative_model_budget(tmp_path):
    client = ScriptedClient([{'content': 'bad JSON'}, reply(valid_plan())])
    budget = BudgetLedger(tmp_path/'budget.sqlite3', 'workflow', 'u', 'c', BudgetLimits(model=1))
    with pytest.raises(PlanningFailure) as caught:
        asyncio.run(ModelPlanner(client).plan(TEXT, {}, budget=budget))
    assert len(client.calls) == 1
    assert caught.value.failure['code'] == 'budget_exhausted'
    assert budget.summary()['used']['model'] == 1


def test_replanner_sees_existing_state_and_returns_only_a_local_change():
    change = {'base_version': 3, 'reason': '用户修改日期', 'add': [], 'update': [], 'cancel': ['slots']}
    client = ScriptedClient([reply(change)])
    existing = {'version': 3, 'tasks': {'drug': {'status': 'succeeded', 'result': {'evidence': ['saved']}}}}
    decision = asyncio.run(ModelPlanner(client).replan(TEXT, {}, existing=existing,
        event={'kind': 'user_change', 'message': '取消号源查询'}))
    assert decision.proposal.cancel == ['slots']
    payload = json.loads(client.calls[0]['messages'][-1]['content'])
    assert payload['existing_workflow'] == existing
    assert payload['event']['kind'] == 'user_change'


def test_sdk_hidden_retries_are_disabled():
    client = create_llm_client('offline-test-placeholder')
    assert client.max_retries == 0
    asyncio.run(client.close())


def test_real_openai_sdk_wire_contract_and_reported_usage_with_mock_http(tmp_path):
    requests = []
    def handle(request):
        payload = json.loads(request.content)
        assert request.url.path == '/v1/chat/completions'
        assert all('durable_preferences' not in message for message in payload['messages'])
        requests.append(payload)
        return httpx.Response(200, json={'id': 'offline-chat', 'object': 'chat.completion', 'created': 1,
            'model': 'offline', 'choices': [{'index': 0, 'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': json.dumps(valid_plan(), ensure_ascii=False)}}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 3, 'total_tokens': 13}})
    async def run():
        client = AsyncOpenAI(api_key='offline-fixture', base_url='http://mock.invalid/v1', max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
        budget = BudgetLedger(tmp_path/'sdk.sqlite3', 'wire-workflow', 'u', 'c')
        try:
            decision = await ModelPlanner(client).plan(TEXT, {'agent_route': 'appointment'}, budget=budget)
            assert len(decision.proposal.tasks) == 2
            assert json.loads(requests[0]['messages'][-1]['content'])['user_request'] == TEXT
            assert budget.summary()['tokens']['total_tokens'] == 13
            assert budget.summary()['used']['model'] == 1
            await create_chat_completion(client, model='offline', messages=[{'role': 'assistant',
                'content': '用户偏好背景', 'durable_preferences': {'private_server_metadata': True}}])
            assert requests[-1]['messages'] == [{'role': 'assistant', 'content': '用户偏好背景'}]
        finally:
            await client.close()
    asyncio.run(run())


def test_real_sdk_provider_error_does_not_hide_retries_or_leak_server_body(tmp_path):
    calls = []
    def handle(request):
        calls.append(request)
        return httpx.Response(503, json={'error': {'message': 'private-provider-payload', 'type': 'unavailable'}})
    async def run():
        client = AsyncOpenAI(api_key='offline-fixture', base_url='http://mock.invalid/v1', max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
        budget = BudgetLedger(tmp_path/'sdk.sqlite3', 'wire-workflow', 'u', 'c')
        try:
            with pytest.raises(PlanningFailure) as caught:
                await ModelPlanner(client).plan(TEXT, {}, budget=budget)
            assert caught.value.failure['code'] == 'model_unavailable'
            assert len(calls) == budget.summary()['used']['model'] == 1
            assert 'private-provider-payload' not in json.dumps(caught.value.events)
        finally:
            await client.close()
    asyncio.run(run())


def test_shared_model_context_does_not_double_charge_planning(tmp_path):
    client = ScriptedClient([reply(valid_plan())])
    budget = BudgetLedger(tmp_path/'budget.sqlite3', 'workflow', 'u', 'c')
    with use_budget(budget):
        asyncio.run(ModelPlanner(client).plan(TEXT, {}, budget=budget))
    assert budget.summary()['used']['model'] == 1
