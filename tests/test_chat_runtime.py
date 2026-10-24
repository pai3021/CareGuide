import asyncio
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from careguide.agent_orchestrator import AgentOrchestrator
from careguide.api import app
from careguide.domain import IntentCategory
from careguide.intent_recognizer import IntentRecognizer, IntentResult
from careguide.runtime import get_chat_runtime, get_registry
from careguide.task_context import SessionMemory
from careguide.trace_store import validate_trace_completeness
from careguide.tool_manager import CircuitState, MCPToolManager, Tool


class ScriptedClient:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self.replies.pop(0)
        if isinstance(response, Exception):
            raise response
        return {'choices': [{'message': response}]}


class FixedRecognizer:
    def __init__(self, result):
        self.result, self.histories = result, []

    async def recognize(self, message, history):
        self.histories.append(list(history))
        return self.result


def tool_reply(name, arguments):
    return {'content': None, 'tool_calls': [{'id': 'call-1', 'type': 'function',
            'function': {'name': name, 'arguments': json.dumps(arguments)}}]}


def result(*intents, disposition='route'):
    return IntentResult(intents[0] if intents else None, 0.9, list(intents), disposition)


def test_llm_few_shot_is_actually_sent_and_context_is_preserved():
    client = ScriptedClient([{'content': json.dumps({'intent': 'medication_query',
        'intents': ['medication_query'], 'confidence': 0.95, 'disposition': 'route',
        'entities': {}, 'reasoning': '延续上一轮药品问题'})}])
    recognizer = IntentRecognizer(client)
    answer = asyncio.run(recognizer.recognize('这个有什么禁忌', [{'role': 'user', 'content': '精氨酸布洛芬片0.2g'}]))
    assert answer.intent == IntentCategory.MEDICATION_QUERY
    prompt = client.calls[0]['messages'][0]['content']
    assert 'examples' in prompt and '精氨酸布洛芬片0.2g' in prompt
    assert recognizer.examples and all(c.split == 'dev' for c in recognizer.examples)


def test_llm_failure_uses_explicit_rule_fallback_without_clinical_assurance():
    recognizer = IntentRecognizer(ScriptedClient([RuntimeError('provider down')]))
    answer = asyncio.run(recognizer.recognize('挂号需要什么材料'))
    assert answer.intent == IntentCategory.APPOINTMENT
    assert answer.source_scores['llm']['failed']


def test_urgent_preempts_ordinary_llm_route_and_all_tools():
    client = ScriptedClient([{'content': json.dumps({'intent': 'appointment', 'intents': ['appointment'],
                'confidence': 0.99, 'disposition': 'route', 'entities': {}})}])
    engine = AgentOrchestrator(IntentRecognizer(client), get_registry(), client=client)
    answer = asyncio.run(engine.process('现在家人叫不醒，帮忙挂号', 'user', 'conv'))
    assert answer['workflow_status'] == 'urgent_handoff'
    assert answer['tools_used'] == [] and answer['agent_types'] == ['escalation']
    assert answer['primary_agent'] == 'escalation'
    assert len(client.calls) == 1


def test_negated_emergency_word_does_not_trigger_rule_handoff():
    assert not IntentRecognizer._pattern('没有叫不醒，只想查挂号流程').get('urgent')


def test_real_handler_tool_loop_followup_citations_and_complete_trace():
    client = ScriptedClient([tool_reply('lookup_drug_label', {'drug_name': '精氨酸布洛芬片',
        'dosage_form': '片剂', 'specification': '0.2g', 'section': 'precautions'}),
        {'content': '不可信的模型医疗结论：完全安全'}])
    understood = result(IntentCategory.MEDICATION_QUERY)
    understood.entities = {'drug_name': '精氨酸布洛芬片', 'dosage_form': '片剂', 'specification': '0.2g'}
    engine = AgentOrchestrator(FixedRecognizer(understood), get_registry(), client=client)
    answer = asyncio.run(engine.process('查说明书', 'user', 'conv'))
    assert answer['evidence'] and 'https://www.nmpa.gov.cn/' in answer['response']
    assert '完全安全' not in answer['response']
    assert any(m['role'] == 'tool' for m in client.calls[1]['messages'])
    assert validate_trace_completeness(engine.traces.get(answer['request_id']))['complete']


def test_cross_role_tool_is_denied_before_execution():
    client = ScriptedClient([tool_reply('lookup_drug_label', {'drug_name': '精氨酸布洛芬片'})])
    engine = AgentOrchestrator(FixedRecognizer(result(IntentCategory.APPOINTMENT)), get_registry(), client=client)
    answer = asyncio.run(engine.process('挂号问题', 'user', 'conv'))
    trace = engine.traces.get(answer['request_id'])
    assert not trace['execution']['tool_calls'][0]['success']
    assert not answer['evidence']


def test_model_cannot_fill_unconfirmed_product_specification():
    understood = result(IntentCategory.MEDICATION_QUERY)
    understood.entities = {'drug_name': '精氨酸布洛芬片', 'dosage_form': '片剂'}
    client = ScriptedClient([tool_reply('lookup_drug_label', {
        'drug_name': '其他药品', 'dosage_form': '片剂', 'specification': '0.2g'})])
    engine = AgentOrchestrator(FixedRecognizer(understood), get_registry(), client=client)
    answer = asyncio.run(engine.process('规格还没确认', 'user', 'conv'))
    assert answer['workflow_status'] == 'needs_clarification' and not answer['evidence']
    call = engine.traces.get(answer['request_id'])['execution']['tool_calls'][0]
    assert call['input']['drug_name'] == '精氨酸布洛芬片'
    assert 'specification' not in call['input']


def test_exact_hospital_mapping_overrides_model_id_and_invented_campus():
    client = ScriptedClient([{'content': json.dumps({'intent': 'appointment',
        'intents': ['appointment'], 'confidence': 0.95, 'disposition': 'route',
        'entities': {'hospital_id': 'invented-id', 'campus': '本院'}})}])
    recognizer = IntentRecognizer(client)
    answer = asyncio.run(recognizer.recognize('查询北京大学国际医院挂号指南，院区未确认'))
    assert answer.entities['hospital_id'] == recognizer._explicit_entities(
        '北京大学国际医院', [])['hospital_id']
    assert 'campus' not in answer.entities


def test_confirmed_campus_followup_resolves_model_clarification():
    client = ScriptedClient([{'content': json.dumps({'intent': 'appointment',
        'intents': ['appointment'], 'confidence': 0.9,
        'disposition': 'needs_clarification', 'entities': {},
        'requested_action': 'booking_transaction'})}])
    recognizer = IntentRecognizer(client)
    answer = asyncio.run(recognizer.recognize('本院', [
        {'role': 'user', 'content': '北京大学国际医院挂号流程'}]))
    assert answer.disposition == 'route' and answer.intent == IntentCategory.APPOINTMENT
    assert answer.entities['hospital_id'] and answer.entities['campus'] == '本院'
    assert answer.requested_action == 'sourced_information'


def test_provider_failure_after_completed_call_preserves_evidence():
    client = ScriptedClient([tool_reply('search_care_guidance', {'query': '头痛记录'}), RuntimeError('down')])
    engine = AgentOrchestrator(FixedRecognizer(result(IntentCategory.SYMPTOM_QUERY)), get_registry(), client=client)
    answer = asyncio.run(engine.process('就诊准备', 'user', 'conv'))
    assert answer['evidence']
    assert engine.traces.get(answer['request_id'])['execution']['tool_calls']


def test_provider_cannot_repeat_read_tool_with_paraphrased_query():
    client = ScriptedClient([
        tool_reply('search_care_guidance', {'query': '头痛怎么记录', 'top_k': 5}),
        tool_reply('search_care_guidance', {'query': '头痛日记需要写什么', 'top_k': 10}),
    ])
    engine = AgentOrchestrator(
        FixedRecognizer(result(IntentCategory.SYMPTOM_QUERY)), get_registry(), client=client)

    answer = asyncio.run(engine.process('反复头痛，就诊前怎么记录？', 'user', 'conv'))

    calls = engine.traces.get(answer['request_id'])['execution']['tool_calls']
    assert len(calls) == 1
    assert calls[0]['tool_name'] == 'search_care_guidance'
    assert calls[0]['input']['query'] == '反复头痛，就诊前怎么记录？'
    assert calls[0]['input']['top_k'] == 1
    assert len(client.calls) == 2
    assert '工具调用预算已用完' not in answer['response']


def test_multirole_collaboration_keeps_individual_scopes_and_skills():
    engine, _, _ = get_chat_runtime()
    other = AgentOrchestrator(FixedRecognizer(result(IntentCategory.SYMPTOM_QUERY, IntentCategory.APPOINTMENT)),
                             get_registry(), skills=engine.skills)
    answer = asyncio.run(other.process('头痛就诊准备和挂号材料', 'user', 'conv'))
    trace = other.traces.get(answer['request_id'])
    assert answer['agent_types'] == ['triage', 'appointment']
    assert len(trace['execution']['execution_waves'][0]['agents']) == 2
    assert all(a['selected_skill'] for a in trace['execution']['execution_waves'][0]['agents'])


def test_http_multiturn_memory_does_not_cross_users():
    recognizer = FixedRecognizer(result(IntentCategory.APPOINTMENT))
    engine = AgentOrchestrator(recognizer, get_registry())
    _, _, monitor = get_chat_runtime()
    app.dependency_overrides[get_chat_runtime] = lambda: (engine, SessionMemoryShared, monitor)
    try:
        client = TestClient(app)
        for user, text in [('a', '挂号'), ('a', '材料呢'), ('b', '材料呢')]:
            assert client.post('/chat', json={'message': text, 'user_id': user, 'conv_id': 'same'}).status_code == 200
        assert len(recognizer.histories[1]) == 2
        assert recognizer.histories[2] == []
    finally:
        app.dependency_overrides.clear()


SessionMemoryShared = SessionMemory()


def test_tool_manager_timeout_and_circuit_breaker():
    async def slow(params, context):
        await asyncio.sleep(0.05)
    manager = MCPToolManager('not-configured')
    tool = Tool('slow', 'test', slow, {'type': 'object'}, timeout_s=0.001)
    manager.register(tool)
    async def run():
        for _ in range(5):
            assert not (await manager.call('slow', {})).success
    asyncio.run(run())
    assert tool.breaker.state == CircuitState.OPEN


def test_local_vectors_are_labelled_hash_fallback_and_cache_is_not_shared_mutably():
    recognizer = IntentRecognizer(embedding_mode='local_ngram')
    async def run():
        first = await recognizer.recognize('挂号需要哪些材料')
        first.entities['tampered'] = True
        second = await recognizer.recognize('挂号需要哪些材料')
        assert 'tampered' not in second.entities
        assert second.source_scores['embedding_mode'] == 'local_ngram'
    asyncio.run(run())


def test_memory_outage_does_not_discard_completed_tool_query():
    class UnavailableMemory:
        async def history(self, *args):
            raise RuntimeError('Redis down')
        async def add(self, *args):
            raise RuntimeError('Redis down')
    recognizer = FixedRecognizer(result(IntentCategory.SYMPTOM_QUERY))
    engine = AgentOrchestrator(recognizer, get_registry())
    _, _, monitor = get_chat_runtime()
    app.dependency_overrides[get_chat_runtime] = lambda: (engine, UnavailableMemory(), monitor)
    try:
        response = TestClient(app).post('/chat', json={'message': '头痛就诊记录'})
        assert response.status_code == 200
        assert response.json()['evidence']
        assert response.json()['memory_status'] == 'unavailable'
    finally:
        app.dependency_overrides.clear()


def test_exact_medical_entity_extraction_does_not_replace_unknown_new_product():
    recognizer = IntentRecognizer(embedding_mode='disabled')
    entities = recognizer._explicit_entities('精氨酸布洛芬片0.2g的说明书', [])
    assert entities['drug_name'] == '精氨酸布洛芬片' and entities['specification'] == '0.2g'
    history = [{'role': 'user', 'content': '精氨酸布洛芬片0.2g说明书'}]
    assert recognizer._explicit_entities('阿托伐他汀的禁忌', history).get('drug_name') is None


def test_offline_hospital_clarification_followup_keeps_task_and_scope(monkeypatch):
    class VerifiedSourceClock:
        @staticmethod
        def now(zone):
            return datetime(2026, 9, 21, tzinfo=zone)

    # This case tests scope recovery while its frozen source is valid.
    # Expiry rejection is covered separately with an explicit expired clock.
    monkeypatch.setattr('careguide.tools.datetime', VerifiedSourceClock)
    legacy = Path(__file__).resolve().parents[1] / 'data' / 'medical_v1'
    monkeypatch.setenv('CAREGUIDE_CORPUS_DIR', str(legacy))
    get_registry.cache_clear()
    engine = AgentOrchestrator(IntentRecognizer(), get_registry())
    history = [{'role': 'user', 'content': '北京大学国际医院挂号流程'},
               {'role': 'assistant', 'content': '请确认院区'}]
    answer = asyncio.run(engine.process('本院', 'user', 'conv', history))
    assert answer['workflow_status'] == 'completed'
    assert answer['intent'] == 'appointment'
    assert all(item['scope']['hospital_id'] == 'pkuih' for item in answer['evidence'])
    get_registry.cache_clear()
