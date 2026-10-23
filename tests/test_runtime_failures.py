"""Fault injection through the real role loop, dispatcher and reply path."""
import asyncio
import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from careguide.agent_orchestrator import AgentOrchestrator
from careguide.corpus import Corpus
from careguide.domain import AgentType, IntentCategory
from careguide.errors import ErrorCode, aggregate_status
from careguide.llm_utils import extract_tool_calls, build_openai_tool_followup
from careguide.intent_recognizer import IntentRecognizer
from careguide.retrieval.knowledge_base import MedicalKnowledgeBase
from careguide.task_context import TaskContext
from careguide.tools import MedicalToolRegistry
from tests.test_chat_runtime import FixedRecognizer, ScriptedClient, result, tool_reply


def response(arguments):
    return {'choices': [{'message': {'tool_calls': [{
        'id': 'bad-call', 'function': {'name': 'lookup_drug_label', 'arguments': arguments}}]}}]}


def medication_engine(client=None, **kwargs):
    understood = result(IntentCategory.MEDICATION_QUERY)
    understood.entities = {'drug_name': '精氨酸布洛芬片', 'dosage_form': '片剂',
                           'specification': '0.2g'}
    registry = MedicalToolRegistry(MedicalKnowledgeBase(Corpus()))
    return AgentOrchestrator(FixedRecognizer(understood), registry, client=client, **kwargs)


@pytest.mark.parametrize('arguments', ['{broken', '[]', 'null', '"text"', None])
def test_malformed_arguments_keep_an_explicit_parse_failure(arguments):
    call = extract_tool_calls(response(arguments))[0]
    assert call['parse_error'] in {'invalid_json', 'expected_object'}
    assert call['id'] == 'bad-call'


def test_model_format_error_is_repaired_without_asking_user_for_known_fields():
    bad = response('{broken')['choices'][0]['message']
    client = ScriptedClient([bad, tool_reply('lookup_drug_label', {'section': 'precautions'}),
                             {'content': 'done'}])
    engine = medication_engine(client)
    answer = asyncio.run(engine.process('精氨酸布洛芬片0.2g的注意事项', 'u', 'c'))
    calls = engine.traces.get(answer['request_id'])['execution']['tool_calls']
    assert calls[0]['failure']['code'] == 'model_format_error'
    assert calls[0]['failure']['missing_fields'] == []
    assert calls[0]['resolved'] is True
    assert calls[1]['success'] and calls[1]['output']['evidence']
    assert answer['workflow_status'] == 'completed'
    assert answer['errors'] == []
    assert len(client.calls) == 3
    feedback = [m for m in client.calls[1]['messages'] if m['role'] == 'tool']
    assert json.loads(feedback[-1]['content'])['failure']['code'] == 'model_format_error'
    assert '请补充' not in answer['response']


def test_format_repairs_are_bounded_and_do_not_execute_a_tool(monkeypatch):
    bad = response('{broken')['choices'][0]['message']
    client = ScriptedClient([bad, bad, tool_reply('lookup_drug_label', {})])
    engine = medication_engine(client)
    invoke = AsyncMock()
    monkeypatch.setattr(engine.tool_manager, 'call', invoke)
    answer = asyncio.run(engine.process('精氨酸布洛芬片0.2g说明书', 'u', 'c'))
    assert len(client.calls) == 2
    invoke.assert_not_called()
    assert answer['workflow_status'] == 'failed'
    assert answer['evidence'] == []
    assert all(error['code'] == 'model_format_error' for error in answer['errors'])
    assert '猜测' in answer['response']


def test_tool_followup_keeps_original_invalid_json_for_provider_protocol():
    original = response('{broken')
    messages = build_openai_tool_followup(original, [('bad-call', {'status': 'model_format_error'})])
    assert messages[0]['tool_calls'][0]['function']['arguments'] == '{broken'
    assert messages[1]['tool_call_id'] == 'bad-call'


def test_actual_missing_product_scope_stops_without_model_repair():
    client = ScriptedClient([tool_reply('lookup_drug_label', {'section': 'all'})])
    engine = medication_engine(client)
    engine.recognizer.result.entities.pop('specification')
    answer = asyncio.run(engine.process('精氨酸布洛芬片说明书', 'u', 'c'))
    assert answer['workflow_status'] == 'needs_clarification'
    assert answer['errors'][0]['code'] == 'missing_user_input'
    assert 'specification' in answer['errors'][0]['missing_fields']
    assert len(client.calls) == 1
    assert not answer['evidence']


def test_permission_denial_is_not_user_missing_input():
    engine = medication_engine()
    call = {'id': 'denied', 'name': 'lookup_drug_label', 'input': {}}
    outcome = asyncio.run(engine._dispatch(AgentType.APPOINTMENT, call))
    assert outcome['output']['status'] == 'permission_denied'
    assert outcome['failure']['code'] == 'permission_denied'
    assert not outcome['failure']['retryable']
    assert outcome['failure']['missing_fields'] == []


def test_external_service_failure_never_becomes_missing_input_or_leaks_details():
    backend = SimpleNamespace(name='weknora_mcp', search=AsyncMock(
        side_effect=RuntimeError('private api_key=do-not-expose')))
    engine = medication_engine(knowledge_backend=backend)
    task = TaskContext.from_intent_result(engine.recognizer.result)
    call = {'id': 'provider-call', 'name': 'lookup_drug_label', 'input': {}}
    outcome = asyncio.run(engine._dispatch(AgentType.MEDICATION, call, task))
    assert outcome['failure']['code'] == 'tool_unavailable'
    assert outcome['failure']['stage'] == 'tool_execution'
    assert outcome['failure']['retryable']
    assert outcome['output']['status'] != 'needs_clarification'
    assert 'do-not-expose' not in json.dumps(outcome)


def test_timeout_classification_reaches_dispatcher():
    engine = medication_engine()
    tool = engine.tool_manager._tools['medication:lookup_drug_label']
    async def slow(params, context):
        await asyncio.sleep(0.1)
    tool.handler, tool.timeout_s = slow, 0.001
    task = TaskContext.from_intent_result(engine.recognizer.result)
    outcome = asyncio.run(engine._dispatch(AgentType.MEDICATION,
        {'id': 'timeout', 'name': 'lookup_drug_label', 'input': {}}, task))
    assert outcome['failure']['code'] == 'tool_timeout'
    assert outcome['failure']['retryable']


def test_invalid_parameter_type_is_distinct_from_absent_user_input():
    engine = medication_engine()
    task = TaskContext.from_intent_result(engine.recognizer.result)
    outcome = asyncio.run(engine._dispatch(AgentType.MEDICATION,
        {'id': 'invalid', 'name': 'lookup_drug_label', 'input': {'section': 23}}, task))
    assert outcome['failure']['code'] == 'invalid_parameters'
    assert outcome['failure']['missing_fields'] == []


def test_composite_partial_result_keeps_successful_source_and_reports_failed_branch():
    engine = medication_engine()
    understood = result(IntentCategory.SYMPTOM_QUERY, IntentCategory.MEDICATION_QUERY)
    understood.entities = {'drug_name': '未收录药品', 'dosage_form': '片剂', 'specification': '0.2g'}
    engine.recognizer = FixedRecognizer(understood)
    answer = asyncio.run(engine.process('头痛就诊记录和未收录药品说明书', 'u', 'c'))
    assert answer['workflow_status'] == 'partial_completed'
    assert answer['evidence']
    assert answer['errors'][0]['code'] == 'not_covered'
    assert answer['medical_answering_enabled'] is False


def test_unavailable_source_and_empty_search_have_distinct_aggregate_states():
    def call(status):
        return {'success': True, 'output': {'status': status}}
    assert aggregate_status([call('source_unavailable')]) == 'failed'
    assert aggregate_status([call('not_covered')]) == 'not_covered'


def test_expired_source_during_followup_is_not_published_or_reported_complete(monkeypatch):
    class ExpiredClock:
        @staticmethod
        def now(zone):
            return datetime(2026, 10, 20, tzinfo=zone)

    monkeypatch.setattr('careguide.tools.datetime', ExpiredClock)
    registry = MedicalToolRegistry(MedicalKnowledgeBase(Corpus()))
    engine = AgentOrchestrator(IntentRecognizer(), registry)
    history = [{'role': 'user', 'content': '北京大学国际医院挂号流程'},
               {'role': 'assistant', 'content': '请确认院区'}]
    answer = asyncio.run(engine.process('本院', 'u', 'c', history))
    assert answer['intent'] == 'appointment'
    assert answer['workflow_status'] == 'failed'
    assert answer['errors'][0]['code'] == 'source_unavailable'
    assert answer['evidence'] == []


@pytest.mark.parametrize('code', list(ErrorCode))
def test_error_contract_has_safe_identity_and_actionable_next_step(code):
    from careguide.errors import Failure
    payload = Failure(code, 'validation', 'task-1', 'call-1').payload()
    assert payload['task_id'] == 'task-1'
    assert payload['call_id'] == 'call-1'
    assert payload['stage'] == 'validation'
    assert payload['message'] and payload['next_step']
    assert isinstance(payload['missing_fields'], list)
