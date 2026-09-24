import asyncio
import json

from careguide.domain import IntentCategory
from careguide.intent_v2_dataset import (BOUNDARY_LABELS, FINE_LABELS,
                                         load_v2_development_cases, load_v2_evaluation_cases)
from careguide.intent_v2_recognizer import V2IntentRecognizer
from tests.test_chat_runtime import ScriptedClient


def reply(fine=None, *, disposition='route', fine_intents=None):
    labels = fine_intents if fine_intents is not None else ([fine] if fine else [])
    return {'content': json.dumps({'primary_fine_intent': fine, 'fine_intents': labels,
        'disposition': disposition, 'confidence': 0.95, 'entities': {},
        'requested_action': 'sourced_information'}, ensure_ascii=False)}


def test_v2_splits_are_balanced_separate_and_not_loaded_into_recognizer():
    dev, heldout = load_v2_development_cases(), load_v2_evaluation_cases()
    assert len(dev) == 360 and len(heldout) == 180
    assert len(FINE_LABELS) == 15 and len(BOUNDARY_LABELS) == 3
    assert not {c.scenario_family for c in dev} & {c.scenario_family for c in heldout}
    recognizer = V2IntentRecognizer(embedding_mode='disabled')
    assert len(recognizer.examples) == 360
    assert all(c.split == 'dev' for c in recognizer.examples)
    assert all(label in FINE_LABELS for label in recognizer.templates)


def test_v2_llm_predicts_fine_intent_and_maps_to_existing_agent():
    client = ScriptedClient([reply('drug_contraindications')])
    recognizer = V2IntentRecognizer(client)
    result = asyncio.run(recognizer.recognize('精氨酸布洛芬片的禁忌信息'))
    assert result.fine_intent == 'drug_contraindications'
    assert result.fine_intents == ['drug_contraindications']
    assert result.intent == IntentCategory.MEDICATION_QUERY
    assert result.source_scores['taxonomy'] == 'medical_intent_v2'
    prompt = client.calls[0]['messages'][0]['content']
    assert 'drug_contraindications' in prompt and 'primary_fine_intent' in prompt


def test_v2_multitask_and_urgent_preemption():
    client = ScriptedClient([reply('appointment_materials', fine_intents=[
        'appointment_materials', 'drug_label'])])
    recognizer = V2IntentRecognizer(client)
    result = asyncio.run(recognizer.recognize('中大医院挂号材料和药品说明书都想查'))
    assert result.fine_intents == ['appointment_materials', 'drug_label']
    assert result.intents == [IntentCategory.APPOINTMENT, IntentCategory.MEDICATION_QUERY]
    emergency = V2IntentRecognizer(ScriptedClient([reply('appointment_process')]))
    result = asyncio.run(emergency.recognize('现在家人叫不醒，帮忙查挂号流程'))
    assert result.disposition == 'urgent_handoff' and result.intent == IntentCategory.EMERGENCY
    assert not result.fine_intents


def test_v2_local_handoff_is_not_a_fourth_domain_agent():
    recognizer = V2IntentRecognizer(ScriptedClient([reply('handoff_request')]))
    result = asyncio.run(recognizer.recognize('请转人工协助'))
    assert result.fine_intent == 'handoff_request'
    assert result.intent is None and result.intents == []


def test_v2_boundary_and_provider_failure():
    recognizer = V2IntentRecognizer(ScriptedClient([reply(disposition='needs_clarification')]))
    result = asyncio.run(recognizer.recognize('那个呢'))
    assert result.disposition == 'needs_clarification' and result.fine_intent == ''
    failed = V2IntentRecognizer(ScriptedClient([RuntimeError('down')]))
    result = asyncio.run(failed.recognize('挂号需要哪些材料'))
    assert result.source_scores['llm']['failed']
    assert result.fine_intent == 'appointment_materials'
