"""The Zhongda pilot must not leak frozen Beijing hospital material."""
import asyncio
import json
from pathlib import Path

from careguide.corpus import Corpus, DEFAULT_CORPUS
from careguide.agent_orchestrator import AgentOrchestrator
from careguide.domain import IntentCategory
from careguide.intent_recognizer import IntentRecognizer
from careguide.domain import AgentType
from careguide.retrieval.knowledge_base import MedicalKnowledgeBase
from careguide.tools import MedicalToolRegistry


ZHONGDA = Path(__file__).resolve().parents[1] / 'data' / 'medical_zhongda_v1'
ZHONGDA_V2 = Path(__file__).resolve().parents[1] / 'data' / 'medical_zhongda_v2'


def test_zhongda_version_selection_preserves_frozen_default(monkeypatch):
    monkeypatch.delenv('CAREGUIDE_CORPUS_DIR', raising=False)
    original = Corpus()
    assert original.version == 'careguide-medical-corpus-v1'
    assert DEFAULT_CORPUS.name == 'medical_v1'
    monkeypatch.setenv('CAREGUIDE_CORPUS_DIR', str(ZHONGDA))
    pilot = Corpus()
    assert pilot.version == 'careguide-medical-zhongda-v1'
    assert pilot.fingerprint != original.fingerprint
    assert {r.hospital_id for r in pilot.records if r.hospital_id} == {'zhongda'}
    assert all('北京大学' not in r.content for r in pilot.records)


def test_zhongda_hospital_and_campus_scope():
    tools = MedicalToolRegistry(MedicalKnowledgeBase(Corpus(ZHONGDA)))
    query = lambda hospital, campus: tools.query(AgentType.APPOINTMENT, 'lookup_hospital_guide',
        {'hospital_id': hospital, 'campus': campus, 'section': 'materials'})
    assert query('zhongda', '本部')['status'] == 'found'
    assert query('zhongda', '江北院区')['status'] == 'not_covered'
    assert query('pkuih', '本部')['status'] == 'not_covered'
    assert tools.query(AgentType.APPOINTMENT, 'lookup_hospital_guide',
        {'hospital_id': 'zhongda', 'campus': '本部', 'section': 'cancellation'})['status'] == 'not_covered'


def test_zhongda_department_scope_does_not_claim_other_services():
    tools = MedicalToolRegistry(MedicalKnowledgeBase(Corpus(ZHONGDA)))
    def lookup(department):
        return tools.query(AgentType.TRIAGE, 'lookup_department',
            {'hospital_id': 'zhongda', 'campus': '本部', 'department': department})
    assert lookup('呼吸内科')['status'] == 'found'
    assert lookup('神经内科')['status'] == 'not_covered'


def test_zhongda_short_name_is_resolved_only_in_selected_corpus(monkeypatch):
    monkeypatch.setenv('CAREGUIDE_CORPUS_DIR', str(ZHONGDA))
    entities = IntentRecognizer(embedding_mode='disabled')._explicit_entities('中大医院本部挂号材料', [])
    assert entities['hospital_id'] == 'zhongda'
    assert entities['campus'] == '本部'


def test_zhongda_v2_adds_only_verified_main_campus_scope():
    first, second = Corpus(ZHONGDA), Corpus(ZHONGDA_V2)
    assert first.version == 'careguide-medical-zhongda-v1'
    assert second.version == 'careguide-medical-zhongda-v2'
    assert {r.id for r in first.records} < {r.id for r in second.records}
    assert len(second.records) == len(first.records) + 3
    registry = MedicalToolRegistry(MedicalKnowledgeBase(second))
    for department in ('消化内科', '骨科'):
        reply = registry.query(AgentType.TRIAGE, 'lookup_department',
            {'hospital_id': 'zhongda', 'campus': '本部', 'department': department})
        assert reply['status'] == 'found'
        assert all(e['scope']['campus'] == '本部' for e in reply['evidence'])
    assert registry.query(AgentType.TRIAGE, 'lookup_department',
        {'hospital_id': 'zhongda', 'campus': '江北院区', 'department': '骨科'})['status'] == 'not_covered'


def test_named_department_information_corrects_appointment_misroute(monkeypatch):
    from tests.test_chat_runtime import ScriptedClient

    monkeypatch.setenv('CAREGUIDE_CORPUS_DIR', str(ZHONGDA_V2))
    mistaken = {'intent': 'appointment', 'intents': ['appointment'], 'confidence': 0.9,
                'disposition': 'route', 'entities': {}, 'reasoning': 'hospital name'}
    recognizer = IntentRecognizer(ScriptedClient([{'content': json.dumps(mistaken)}]))
    result = asyncio.run(recognizer.recognize('中大医院本部有消化内科的资料吗？'))
    assert result.intent == IntentCategory.SYMPTOM_QUERY
    assert result.fine_intent == 'department_query'
    assert result.entities['department'] == '消化内科'
    assert result.source_scores['route_override'] == 'explicit_department_information'
    booking = IntentRecognizer(ScriptedClient([{'content': json.dumps(mistaken)}]))
    actual_booking = asyncio.run(booking.recognize('中大医院本部消化内科怎么预约挂号？'))
    assert actual_booking.intent == IntentCategory.APPOINTMENT


def test_named_department_chat_uses_one_exact_lookup(monkeypatch):
    from tests.test_chat_runtime import ScriptedClient

    monkeypatch.setenv('CAREGUIDE_CORPUS_DIR', str(ZHONGDA_V2))
    mistaken = {'intent': 'appointment', 'intents': ['appointment'], 'confidence': 0.9,
                'disposition': 'route', 'entities': {}, 'reasoning': 'hospital name'}
    client = ScriptedClient([{'content': json.dumps(mistaken)}])
    knowledge = MedicalKnowledgeBase(Corpus(ZHONGDA_V2))
    engine = AgentOrchestrator(IntentRecognizer(client), MedicalToolRegistry(knowledge), client=client)
    answer = asyncio.run(engine.process('中大医院本部有消化内科的资料吗？', 'pilot-test', 'pilot-test'))
    assert answer['workflow_status'] == 'completed'
    assert answer['tools_used'] == ['lookup_department']
    assert {item['record_id'] for item in answer['evidence']} == {'zhongda-gastroenterology-main-campus'}
    assert len(client.calls) == 1  # Intent only; exact source lookup does not need a model tool loop.
