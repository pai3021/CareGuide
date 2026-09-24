import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from careguide.agent_orchestrator import AgentOrchestrator
from careguide.business_store import BusinessStore
from careguide.corpus import Corpus
from careguide.domain import AgentType, IntentCategory
from careguide.intent_recognizer import IntentRecognizer, IntentResult
from careguide.knowledge_workflow import KnowledgeBundle, KnowledgeWorkflow
from careguide.runtime import get_registry, get_chat_runtime
from careguide.api import app


def test_atomic_handoff_reuse_owner_status_and_restart(tmp_path):
    store = BusinessStore(tmp_path/'records.sqlite3')
    def create(_):
        return store.create_handoff('alice', 'conversation', 'request', 'synthetic handoff', [], [], 'LOW')
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(create, range(8)))
    assert len({record['id'] for record, reused in records}) == 1
    assert sum(not reused for record, reused in records) == 1
    identity = records[0][0]['id']
    assert store.get_handoff(identity, 'bob') is None
    with pytest.raises(ValueError):
        store.transition_handoff(identity, 'alice', 'resolved')
    store.transition_handoff(identity, 'alice', 'in_progress')
    store.transition_handoff(identity, 'alice', 'resolved', 'synthetic completion')
    assert BusinessStore(store.path).get_handoff(identity, 'alice')['status'] == 'resolved'


def test_second_wave_contains_first_wave_evidence_and_trace(tmp_path):
    engine = AgentOrchestrator(IntentRecognizer(), get_registry(),
        business_store=BusinessStore(tmp_path/'records.sqlite3'))
    reply = asyncio.run(engine.process('精氨酸布洛芬片0.2g说明书，请转人工协助', 'alice', 'conversation'))
    trace = engine.traces.get(reply['request_id'])
    assert reply['workflow_status'] == 'escalated' and reply['evidence']
    assert [wave['wave'] for wave in trace['execution']['execution_waves']] == [1, 2]
    assert reply['handoff']['evidence'] == reply['evidence']
    assert trace['side_effects'][0]['id'] == reply['handoff']['id']
    assert trace['task_understanding']['shared_task_context']['state_history'][-1] == 'escalated'
    repeated = asyncio.run(engine.process('转人工协助', 'alice', 'conversation'))
    assert repeated['handoff']['id'] == reply['handoff']['id']
    assert engine.traces.get(repeated['request_id'])['side_effects'] == []
    looked_up = asyncio.run(engine.process('查询咨询记录 '+reply['handoff']['id'], 'alice', 'other'))
    assert looked_up['handoff']['id'] == reply['handoff']['id']


def test_emergency_never_creates_local_queue_or_executes_normal_tools(tmp_path):
    store = BusinessStore(tmp_path/'records.sqlite3')
    engine = AgentOrchestrator(IntentRecognizer(), get_registry(), business_store=store)
    reply = asyncio.run(engine.process('现在家人叫不醒，请转人工协助', 'alice', 'conversation'))
    assert reply['urgency'] == 'CRITICAL' and reply['workflow_status'] == 'urgent_handoff'
    assert reply['primary_agent'] == 'escalation'
    assert not reply['tools_used'] and not store.list_handoffs('alice')


def test_handoff_outage_preserves_completed_source_query():
    class UnavailableStore:
        def create_handoff(self, *args):
            raise OSError('synthetic storage outage')
    engine = AgentOrchestrator(IntentRecognizer(), get_registry(), business_store=UnavailableStore())
    reply = asyncio.run(engine.process('精氨酸布洛芬片0.2g说明书，请转人工协助', 'alice', 'conversation'))
    assert reply['evidence'] and reply['workflow_status'] == 'completed' and reply['handoff'] is None
    trace = engine.traces.get(reply['request_id'])
    assert not trace['side_effects'] and not trace['execution']['tool_calls'][-1]['success']


def test_embedding_emergency_similarity_alone_does_not_trigger_urgent_path(monkeypatch):
    recognizer = IntentRecognizer()
    async def similar(message):
        return {'intent': IntentCategory.EMERGENCY, 'confidence': 0.9}
    monkeypatch.setattr(recognizer, '_embedding', similar)
    reply = asyncio.run(recognizer.recognize('请转人工协助'))
    assert reply.disposition == 'needs_clarification' and reply.urgency != 'CRITICAL'


def test_interaction_pair_keeps_explicit_second_substance_and_unknown_pair_is_not_covered():
    engine = AgentOrchestrator(IntentRecognizer(), get_registry())
    known = asyncio.run(engine.process('精氨酸布洛芬片0.2g和肝素相互作用', 'alice', 'known'))
    known_trace = engine.traces.get(known['request_id'])
    assert known_trace['execution']['tool_calls'][0]['tool_name'] == 'lookup_drug_interaction'
    assert known_trace['execution']['tool_calls'][0]['input']['other_drug'] == '肝素'
    assert any(item['record_id'] == 'ibuprofen-arginine-0-2g-heparin' for item in known['evidence'])
    unknown = asyncio.run(engine.process('精氨酸布洛芬片0.2g和维生素C相互作用', 'alice', 'unknown'))
    unknown_trace = engine.traces.get(unknown['request_id'])
    assert unknown_trace['execution']['tool_calls'][0]['tool_name'] == 'lookup_drug_interaction'
    assert unknown_trace['execution']['tool_calls'][0]['input']['other_drug'] == '维生素C'
    assert unknown['workflow_status'] == 'not_covered' and not unknown['evidence']


def test_knowledge_review_publication_and_frozen_seed(tmp_path):
    corpus = Corpus()
    record = corpus.records[0]
    source = corpus._sources[record.source_id]
    # Copy an existing verified fact into a new test-only ID, never invent facts.
    bundle = KnowledgeBundle(sources=[source], records=[record.model_copy(update={'id': 'test-only-copy'})])
    workflow = KnowledgeWorkflow(BusinessStore(tmp_path/'records.sqlite3'))
    item = workflow.submit(bundle, 'developer')
    assert len(workflow.extend(Corpus()).records) == len(corpus.records)
    with pytest.raises(ValueError):
        workflow.publish(item['id'], corpus)
    workflow.review(item['id'], 'source-reviewer', True, 'test: exact existing verified source')
    workflow.publish(item['id'], corpus)
    updated = workflow.extend(Corpus())
    assert len(updated.records) == len(corpus.records)+1 and updated.fingerprint != corpus.fingerprint
    assert len(Corpus().records) == len(corpus.records)
    duplicate = workflow.submit(bundle, 'developer')
    workflow.review(duplicate['id'], 'source-reviewer', True, 'test duplicate')
    with pytest.raises(ValueError):
        workflow.publish(duplicate['id'], updated)


def test_candidate_penalty_changes_selection(tmp_path):
    engine = AgentOrchestrator(IntentRecognizer(), get_registry())
    async def alternate(agent, message, history, task):
        return await engine._run_agent(agent, message, history, task)
    engine.register_candidate(AgentType.TRIAGE, 'alternate', alternate)
    engine.update_routing_penalties({'triage:default': 0.5, 'triage:alternate': 0})
    reply = asyncio.run(engine.process('头痛就诊前记录哪些信息', 'alice', 'conversation'))
    agent = engine.traces.get(reply['request_id'])['execution']['execution_waves'][0]['agents'][0]
    assert agent['candidate'] == 'alternate'


def test_primary_intent_order_controls_owner():
    engine = AgentOrchestrator(IntentRecognizer(), get_registry())
    result = IntentResult(IntentCategory.APPOINTMENT, 0.9,
        [IntentCategory.MEDICATION_QUERY, IntentCategory.APPOINTMENT], 'route')
    assert engine.route_decision(result) == [AgentType.APPOINTMENT, AgentType.MEDICATION]


def test_repeated_model_query_stops_without_budget_failure():
    from tests.test_chat_runtime import ScriptedClient, FixedRecognizer, tool_reply
    understood = IntentResult(IntentCategory.MEDICATION_QUERY, 0.9,
        [IntentCategory.MEDICATION_QUERY], 'route',
        {'drug_name': '精氨酸布洛芬片', 'dosage_form': '片剂', 'specification': '0.2g'})
    arguments = {'drug_name': '精氨酸布洛芬片', 'section': 'all'}
    client = ScriptedClient([tool_reply('lookup_drug_label', arguments)]*2)
    engine = AgentOrchestrator(FixedRecognizer(understood), get_registry(), client=client)
    reply = asyncio.run(engine.process('查询说明书', 'alice', 'conversation'))
    agent = engine.traces.get(reply['request_id'])['execution']['execution_waves'][0]['agents'][0]
    assert reply['evidence'] and not agent['error'] and len(agent['tool_calls']) == 1


def test_management_apis_and_source_upload_contract(tmp_path, monkeypatch):
    monkeypatch.setenv('CAREGUIDE_BUSINESS_DB', str(tmp_path/'records.sqlite3'))
    get_registry.cache_clear(); get_chat_runtime.cache_clear()
    with TestClient(app) as client:
        reply = client.post('/chat', json={'message': '请转人工协助', 'user_id': 'alice', 'conv_id': 'test'}).json()
        identity = reply['handoff']['id']
        assert client.get('/handoffs', params={'user_id': 'alice'}).json()['records'][0]['id'] == identity
        assert client.get('/handoffs/'+identity, params={'user_id': 'bob'}).status_code == 404
        assert client.post('/handoffs/'+identity+'/status', json={'user_id': 'alice', 'status': 'in_progress'}).status_code == 200
        assert client.post('/knowledge/upload', json={'documents': ['unverified claim']}).status_code == 422
        assert client.post('/eval/run', json={'limit': 1, 'split': 'heldout'}).status_code == 422
        assert client.get('/metrics').status_code == 200
    get_registry.cache_clear(); get_chat_runtime.cache_clear()
