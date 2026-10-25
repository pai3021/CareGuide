"""Real embedding SDK with controlled vectors: protocol/authority tests, not semantic quality."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import threading
import time
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient
import httpx
from openai import OpenAI
import pytest

from careguide import runtime as factory
from careguide.api import app
from careguide.corpus import Corpus, DEFAULT_CORPUS
from careguide.orchestration.budget import BudgetLedger, BudgetLimits
from careguide.orchestration.semantic_intent import (
    SemanticIntentRetriever, INSTRUCTION, load_bank, record_response)
from careguide.orchestration.understanding import LocalUnderstanding, compose_plan
from careguide.retrieval.embedding_provider import OpenAICompatibleEmbeddingFunction
from tests.test_chat_runtime import ScriptedClient
from tests.test_model_planner import reply


def provider(*, vector=None, model='test-embedding', handler=None, timeouts=None):
    calls = []
    def respond(request):
        payload = json.loads(request.content)
        if timeouts is not None:
            timeouts.append(request.extensions['timeout'])
        calls.append(payload)
        if handler:
            return handler(payload)
        vectors = [(vector(text) if vector else [1., 0., 0.]) for text in payload['input']]
        return httpx.Response(200, json={'object': 'list', 'model': model,
            'data': [{'index': i, 'embedding': value, 'object': 'embedding'} for i, value in enumerate(vectors)],
            'usage': {'prompt_tokens': 7, 'total_tokens': 7}})
    sdk = OpenAI(api_key='offline-test', max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    return OpenAICompatibleEmbeddingFunction(api_key='offline-test', base_url='https://offline.test/v1',
        model=model, dimensions=3, query_instruction=INSTRUCTION, max_retries=0,
        client=sdk, response_observer=record_response), calls


def ledger(tmp_path, name='w', **limits):
    return BudgetLedger(tmp_path/'budget.sqlite3', name, 'u', 'c', BudgetLimits(**limits))


def test_real_sdk_business_bank_usage_cache_scope_and_no_authority(tmp_path):
    bank = load_bank()
    primary = bank['examples'][0]
    embedding, calls = provider(vector=lambda text: [1., 0., 0.] if text == primary['text'] or text.startswith('Instruct:') else [0., 1., 0.])
    retriever = SemanticIntentRetriever(embedding)
    async def run():
        budget = ledger(tmp_path)
        rows, meta = await retriever.retrieve('想找看胃病的大夫', budget=budget, scope=('u', 'c'))
        assert rows[0]['example_id'] == primary['id']
        assert all(row['reference_only'] and not row['calibrated'] for row in rows)
        assert meta['status'] == 'retrieved' and meta['remote_calls'] == 2
        assert budget.summary()['used']['model'] == 2
        assert budget.summary()['tokens']['prompt_tokens'] == 14
        assert len(calls[0]['input']) == len(bank['examples'])
        assert calls[1]['input'] == [f'Instruct: {INSTRUCTION}\nQuery: 想找看胃病的大夫']
        assert all(call['model'] == 'test-embedding' and call['dimensions'] == 3 for call in calls)
        rows[0]['actions'].append('forged_action')
        cached, meta = await retriever.retrieve('想找看胃病的大夫', budget=budget, scope=('u', 'c'))
        assert meta['status'] == 'cache_hit' and meta['remote_calls'] == 0
        assert 'forged_action' not in cached[0]['actions']
        other = ledger(tmp_path, 'other')
        await retriever.retrieve('想找看胃病的大夫', budget=other, scope=('other', 'c'))
        assert other.summary()['used']['model'] == 1 and len(calls) == 3
        assert all('想找' not in key for key in retriever._cache)
        await retriever.close()
    asyncio.run(run())


def test_prebuilt_index_validates_bank_model_instruction_and_avoids_warmup_call(tmp_path):
    embedding, calls = provider()
    async def run():
        original = SemanticIntentRetriever(embedding)
        artifact = await original.export_index(budget=ledger(tmp_path, 'build'))
        index = tmp_path/'index.json'
        index.write_text(json.dumps(artifact), encoding='utf-8')
        attached = SemanticIntentRetriever(embedding, index_path=index)
        _, meta = await attached.retrieve('有没有别的可约时间', budget=ledger(tmp_path, 'query'), scope=('u', 'c'))
        assert meta['remote_calls'] == 1 and len(calls) == 2
        changed = deepcopy(load_bank())
        changed['examples'][0]['text'] += '新版本'
        mismatch = SemanticIntentRetriever(embedding, bank=changed, index_path=index)
        rows, meta = await mismatch.retrieve('未知请求', budget=ledger(tmp_path, 'mismatch'), scope=('u', 'c'))
        assert rows == [] and meta['status'] == 'unavailable' and meta['remote_calls'] == 0
        assert len(calls) == 2
        await original.close()
    asyncio.run(run())


@pytest.mark.parametrize('bad', [[0., 0., 0.], [1., 2.], [float('nan'), 0., 1.]])
def test_invalid_vectors_fail_closed_and_enter_cooldown(tmp_path, bad):
    embedding, calls = provider(vector=lambda text: bad)
    retriever = SemanticIntentRetriever(embedding)
    async def run():
        budget = ledger(tmp_path)
        rows, meta = await retriever.retrieve('未知', budget=budget, scope=('u', 'c'))
        assert rows == [] and meta['status'] == 'unavailable'
        assert budget.summary()['used']['model'] == 1
        assert (await retriever.retrieve('另一个未知', budget=budget, scope=('u', 'c')))[1]['status'] == 'cooldown'
        assert len(calls) == 1
        await retriever.close()
    asyncio.run(run())


def test_exhausted_or_missing_budget_never_dispatches_embedding(tmp_path):
    embedding, calls = provider()
    retriever = SemanticIntentRetriever(embedding)
    async def run():
        for budget in [None, ledger(tmp_path, model=0)]:
            rows, meta = await retriever.retrieve('未知', budget=budget, scope=('u', 'c'))
            assert rows == [] and meta['status'] == 'budget_exhausted'
        assert calls == []
        await retriever.close()
    asyncio.run(run())


def test_timeout_charges_one_attempt_without_retry_and_bounds_provider_failure(tmp_path):
    entered, release = threading.Event(), threading.Event()
    def block(payload):
        entered.set()
        release.wait(2)
        return httpx.Response(503, json={'error': {'message': 'private provider diagnostic'}})
    embedding, calls = provider(handler=block)
    retriever = SemanticIntentRetriever(embedding, timeout=.05)
    async def run():
        budget = ledger(tmp_path)
        rows, meta = await retriever.retrieve('未知', budget=budget, scope=('u', 'c'))
        release.set()
        assert entered.is_set() and rows == [] and meta['status'] == 'unavailable'
        assert meta['diagnostic_type'] == 'TimeoutError' and len(calls) == 1
        assert 'private' not in json.dumps(meta)
        assert budget.summary()['used']['model'] == 1
        assert (await retriever.retrieve('再试', budget=budget, scope=('u', 'c')))[1]['status'] == 'cooldown'
        await retriever.close()
    asyncio.run(run())


def test_high_similarity_is_only_a_reference_and_cannot_authorize_negated_booking(tmp_path):
    target = next(e['text'] for e in load_bank()['examples'] if e['id'] == 'book-1')
    embedding, _ = provider(vector=lambda text: [1., 0., 0.] if text == target or text.startswith('Instruct:') else [0., 1., 0.])
    understanding = LocalUnderstanding(Corpus(DEFAULT_CORPUS), semantic=SemanticIntentRetriever(embedding))
    async def run():
        text = '我只看看还有哪些号，暂时不要预约'
        result = await understanding.understand(text, message_id='m', at='2026-10-25T10:00:00+00:00', classify=False)
        before = deepcopy(result.goals)
        await understanding.enrich(text, result, budget=ledger(tmp_path), scope=('u', 'c'))
        assert result.candidates[0]['example_id'] == 'book-1'
        assert result.candidates[0]['similarity'] == 1
        assert result.goals == before and result.relation == 'unknown'
        assert compose_plan(text, result)[0] is None
        assert result.semantic_retrieval['direct_acceptance'] is False
        await understanding.close()
    asyncio.run(run())


def test_production_factory_passes_real_embedding_candidates_to_one_planner(monkeypatch, tmp_path):
    for key in tuple(__import__('os').environ):
        if key.startswith('CAREGUIDE_'):
            monkeypatch.delenv(key)
    for key, value in {'CAREGUIDE_BUSINESS_DB': str(tmp_path/'factory.sqlite3'),
        'CAREGUIDE_CORPUS_DIR': str(DEFAULT_CORPUS), 'CAREGUIDE_HOSPITAL_BACKEND': 'memory',
        'CAREGUIDE_LLM_API_KEY': 'offline-test', 'CAREGUIDE_EMBEDDING_API_KEY': 'offline-test',
        'CAREGUIDE_INTENT_SEMANTIC_MODE': 'remote'}.items():
        monkeypatch.setenv(key, value)
    client = ScriptedClient([reply({'kind': 'clarify', 'question': '您想找哪个科室的医生？'})])
    client.close = AsyncMock()
    embedding, calls = provider()
    options = []
    def configured(**kwargs):
        options.append(kwargs)
        return embedding
    monkeypatch.setattr('careguide.llm_utils.create_llm_client', lambda *args: client)
    monkeypatch.setattr('careguide.retrieval.embedding_provider.OpenAICompatibleEmbeddingFunction', configured)
    monkeypatch.setattr('careguide.orchestration.understanding.LocalClassifier.predict',
        lambda *args: {'label': '其他', 'scores': {'其他': 1}, 'margin': 1})
    factory.get_chat_runtime.cache_clear()
    factory.get_registry.cache_clear()
    try:
        with TestClient(app) as http:
            result = http.post('/chat', json={'message': '想找看胃病的大夫', 'user_id': 'u', 'conv_id': 'c'}).json()
            assert result['workflow_status'] == 'waiting_user'
            assert result['budget']['used']['model'] == 3  # bank + query + one planner
            assert len(calls) == 2 and len(client.calls) == 1
            payload = json.loads(client.calls[0]['messages'][-1]['content'])
            semantic = payload['local_understanding']['semantic_retrieval']
            assert semantic['status'] == 'retrieved' and semantic['remote_calls'] == 2
            assert any(row['source'] == 'semantic_embedding' for row in payload['local_understanding']['candidates'])
            assert 'vectors' not in json.dumps(payload)
            assert options[0]['max_retries'] == 0 and options[0]['query_instruction'] == INSTRUCTION
            assert options[0]['base_url'] == 'https://api.siliconflow.cn/v1'
            assert options[0]['model'] == 'Qwen/Qwen3-Embedding-0.6B'
            next_query = http.post('/chat', json={'message': '查消化内科医生', 'user_id': 'u', 'conv_id': 'c'}).json()
            assert next_query['workflow_status'] == 'completed'
            # Local routing still skips embedding/planning; final answer attempts one stream.
            assert next_query['budget']['used']['model'] == 1 and len(calls) == 2
            assert next_query['answer_generation']['status'] == 'deterministic_fallback'
    finally:
        factory.get_chat_runtime.cache_clear()
        factory.get_registry.cache_clear()


def test_explicit_remote_requires_config_and_auto_does_not_create_fake_vectors(monkeypatch):
    monkeypatch.delenv('CAREGUIDE_EMBEDDING_API_KEY', raising=False)
    assert factory.intent_semantic_retriever('auto') == (None, 'provider_unconfigured')
    assert factory.intent_semantic_retriever('local_ngram') == (None, 'local_ngram')
    with pytest.raises(ValueError, match='CAREGUIDE_EMBEDDING_API_KEY'):
        factory.intent_semantic_retriever('remote')


def test_evaluation_cases_are_independent_and_baseline_is_network_free():
    from scripts.evaluate_intent_semantic import evaluate
    result = asyncio.run(evaluate('local_char_ngram'))
    assert result['cases'] == 26 and result['scored_cases'] == 22
    assert result['remote_calls'] == 0 and result['direct_acceptance'] is False
    assert 0 <= result['mean_action_coverage'] <= 1
    assert {r['category'] for r in result['rows']} >= {'negation', 'reference', 'multiple_goals', 'out_of_scope'}


def test_concurrent_requests_share_only_public_index_and_charge_each_query(tmp_path):
    embedding, calls = provider()
    retriever = SemanticIntentRetriever(embedding)
    async def run():
        left, right = ledger(tmp_path, 'left'), ledger(tmp_path, 'right')
        output = await asyncio.gather(
            retriever.retrieve('第一位用户的问题', budget=left, scope=('left', 'c')),
            retriever.retrieve('第二位用户的问题', budget=right, scope=('right', 'c')))
        assert all(meta['status'] == 'retrieved' for _, meta in output)
        assert len(calls) == 3
        assert sorted([left.summary()['used']['model'], right.summary()['used']['model']]) == [1, 2]
        assert sum(len(call['input']) > 1 for call in calls) == 1
        await retriever.close()
    asyncio.run(run())


def test_embedding_cost_survives_model_discovered_new_workflow(tmp_path):
    from scripts.reproduce_orchestration import build
    from careguide.orchestration.unified import UnifiedRuntime
    from careguide.orchestration.understanding_planner import UnderstandingPlanner
    embedding, calls = provider()
    old, _ = build(tmp_path, [])
    runtime = UnifiedRuntime(old.engine, old.store.path,
        understanding=LocalUnderstanding(old.engine.registry.corpus, semantic=SemanticIntentRetriever(embedding)))
    async def run():
        first = await runtime.process('查消化内科医生', 'u', 'c')
        original = runtime.store.get(first['workflow']['id'], 'u', 'c')
        text = '帮我找一位心内科的大夫'
        client = ScriptedClient([reply({'kind': 'new', 'plan': {'goal': text, 'tasks': [{
            'id': 'doctor', 'fragment': text, 'action': 'search_doctors', 'agent': 'appointment',
            'arguments': {'department': '心血管内科'}, 'completion': 'items'}]}})])
        runtime.engine.client = client
        runtime.planner = UnderstandingPlanner(client)
        result = await runtime.process(text, 'u', 'c')
        assert result['workflow_status'] == 'completed' and result['workflow']['id'] != original.id
        # Bank, query, planner and attempted answer all stay on the same ledger.
        assert result['budget']['used']['model'] == 4
        assert runtime.ledger(original).summary()['used']['model'] == 4
        assert result['budget']['used']['tool'] == 2 and len(calls) == 2 and len(client.calls) == 2
        await runtime.understanding.close()
    asyncio.run(run())


def test_no_chat_planner_means_no_remote_embedding_cost(tmp_path):
    from scripts.reproduce_orchestration import build
    from careguide.orchestration.unified import UnifiedRuntime
    embedding, calls = provider()
    old, _ = build(tmp_path, [])
    runtime = UnifiedRuntime(old.engine, old.store.path,
        understanding=LocalUnderstanding(old.engine.registry.corpus, semantic=SemanticIntentRetriever(embedding)))
    async def run():
        result = await runtime.process('想找看胃病的大夫', 'u', 'c')
        assert result['workflow_status'] == 'waiting_user'
        assert result['budget']['used']['model'] == 0 and calls == []
        trace = runtime.traces.get(result['request_id'])
        local = trace['task_understanding']['intent_signals']['local_understanding']
        assert local['semantic_retrieval']['status'] == 'no_planner'
        await runtime.understanding.close()
    asyncio.run(run())


def test_http_timeout_is_capped_by_actual_request_budget(tmp_path):
    from careguide.orchestration.budget import use_budget
    timeouts = []
    embedding, calls = provider(timeouts=timeouts)
    retriever = SemanticIntentRetriever(embedding, timeout=3)
    async def run():
        with use_budget(ledger(tmp_path), deadline=time.monotonic()+.8):
            rows, metadata = await retriever.retrieve('一个需要理解的请求', budget=ledger(tmp_path), scope=('u', 'c'))
        assert rows and metadata['remote_calls'] == len(calls) == 2
        assert all(0 < value <= .8 for timeout in timeouts for value in timeout.values())
        await retriever.close()
    asyncio.run(run())
