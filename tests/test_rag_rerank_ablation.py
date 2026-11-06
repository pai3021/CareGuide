"""Mock HTTP contract tests; no provider calls or retrieval-quality claims."""
import json

import httpx
import pytest

from careguide.retrieval.reranker import OpenAICompatibleReranker
from scripts.compare_rag_rerank_models import ENGLISH_INSTRUCTION
from scripts.evaluate_rag_rank_retention import retain_ranking


def test_fixed_model_instruction_and_full_ranking_are_sent_without_query_changes(monkeypatch):
    calls = []
    real_client = httpx.Client

    def handler(request):
        payload = json.loads(request.content)
        calls.append(payload)
        assert payload['query'] == '固定的原始查询'
        assert payload['instruction'] == ENGLISH_INSTRUCTION
        assert payload['model'] == 'Qwen/Qwen3-Reranker-4B'
        assert payload['documents'] == ['first','second','third']
        assert payload['top_n'] == 3
        return httpx.Response(200,json={'results':[
            {'index':2,'relevance_score':.9}, {'index':0,'relevance_score':.7},
            {'index':1,'relevance_score':.1}], 'meta':{'tokens':{'input_tokens':50}}})

    monkeypatch.setattr(httpx,'Client',lambda **kwargs:real_client(
        **kwargs,transport=httpx.MockTransport(handler)))
    adapter = OpenAICompatibleReranker(api_key='test-only',base_url='https://example.invalid/v1',
        model='Qwen/Qwen3-Reranker-4B',instruction=ENGLISH_INSTRUCTION,max_retries=0)
    ids,scores,meta = adapter.rerank('固定的原始查询', [('a','first'),('b','second'),('c','third')],top_n=3)
    assert ids == ['c','a','b'] and scores == [.9,.7,.1]
    assert len(calls) == 1 and meta['tokens']['input_tokens'] == 50


def test_disabled_retries_do_not_send_a_second_paid_attempt(monkeypatch):
    calls = []
    real_client = httpx.Client

    def handler(request):
        calls.append(request)
        return httpx.Response(429,json={'error':'rate limited'})

    monkeypatch.setattr(httpx,'Client',lambda **kwargs:real_client(
        **kwargs,transport=httpx.MockTransport(handler)))
    adapter = OpenAICompatibleReranker(api_key='test-only',base_url='https://example.invalid/v1',
        model='Qwen/Qwen3-Reranker-4B',max_retries=0)
    with pytest.raises(httpx.HTTPStatusError):
        adapter.rerank('query',[('a','passage')],top_n=1)
    assert len(calls) == 1


def test_rank_retention_requires_complete_same_pool_and_preserves_endpoints():
    original,reranked = ['a','b','c'],['c','b','a']
    assert retain_ranking(original,reranked,0) == original
    assert retain_ranking(original,reranked,1) == reranked
    with pytest.raises(ValueError,match='complete'):
        retain_ranking(original,['c','b'],.5)
    with pytest.raises(ValueError,match='complete'):
        retain_ranking(['a','a'],['a','b'],.5)
