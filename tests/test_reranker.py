import asyncio
import json

import httpx
import pytest

from careguide.retrieval.reranker import OpenAICompatibleReranker


def test_reranker_preserves_provider_order_and_candidate_identity():
    async def handler(request):
        payload = json.loads(request.content)
        assert payload["documents"] == ["first", "second", "third"]
        return httpx.Response(200, json={"results": [
            {"index": 1, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.4},
        ], "meta": {"tokens": {"input_tokens": 3}}})

    async def run():
        reranker = OpenAICompatibleReranker(
            api_key="test", base_url="https://provider.example/v1",
            model="reranker", concurrency=1)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await reranker._one(
                client, asyncio.Semaphore(1), "query",
                [("a", "first"), ("b", "second"), ("c", "third")], 2)

    ids, scores, meta = asyncio.run(run())
    assert ids == ["b", "a"]
    assert scores == [0.9, 0.4]
    assert meta["tokens"]["input_tokens"] == 3


def test_reranker_rejects_invalid_indices():
    async def handler(request):
        return httpx.Response(200, json={"results": [
            {"index": 2, "relevance_score": 0.9},
        ]})

    async def run():
        reranker = OpenAICompatibleReranker(
            api_key="test", base_url="https://provider.example/v1", model="reranker")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await reranker._one(
                client, asyncio.Semaphore(1), "query", [("a", "first")], 1)

    with pytest.raises(RuntimeError, match="invalid index"):
        asyncio.run(run())
