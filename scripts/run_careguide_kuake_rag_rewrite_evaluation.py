"""Evaluate multi-query Dense retrieval, ID deduplication, and reranking on dev."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

from careguide.corpus import Corpus
from careguide.evaluation.kuake_ir_dense import ranking_metrics
from careguide.llm_utils import (create_chat_completion, create_llm_client,
                                 extract_text_content)
from careguide.retrieval.knowledge_base import versioned_collection_name
from careguide.runtime import chroma_client, semantic_embedding, semantic_reranker
from scripts.private_config import load_config
from scripts.run_careguide_kuake_rag_evaluation import load_cases, CORPUS_DIR, ROOT


async def rewrite_one(client, model: str, query: str, semaphore: asyncio.Semaphore) -> list[str]:
    prompt = (
        "把下面的中文医疗检索问题改写为两个语义等价、表达不同的检索查询。"
        "不得新增症状、药品、人群或诊断信息。只返回JSON："
        '{"rewrites":["改写1","改写2"]}\n原问题：' + query
    )
    async with semaphore:
        for attempt in range(8):
            try:
                response = await create_chat_completion(
                    client, model=model, temperature=0.0, max_tokens=160,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"})
                content = extract_text_content(response).strip()
                start, end = content.find("{"), content.rfind("}")
                if start < 0 or end < start:
                    raise ValueError("rewrite response is not a JSON object")
                data = json.loads(content[start:end + 1])
                values = data.get("rewrites")
                if not isinstance(values, list):
                    raise ValueError("rewrite response is missing rewrites")
                rewrites = []
                for value in values:
                    text = str(value).strip()
                    if text and text != query and text not in rewrites:
                        rewrites.append(text[:200])
                if len(rewrites) != 2:
                    raise ValueError("rewrite response must contain two distinct queries")
                return rewrites
            except Exception:
                if attempt == 7:
                    return []
                await asyncio.sleep(min(30.0, (2 ** attempt) + random.random()))
    raise RuntimeError("unreachable rewrite retry state")


async def build_rewrites(cases: list[dict], cache_path: Path, *, model: str,
                         client, concurrency: int) -> dict[str, list[str]]:
    cached: dict[str, dict] = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            cached[row["id"]] = row
    for case in cases:
        row = cached.get(case["id"])
        if row and (row.get("query") != case["query"] or row.get("model") != model):
            raise ValueError("rewrite cache identity mismatch")

    missing = [case for case in cases if case["id"] not in cached]
    semaphore = asyncio.Semaphore(concurrency)
    for start in range(0, len(missing), 50):
        batch = missing[start:start + 50]
        values = await asyncio.gather(*[
            rewrite_one(client, model, case["query"], semaphore) for case in batch
        ], return_exceptions=True)
        errors = []
        for case, rewrites in zip(batch, values):
            if isinstance(rewrites, BaseException):
                errors.append((case["id"], rewrites))
                continue
            cached[case["id"]] = {"id": case["id"], "query": case["query"],
                                  "rewrites": rewrites, "model": model}
        ordered = [cached[case["id"]] for case in cases if case["id"] in cached]
        temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                     for row in ordered), encoding="utf-8")
        temporary.replace(cache_path)
        print(f"rewritten {len(cached)}/{len(cases)}", flush=True)
        if errors:
            identity, error = errors[0]
            raise RuntimeError(
                f"rewrite failed after retries for {identity}; "
                f"saved {len(cached)} successful rows"
            ) from error
    return {case["id"]: cached[case["id"]]["rewrites"] for case in cases}


async def build_rewrites_and_close(cases: list[dict], cache_path: Path, *, model: str,
                                   client, concurrency: int) -> dict[str, list[str]]:
    try:
        return await build_rewrites(cases, cache_path, model=model, client=client,
                                    concurrency=concurrency)
    finally:
        await client.close()


def deduplicate(rankings: list[list[str]], limit: int) -> list[str]:
    seen, merged = set(), []
    for ranking in rankings:
        for identity in ranking:
            if identity not in seen:
                seen.add(identity)
                merged.append(identity)
                if len(merged) >= limit:
                    return merged
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--recall-k", type=int, default=20)
    parser.add_argument("--pool-k", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--rewrite-concurrency", type=int, default=8)
    parser.add_argument("--cache", type=Path,
                        default=ROOT / "data/eval/results/careguide_kuake_rag_dev_rewrites_mimo_v1.jsonl")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/eval/results/careguide_kuake_rag_dev_rewrite_dense_rerank_v1.json")
    args = parser.parse_args()
    if not args.top_k <= args.recall_k <= args.pool_k:
        parser.error("expected top-k <= recall-k <= pool-k")
    load_config(args.config_file)
    cases, manifest = load_cases("dev")
    if args.limit:
        cases = cases[:args.limit]
    model = os.environ.get("CAREGUIDE_LLM_MODEL", "")
    key = os.environ.get("CAREGUIDE_LLM_API_KEY", "")
    if not model or not key:
        parser.error("query rewriting requires the configured chat provider")
    client = create_llm_client(key, os.environ.get("CAREGUIDE_LLM_BASE_URL") or None)
    rewrites = asyncio.run(build_rewrites_and_close(
        cases, args.cache, model=model, client=client,
        concurrency=args.rewrite_concurrency))

    corpus = Corpus(CORPUS_DIR)
    embedding = semantic_embedding()
    collection = chroma_client().get_collection(
        versioned_collection_name(corpus), embedding_function=None)
    query_groups = [[case["query"], *rewrites[case["id"]]] for case in cases]
    flat_queries = [query for group in query_groups for query in group]
    vectors = embedding.embed_queries(flat_queries)
    result = collection.query(query_embeddings=vectors, n_results=args.recall_k,
                              include=["distances"])
    flat_rankings = [[str(identity) for identity in values] for values in result["ids"]]
    pools = []
    offset = 0
    for group in query_groups:
        group_rankings = flat_rankings[offset:offset + len(group)]
        pools.append(deduplicate(group_rankings, args.pool_k))
        offset += len(group)

    gold = [case["gold_record_id"] for case in cases]
    by_id = {record.id: record for record in corpus.records}
    reranker = semantic_reranker()
    jobs = [(case["query"], [(identity, by_id[identity].content) for identity in pool])
            for case, pool in zip(cases, pools)]
    reranked = []
    for start in range(0, len(jobs), 50):
        batch = asyncio.run(reranker.rerank_many(jobs[start:start + 50], top_n=args.top_k))
        reranked.extend(batch)
        print(f"reranked {len(reranked)}/{len(jobs)}", flush=True)
    rankings = [identities for identities, _, _ in reranked]
    cache_sha256 = hashlib.sha256(args.cache.read_bytes()).hexdigest()
    report = {
        "evaluation": "CareGuide KUAKE multi-query Dense dedup rerank",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "split": "dev", "frozen_test": False,
        "case_count": len(cases), "rewrite_model": model,
        "rewrites_per_query": 2, "recall_k_per_query": args.recall_k,
        "rewrite_success_count": sum(len(values) == 2 for values in rewrites.values()),
        "rewrite_fallback_count": sum(len(values) != 2 for values in rewrites.values()),
        "deduplicated_pool_k": args.pool_k, "rerank_model": reranker.model,
        "rewrite_cache_sha256": cache_sha256,
        "candidate_gold_coverage": sum(target in pool for target, pool in zip(gold, pools)) / len(gold),
        "mean_unique_candidates": sum(len(pool) for pool in pools) / len(pools),
        "pre_rerank_metrics": ranking_metrics(
            [pool[:args.top_k] for pool in pools], gold, cutoff=args.top_k),
        "metrics": ranking_metrics(rankings, gold, cutoff=args.top_k),
        "runtime_corpus_fingerprint": corpus.fingerprint,
        "official_kuake_dev_used": False,
        "source_manifest_sha256": manifest["runtime_corpus_manifest_sha256"],
        "rerank_input_tokens": sum(int((meta.get("tokens") or {}).get(
            "input_tokens", 0)) for _, _, meta in reranked),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
