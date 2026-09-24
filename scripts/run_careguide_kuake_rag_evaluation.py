"""Evaluate the runtime-aligned CareGuide KUAKE RAG corpus."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

from careguide.corpus import Corpus
from careguide.evaluation.kuake_ir_dense import ranking_metrics, reciprocal_rank_fusion
from careguide.retrieval.knowledge_base import MedicalKnowledgeBase, versioned_collection_name
from careguide.runtime import chroma_client, semantic_embedding, semantic_reranker
from scripts.private_config import load_config


ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "data" / "careguide_kuake_rag_v1"
EVAL_DIR = ROOT / "data" / "eval" / "careguide_kuake_rag_v1"


def load_cases(split: str) -> tuple[list[dict], dict]:
    manifest = json.loads((EVAL_DIR / "manifest.json").read_text(encoding="utf-8"))
    name = f"{split}.jsonl"
    raw = (EVAL_DIR / name).read_bytes()
    expected = manifest["files"][name]
    if hashlib.sha256(raw).hexdigest() != expected["sha256"]:
        raise ValueError(f"CareGuide KUAKE RAG {split} integrity mismatch")
    rows = [json.loads(line) for line in raw.decode().splitlines()]
    if len(rows) != expected["count"] or any(row["split"] != split for row in rows):
        raise ValueError(f"CareGuide KUAKE RAG {split} count mismatch")
    return rows, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--run-frozen-test", action="store_true")
    parser.add_argument("--mode", choices=("bm25", "dense", "hybrid",
                                           "dense_rerank", "hybrid_rerank"),
                        default="bm25")
    parser.add_argument("--config-file", type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--rerank-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--lexical-weight", type=float, default=1.0)
    parser.add_argument("--min-dense-score", type=float, default=-1.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.split == "test" and not args.run_frozen_test:
        parser.error("frozen test requires --run-frozen-test after configuration freeze")
    if args.mode != "bm25" and not args.config_file:
        parser.error("dense/hybrid evaluation requires --config-file")
    if args.rerank_k < args.top_k:
        parser.error("--rerank-k must be at least --top-k")
    if args.rrf_k <= 0 or args.dense_weight <= 0 or args.lexical_weight <= 0:
        parser.error("RRF k and weights must be positive")
    if not -1 <= args.min_dense_score <= 1:
        parser.error("--min-dense-score must be between -1 and 1")
    load_config(args.config_file)

    cases, manifest = load_cases(args.split)
    if args.limit is not None:
        if not 1 <= args.limit <= len(cases):
            parser.error("--limit must be within the selected split")
        cases = cases[:args.limit]
    corpus = Corpus(CORPUS_DIR)
    kwargs = {}
    embedding = None
    if args.mode != "bm25":
        embedding = semantic_embedding()
        client = chroma_client()
        name = versioned_collection_name(corpus)
        kwargs = {"collection": client.get_collection(name, embedding_function=None),
                  "embedding": embedding}
    retrieval_mode = ("bm25" if args.mode == "bm25" else
                      "dense" if args.mode.startswith("dense") else "hybrid")
    knowledge = MedicalKnowledgeBase(corpus, mode=retrieval_mode, **kwargs)
    reranker = None
    rerank_meta = []
    if args.mode == "bm25":
        rankings = []
        for case in cases:
            hits = knowledge.search(case["query"], list(corpus.records),
                                    today=date(2026, 9, 22), top_k=args.top_k)
            rankings.append([hit["record_id"] for hit in hits])
    else:
        candidate_k = max(args.top_k, args.candidate_k)
        vectors = embedding.embed_queries([case["query"] for case in cases])
        dense_result = knowledge.collection.query(
            query_embeddings=vectors, n_results=min(candidate_k, len(corpus.records)),
            include=["distances"])
        dense_rankings = []
        for identities, distances in zip(dense_result["ids"], dense_result["distances"]):
            dense_rankings.append([
                str(identity) for identity, distance in zip(identities, distances)
                if 1.0 - float(distance) >= args.min_dense_score
            ])
        base_rankings = dense_rankings
        if args.mode.startswith("hybrid"):
            lexical_rankings = [[hit.chunk_id for hit in knowledge.lexical_index.search(
                case["query"], top_k=candidate_k)] for case in cases]
            base_rankings = [reciprocal_rank_fusion(
                lexical, dense, constant=args.rrf_k,
                weights=(args.lexical_weight, args.dense_weight))
                             for lexical, dense in zip(lexical_rankings, dense_rankings)]
        if not args.mode.endswith("_rerank"):
            rankings = [values[:args.top_k] for values in base_rankings]
        else:
            reranker = semantic_reranker()
            by_id = {record.id: record for record in corpus.records}
            jobs = []
            for case, values in zip(cases, base_rankings):
                selected = values[:args.rerank_k]
                jobs.append((case["query"], [
                    (identity, by_id[identity].content) for identity in selected
                ]))
            reranked = []
            for start in range(0, len(jobs), 50):
                batch = asyncio.run(reranker.rerank_many(
                    jobs[start:start + 50], top_n=args.top_k))
                reranked.extend(batch)
                print(f"reranked {len(reranked)}/{len(jobs)}", flush=True)
            rankings = [identities for identities, _, _ in reranked]
            rerank_meta = [meta for _, _, meta in reranked]
    gold = [case["gold_record_id"] for case in cases]
    candidate_depth = args.rerank_k if args.mode.endswith("_rerank") else args.top_k
    candidate_gold_coverage = sum(
        target in values[:candidate_depth] for target, values in zip(gold, base_rankings)
    ) / len(gold) if args.mode != "bm25" else None
    report = {
        "evaluation": "CareGuide KUAKE runtime-aligned retrieval",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "frozen_test": args.split == "test",
        "mode": args.mode,
        "candidate_k": args.candidate_k if args.mode != "bm25" else None,
        "candidate_gold_coverage": candidate_gold_coverage,
        "rrf_k": args.rrf_k if args.mode.startswith("hybrid") else None,
        "dense_weight": args.dense_weight if args.mode.startswith("hybrid") else None,
        "lexical_weight": args.lexical_weight if args.mode.startswith("hybrid") else None,
        "min_dense_score": args.min_dense_score if args.mode != "bm25" else None,
        "rerank_k": args.rerank_k if args.mode.endswith("_rerank") else None,
        "rerank_model": reranker.model if reranker else None,
        "rerank_request_count": len(rerank_meta),
        "rerank_input_tokens": sum(int((meta.get("tokens") or {}).get(
            "input_tokens", 0)) for meta in rerank_meta),
        "case_count": len(cases),
        "runtime_corpus_version": corpus.version,
        "runtime_corpus_fingerprint": corpus.fingerprint,
        "runtime_record_count": len(corpus.records),
        "official_kuake_dev_used": False,
        "source_manifest_sha256": manifest["runtime_corpus_manifest_sha256"],
        "metrics": ranking_metrics(rankings, gold, cutoff=args.top_k),
    }
    output = args.output or ROOT / "data" / "eval" / "results" / (
        f"careguide_kuake_rag_{args.split}_{args.mode}_v1.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
