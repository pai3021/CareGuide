"""Evaluate KUAKE-IR Dense and lexical+Dense RRF on the official dev queries."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import chromadb

from careguide.evaluation.kuake_ir import KuakeIRFTSIndex
from careguide.evaluation.kuake_ir_dense import (KuakeIRDenseIndex,
    ranking_metrics, reciprocal_rank_fusion)
from scripts.import_kuake_ir import DEFAULT_SOURCE, EXPECTED, sha256
from scripts.index_kuake_ir_dense import embedding_provider
from scripts.private_config import load_config
from scripts.run_kuake_ir_evaluation import ROOT, pairs


def select_items(query_map: dict[str, str], *, limit: int | None = None,
                 sample_size: int | None = None,
                 sample_seed: str = "careguide-kuake-ir-fast-v1"):
    if limit is not None and sample_size is not None:
        raise ValueError("--limit and --sample-size are mutually exclusive")
    items = list(query_map.items())
    if sample_size is not None:
        if not 1 <= sample_size <= len(items):
            raise ValueError("sample-size must be within the available query count")
        items.sort(key=lambda item: hashlib.sha256(
            f"{sample_seed}\0{item[0]}".encode()).hexdigest())
        return items[:sample_size], "deterministic_hash_sample"
    if limit is not None:
        return items[:limit], "first_n_smoke"
    return items, "official_full_dev"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--chroma-dir", type=Path, default=DEFAULT_SOURCE / "dense_chroma")
    parser.add_argument("--fts-index", type=Path, default=DEFAULT_SOURCE / "kuake_ir_fts.sqlite3")
    parser.add_argument("--dimensions", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--candidate-k", type=int, default=200)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-size", type=int,
                        help="Deterministic query sample; defaults to 200 and still searches the complete corpus")
    parser.add_argument("--sample-seed", default="careguide-kuake-ir-fast-v1")
    parser.add_argument("--full", action="store_true",
                        help="Explicitly evaluate all 1000 official dev queries")
    parser.add_argument("--allow-partial-index", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/eval/results/kuake_ir_dense_hybrid_dev_v1.json")
    args = parser.parse_args()
    if args.full and (args.limit is not None or args.sample_size is not None):
        parser.error("--full cannot be combined with --limit or --sample-size")
    effective_sample_size = (None if args.full or args.limit is not None
                             else args.sample_size or 200)
    load_config(args.config_file)
    source = args.source_dir.resolve(); expected_count, corpus_hash = EXPECTED["corpus.tsv"]
    if sha256(source / "corpus.tsv") != corpus_hash:
        raise ValueError("KUAKE-IR corpus integrity mismatch")
    embedding = embedding_provider(args.dimensions, args.batch_size)
    dense = KuakeIRDenseIndex(chromadb.PersistentClient(path=str(args.chroma_dir)), embedding,
                              corpus_sha256=corpus_hash, expected_count=expected_count)
    indexed_count = dense.collection().count()
    if indexed_count != expected_count and not args.allow_partial_index:
        raise ValueError(f"Dense index incomplete: {indexed_count}/{expected_count}")

    query_map = pairs(source / "KUAKE-IR_dev_query.txt")
    relevance = pairs(source / "KUAKE-IR_dev.tsv")
    items, query_selection = select_items(query_map, limit=args.limit,
                                          sample_size=effective_sample_size,
                                          sample_seed=args.sample_seed)
    vectors = embedding.embed_queries([query for _, query in items])
    dense_rankings = dense.search_many(vectors, top_k=args.candidate_k,
                                       batch_size=args.batch_size)
    lexical_index = KuakeIRFTSIndex(args.fts_index)
    lexical_rankings = [[identity for identity, _ in lexical_index.search(
        query, top_k=args.candidate_k)] for _, query in items]
    hybrid_rankings = [reciprocal_rank_fusion(lexical, semantic)
                       for lexical, semantic in zip(lexical_rankings, dense_rankings)]
    gold = [relevance[query_id] for query_id, _ in items]
    report = {
        "evaluation": "KUAKE-IR dev Dense and Hybrid/RRF",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "split": "dev", "case_count": len(items), "corpus_count": expected_count,
        "dense_index_count": indexed_count, "candidate_k": args.candidate_k,
        "query_selection": query_selection,
        "sample_seed": args.sample_seed if effective_sample_size is not None else None,
        "embedding_model": embedding.model, "embedding_dimensions": embedding.dimensions,
        "official_full_corpus_result": (indexed_count == expected_count and
                                        len(items) == 1000 and
                                        query_selection == "official_full_dev"),
        "lexical": ranking_metrics(lexical_rankings, gold),
        "dense": ranking_metrics(dense_rankings, gold),
        "hybrid_rrf": ranking_metrics(hybrid_rankings, gold),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
