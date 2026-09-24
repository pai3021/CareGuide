"""Build a resumable full-corpus Qwen dense index for KUAKE-IR."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import chromadb

from careguide.evaluation.kuake_ir_dense import (QUERY_INSTRUCTION,
    KuakeIRDenseIndex, write_progress)
from careguide.retrieval.embedding_provider import OpenAICompatibleEmbeddingFunction
from scripts.import_kuake_ir import DEFAULT_SOURCE, EXPECTED, sha256
from scripts.private_config import load_config


def embedding_provider(dimensions: int, batch_size: int):
    key = os.environ.get("CAREGUIDE_EMBEDDING_API_KEY", "")
    if not key or key.startswith("YOUR_"):
        raise ValueError("CAREGUIDE_EMBEDDING_API_KEY is required")
    return OpenAICompatibleEmbeddingFunction(
        api_key=key,
        base_url=os.environ.get("CAREGUIDE_EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1"),
        model=os.environ.get("CAREGUIDE_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B"),
        dimensions=dimensions, encoding_format="float", batch_size=batch_size,
        query_instruction=QUERY_INSTRUCTION, timeout=120, max_retries=4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", type=Path)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--chroma-dir", type=Path,
                        default=DEFAULT_SOURCE / "dense_chroma")
    parser.add_argument("--dimensions", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    load_config(args.config_file)
    corpus = args.source_dir / "corpus.tsv"
    expected_count, expected_hash = EXPECTED["corpus.tsv"]
    if sha256(corpus) != expected_hash:
        raise ValueError("KUAKE-IR corpus integrity mismatch")
    target = min(expected_count, args.limit) if args.limit else expected_count
    print(json.dumps({"verified_corpus": expected_count, "target": target,
                      "dimensions": args.dimensions, "apply": args.apply}, ensure_ascii=False))
    if not args.apply:
        print("Dry run: no collection writes and no external embedding calls.")
        return

    embedding = embedding_provider(args.dimensions, args.batch_size)
    index = KuakeIRDenseIndex(chromadb.PersistentClient(path=str(args.chroma_dir)), embedding,
                              corpus_sha256=expected_hash, expected_count=expected_count)
    started = time.monotonic(); progress_path = args.source_dir / "dense_index_progress.json"
    initial_count = index.collection(create=True).count()
    def progress(total, target_count):
        elapsed = time.monotonic() - started
        value = {"collection": index.name, "total": total, "target": target_count,
                 "elapsed_seconds": round(elapsed, 2),
                 "documents_per_second": round(((total - initial_count) / elapsed), 2) if elapsed else None}
        write_progress(progress_path, value)
        print(json.dumps(value, ensure_ascii=False), flush=True)
    result = index.build(corpus, limit=args.limit, progress=progress)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
