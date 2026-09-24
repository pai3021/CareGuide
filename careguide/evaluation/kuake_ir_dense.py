"""Resumable Chroma dense index and RRF helpers for KUAKE-IR."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


COLLECTION_PREFIX = "careguide_kuake_ir_qwen3"
QUERY_INSTRUCTION = (
    "Given a Chinese medical search query, retrieve passages that directly answer the query."
)


def collection_name(dimensions: int) -> str:
    return f"{COLLECTION_PREFIX}_{int(dimensions)}_v1"


class KuakeIRDenseIndex:
    version = "careguide-kuake-ir-qwen3-dense-v1"

    def __init__(self, client, embedding, *, corpus_sha256: str, expected_count: int):
        self.client, self.embedding = client, embedding
        self.corpus_sha256, self.expected_count = corpus_sha256, int(expected_count)
        self.name = collection_name(embedding.dimensions)

    @property
    def metadata(self) -> dict:
        return {
            "version": self.version,
            "corpus_sha256": self.corpus_sha256,
            "expected_count": self.expected_count,
            "embedding_model": self.embedding.model,
            "embedding_dimensions": self.embedding.dimensions,
            "hnsw:space": "cosine",
        }

    def collection(self, *, create: bool = False):
        names = {item.name for item in self.client.list_collections()}
        if self.name not in names:
            if not create:
                raise ValueError("KUAKE-IR dense collection does not exist")
            return self.client.create_collection(
                self.name, metadata=self.metadata, embedding_function=None)
        value = self.client.get_collection(self.name, embedding_function=None)
        actual = value.metadata or {}
        if any(actual.get(key) != expected for key, expected in self.metadata.items()):
            raise ValueError("KUAKE-IR dense collection metadata mismatch")
        return value

    def build(self, corpus_path: Path, *, limit: int | None = None,
              progress=None) -> dict:
        collection = self.collection(create=True)
        existing = collection.count()
        target = min(self.expected_count, int(limit)) if limit is not None else self.expected_count
        if existing > target:
            raise ValueError("Existing dense index is larger than requested target")
        if existing == target:
            return {"start": existing, "indexed": 0, "total": existing, "target": target}

        ids: list[str] = []
        texts: list[str] = []
        indexed = 0
        with Path(corpus_path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle):
                if line_number < existing:
                    continue
                if line_number >= target:
                    break
                doc_id, text = line.rstrip("\r\n").split("\t", 1)
                ids.append(doc_id); texts.append(text)
                if len(ids) >= self.embedding.batch_size:
                    vectors = self.embedding.embed_documents(texts)
                    collection.add(ids=ids, embeddings=vectors)
                    indexed += len(ids)
                    if progress:
                        progress(existing + indexed, target)
                    ids, texts = [], []
        if ids:
            vectors = self.embedding.embed_documents(texts)
            collection.add(ids=ids, embeddings=vectors)
            indexed += len(ids)
            if progress:
                progress(existing + indexed, target)
        return {"start": existing, "indexed": indexed,
                "total": collection.count(), "target": target}

    def search_many(self, vectors: list[list[float]], *, top_k: int = 200,
                    batch_size: int = 64) -> list[list[str]]:
        collection = self.collection()
        rows: list[list[str]] = []
        for start in range(0, len(vectors), batch_size):
            result = collection.query(
                query_embeddings=vectors[start:start + batch_size],
                n_results=min(top_k, collection.count()), include=["distances"])
            rows.extend([[str(identity) for identity in values] for values in result["ids"]])
        return rows


def reciprocal_rank_fusion(*rankings: Iterable[str], constant: int = 60,
                           weights: Iterable[float] | None = None) -> list[str]:
    rankings = tuple(rankings)
    selected_weights = tuple(weights) if weights is not None else (1.0,) * len(rankings)
    if len(selected_weights) != len(rankings) or any(weight <= 0 for weight in selected_weights):
        raise ValueError("RRF weights must be positive and match the rankings")
    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, selected_weights):
        for rank, identity in enumerate(ranking, 1):
            scores[identity] = scores.get(identity, 0.0) + weight / (constant + rank)
    return sorted(scores, key=lambda identity: (-scores[identity], identity))


def ranking_metrics(rankings: list[list[str]], gold: list[str], *, cutoff: int = 10) -> dict:
    import math
    if len(rankings) != len(gold):
        raise ValueError("Ranking and gold counts disagree")
    ranks = [(values.index(target) + 1 if target in values[:cutoff] else None)
             for values, target in zip(rankings, gold)]
    total = len(ranks) or 1
    return {
        "case_count": len(ranks),
        f"mrr_at_{cutoff}": sum(1 / rank if rank else 0 for rank in ranks) / total,
        "recall_at_1": sum(rank == 1 for rank in ranks) / total,
        "recall_at_5": sum(bool(rank and rank <= 5) for rank in ranks) / total,
        f"recall_at_{cutoff}": sum(bool(rank and rank <= cutoff) for rank in ranks) / total,
        f"ndcg_at_{cutoff}": sum(1 / math.log2(rank + 1) if rank else 0
                                  for rank in ranks) / total,
    }


def write_progress(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
