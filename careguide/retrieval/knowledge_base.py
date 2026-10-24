"""Scope-first BM25 and optional real-embedding Chroma/RRF medical retrieval.

No deterministic hash embeddings or legacy operations corpora are used. Querying
never modifies source files or the Chroma collection. Dense indexing is explicit.
"""
from __future__ import annotations

from datetime import date
from typing import Any

from careguide.corpus import Corpus, Record
from careguide.retrieval.bm25_index import BM25Index

COLLECTION_NAME = 'careguide_medical_v1'


def versioned_collection_name(corpus):
    return COLLECTION_NAME+'_'+corpus.fingerprint[:12]


class MedicalKnowledgeBase:
    def __init__(self, corpus: Corpus, *, collection: Any = None, embedding: Any = None,
                 mode: str = 'bm25', min_dense_score: float = -1.0,
                 reranker: Any = None, rerank_candidates: int = 20,
                 dense_weight: float = 4.0, lexical_weight: float = 1.0,
                 hybrid_candidates: int = 100):
        if mode not in {'bm25', 'dense', 'hybrid'}:
            raise ValueError('Retrieval mode must be bm25, dense or hybrid')
        if not -1 <= min_dense_score <= 1:
            raise ValueError('Dense threshold must be between -1 and 1')
        if dense_weight <= 0 or lexical_weight <= 0:
            raise ValueError('Hybrid RRF weights must be positive')
        if hybrid_candidates < 1:
            raise ValueError('Hybrid candidate depth must be positive')
        if mode != 'bm25' and (collection is None or embedding is None):
            raise ValueError('Dense retrieval requires an explicitly indexed collection and real embedding provider')
        self.corpus, self.collection, self.embedding = corpus, collection, embedding
        self.mode, self.min_dense_score = mode, min_dense_score
        self.reranker = reranker
        self.rerank_candidates = max(1, int(rerank_candidates))
        self.dense_weight = float(dense_weight)
        self.lexical_weight = float(lexical_weight)
        self.hybrid_candidates = int(hybrid_candidates)
        self.lexical_index = BM25Index()
        self.lexical_index.upsert({
            'chunk_id': record.id, 'content': record.content,
            'retrieval_text': self.retrieval_text(record), 'metadata': {}}
            for record in corpus.records)
        if mode != 'bm25':
            self._validate_collection()

    def _validate_collection(self):
        metadata = self.collection.metadata or {}
        if metadata.get('reuse_complete') is False:
            raise ValueError('Medical index reuse has not completed verification')
        expected = {'corpus_fingerprint': self.corpus.fingerprint,
                    'embedding_model': self.embedding.model,
                    'embedding_dimensions': self.embedding.dimensions,
                    'hnsw:space': 'cosine'}
        if self.collection.name not in {COLLECTION_NAME, versioned_collection_name(self.corpus)} or any(metadata.get(k) != v for k, v in expected.items()):
            raise ValueError('Medical collection identity, corpus or embedding metadata mismatch')
        actual_ids = set(self.collection.get(include=[])['ids'])
        if actual_ids != {r.id for r in self.corpus.records}:
            raise ValueError('Medical collection records disagree with verified corpus')

    @staticmethod
    def retrieval_text(record: Record) -> str:
        return ' '.join([record.title, *record.keywords, record.content])

    def search(self, query: str, candidates: list[Record], *, today: date, top_k: int = 5) -> list[dict]:
        # Select scope BEFORE lexical scoring and vector querying. A different
        # hospital or product cannot occupy candidate slots or re-enter via RRF.
        eligible = {r.id: r for r in candidates if self.corpus.fresh(r, today)}
        if not eligible:
            return []
        lexical_limit = self.hybrid_candidates if self.mode == 'hybrid' else len(eligible)
        lexical = self.lexical_index.search(
            query, top_k=min(len(eligible), lexical_limit),
            eligible_ids=set(eligible)) if self.mode != 'dense' else []
        preferred_ids: set[str] = set()
        if lexical:
            # When an official/curated note is essentially tied with an open
            # dataset QA row, prefer the narrower reviewed source.  This is a
            # source-tier tie-break, not a claim that curated data is always
            # semantically relevant; the 90% score gate preserves clear wins.
            top_score = lexical[0].score
            preferred = [hit for hit in lexical
                         if eligible[hit.chunk_id].representation == 'curated_factual_summary'
                         and hit.score >= top_score * 0.9]
            if preferred:
                preferred_ids = {hit.chunk_id for hit in preferred}
                lexical = preferred + [hit for hit in lexical if hit.chunk_id not in preferred_ids]
        dense = []
        if self.mode != 'bm25':
            dense_limit = (self.hybrid_candidates if self.mode == 'hybrid' else
                           max(top_k, self.rerank_candidates if self.reranker else top_k))
            result = self.collection.query(
                query_embeddings=[self.embedding.embed_query(query)],
                n_results=min(dense_limit, len(eligible)), where={'record_id': {'$in': list(eligible)}},
                include=['distances'])
            dense = [(identity, 1-float(distance)) for identity, distance
                     in zip(result['ids'][0], result['distances'][0])
                     if identity in eligible and 1-float(distance) >= self.min_dense_score]
        rows = {}
        if self.mode == 'bm25':
            for hit in lexical:
                tiebreak = lexical[0].score if hit.chunk_id in preferred_ids else 0.0
                rows[hit.chunk_id] = {'mode': 'bm25', 'bm25_score': hit.score,
                                     'matched_terms': hit.matched_term_count,
                                     'source_tier_tiebreak': bool(tiebreak),
                                     'rank_score': hit.score + tiebreak}
        elif self.mode == 'dense':
            rows = {identity: {'mode': 'dense', 'cosine_score': score, 'rank_score': score}
                    for identity, score in dense}
        else:
            # RRF ranking is not a relevance probability or clinical confidence.
            # Weights are development-set ranking parameters, not confidence.
            for rank, hit in enumerate(lexical, 1):
                rows[hit.chunk_id] = {'mode': 'hybrid_rrf', 'bm25_rank': rank,
                                     'bm25_score': hit.score,
                                     'rrf_weights': {'dense': self.dense_weight,
                                                     'bm25': self.lexical_weight},
                                     'rank_score': self.lexical_weight/(60+rank)}
            for rank, (identity, score) in enumerate(dense, 1):
                row = rows.setdefault(identity, {
                    'mode': 'hybrid_rrf', 'rank_score': 0.0,
                    'rrf_weights': {'dense': self.dense_weight,
                                    'bm25': self.lexical_weight}})
                row.update({'dense_rank': rank, 'cosine_score': score})
                row['rank_score'] += self.dense_weight/(60+rank)
        initial_k = max(top_k, self.rerank_candidates) if self.reranker else top_k
        ordered = sorted(rows, key=lambda key: (-rows[key]['rank_score'], key))[:initial_k]
        if self.reranker and ordered:
            identities, scores, _ = self.reranker.rerank(
                query, [(key, eligible[key].content) for key in ordered], top_n=top_k)
            ordered = identities
            for key, score in zip(ordered, scores):
                rows[key]['rerank_model'] = self.reranker.model
                rows[key]['rerank_score'] = score
                rows[key]['pre_rerank_mode'] = rows[key]['mode']
                rows[key]['mode'] = f"{rows[key]['mode']}_rerank"
        return [{**self.corpus.evidence(eligible[key]), 'retrieval': rows[key]} for key in ordered]


def index_collection(client: Any, corpus: Corpus, embedding: Any, *, versioned=False):
    """Explicit offline ingestion operation; never called by query handlers."""
    name = versioned_collection_name(corpus) if versioned else COLLECTION_NAME
    if name in {c.name for c in client.list_collections()}:
        raise ValueError('Medical collection already exists; refusing implicit overwrite')
    metadata = {'corpus_fingerprint': corpus.fingerprint, 'hnsw:space': 'cosine',
                'embedding_model': embedding.model, 'embedding_dimensions': embedding.dimensions}
    # Embedding validation occurs before creating any collection.
    records = list(corpus.records)
    vectors = embedding.embed_documents([MedicalKnowledgeBase.retrieval_text(r) for r in records])
    if len(vectors) != len(records):
        raise ValueError('Embedding count mismatch')
    collection = client.create_collection(name, metadata=metadata, embedding_function=None)
    collection.add(ids=[r.id for r in records], embeddings=vectors,
                   documents=[r.content for r in records],
                   metadatas=[{'record_id': r.id, 'source_id': r.source_id, 'kind': r.kind.value} for r in records])
    return collection
