"""Replay three native exact-search candidate pools with cached real vectors.

No reranker response is reused for changed pools. No API client/configuration
or hospital tools. This is runtime wiring verification, not new quality cases.
"""
import argparse
import json
import sqlite3
from contextlib import closing
from datetime import date
from pathlib import Path

import numpy as np

from careguide.corpus import Corpus
from careguide.retrieval.knowledge_base import MedicalKnowledgeBase, versioned_collection_name
from scripts.analyze_rag_failures import read_json
from scripts.cache_rag_recall_scores import atomic_json
from scripts.compare_rag_embeddings import MODEL, DIMENSIONS, experiment_dir
from scripts.evaluate_rag_candidate_pools import BASELINE, rank
from scripts.evaluate_rag_score_fusion import validate_cache
from scripts.run_careguide_kuake_rag_evaluation import CORPUS_DIR, load_cases


def main():
    import chromadb
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'work-dir', 'exact-cache', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    plan, cache = read_json(args.plan), read_json(args.exact_cache)
    validate_cache(cache)
    if cache['metadata'].get('search_algorithm') != 'exact_cosine':
        raise ValueError('Requires actual exact cosine scores')
    work = experiment_dir(args.work_dir)
    cases, _ = load_cases('dev')
    chosen = [next(c for c in cases if c['id'] == key) for key in plan['case_ids'][:3]]
    expected = {row['id']: row for row in cache['rows']}
    with closing(sqlite3.connect((work/'vectors.sqlite3').as_uri()+'?mode=ro', uri=True)) as db:
        vectors = {c['query']: np.frombuffer(db.execute('select vector from vectors where kind=? and id=?',
                   ('query', c['id'])).fetchone()[0], dtype='<f4').tolist() for c in chosen}
    class CachedEmbedding:
        model, dimensions = MODEL, DIMENSIONS
        def embed_query(self, query):
            return vectors[query]
    corpus = Corpus(CORPUS_DIR)
    client = chromadb.PersistentClient(path=str(work/'chroma'), settings=chromadb.Settings(anonymized_telemetry=False))
    knowledge = MedicalKnowledgeBase(corpus, mode='hybrid', dense_search='exact', embedding=CachedEmbedding(),
        collection=client.get_collection(versioned_collection_name(corpus), embedding_function=None), hybrid_candidates=100)
    outcomes = []
    for case in chosen:
        hits = knowledge.search(case['query'], list(corpus.records), today=date(2026, 11, 6), top_k=80)
        ids = [r['record_id'] for r in hits]
        if ids != rank(expected[case['id']], BASELINE)[:80]:
            raise ValueError('Native exact candidate pool differs from offline exact cosine result')
        outcomes.append({'id': case['id'], 'candidate_count': len(ids), 'candidate_ids_match': True})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, {'external_requests': 0, 'dense_search': 'exact', 'outcomes': outcomes,
        'scope': 'Three native candidate pool checks, no extra quality samples or hypothetical reranking.'})
    print(json.dumps({'native_exact_checks_passed': len(outcomes), 'external_requests': 0}))


if __name__ == '__main__':
    main()
