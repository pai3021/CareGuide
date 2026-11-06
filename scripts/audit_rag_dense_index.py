"""Audit ANN against exact cosine using the existing 4B vector checkpoint.

Read-only SQLite, no model clients/config, no frozen test. This changes only
nearest-neighbor search; BM25 and fusion stay fixed. Reports candidate quality,
not a hypothetical reranker outcome on changed input pools.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path

import numpy as np

from scripts.analyze_rag_failures import read_json
from scripts.cache_rag_recall_scores import atomic_json
from scripts.compare_rag_embeddings import MODEL, DIMENSIONS, cache_vectors, inputs, validate
from scripts.evaluate_rag_candidate_pools import BASELINE, partition_rows, rank
from scripts.evaluate_rag_score_fusion import metrics, paired_recall
from scripts.rerank_rag_candidate_plan import sha


def unit_vectors(values):
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError('Expected finite vector matrix')
    lengths = np.sqrt(np.sum(matrix*matrix, axis=1, keepdims=True))
    if np.any(lengths <= 0):
        raise ValueError('Zero vector has no cosine direction')
    return matrix/lengths


def exact_top_k(ids, documents, queries, *, k):
    if documents.shape[1] != queries.shape[1] or len(ids) != len(documents) or len(set(ids)) != len(ids) or k < 1:
        raise ValueError('Unaligned exact search inputs')
    # Explicit C summation avoids depending on a local BLAS DLL installation.
    similarities = np.einsum('qd,nd->qn', queries, documents, optimize=False)
    return [[(str(ids[i]), float(scores[i])) for i in np.lexsort((ids, -scores))[:k]] for scores in similarities]


def measure(rows):
    result = metrics(rows, [rank(row, BASELINE) for row in rows])
    result['dense_top100'] = sum(row['gold_record_id'] in dict(row['streams']['dense']) for row in rows)/len(rows)
    result['union_top100_each'] = sum(any(row['gold_record_id'] in dict(v) for v in row['streams'].values()) for row in rows)/len(rows)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'baseline-scores', 'vector-db', 'output', 'exact-cache'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.exact_cache.exists():
        parser.error('Use new result paths')
    plan, baseline = read_json(args.plan), read_json(args.baseline_scores)
    validate(plan, baseline)
    if baseline['metadata']['embedding_model'] != MODEL or baseline['metadata']['embedding_dimensions'] != DIMENSIONS:
        raise ValueError('Requires the existing 4B/1024 ANN cache')
    corpus, cases, _, docs = inputs()
    vector_hash_before = sha(args.vector_db)
    with closing(sqlite3.connect(args.vector_db.resolve().as_uri()+'?mode=ro', uri=True)) as db:
        expected = {'plan_sha256': sha(args.plan), 'model': MODEL, 'dimensions': DIMENSIONS,
                    'endpoint': 'https://api.siliconflow.cn/v1/embeddings'}
        if json.loads(db.execute('select identity from config').fetchone()[0]) != expected:
            raise ValueError('Vector checkpoint model/plan identity differs')
        stored_docs = cache_vectors(db, docs, 'doc')
        stored_queries = cache_vectors(db, [(c['id'], c['query']) for c in cases], 'query')
    if len(stored_docs) != len(docs) or len(stored_queries) != len(cases):
        raise ValueError('Requires complete cached documents and queries')
    ids = np.asarray(sorted(stored_docs))
    documents = unit_vectors([np.frombuffer(stored_docs[key][1], dtype='<f4') for key in ids])
    sources = {row['id']: row for row in baseline['rows']}
    exact, overlaps, times = [], [], []
    for start in range(0, len(cases), 16):
        selected = cases[start:start+16]
        queries = unit_vectors([np.frombuffer(stored_queries[c['id']][1], dtype='<f4') for c in selected])
        tick = time.perf_counter()
        found = exact_top_k(ids, documents, queries, k=100)
        times.append(time.perf_counter()-tick)
        for case, dense in zip(selected, found):
            original = sources[case['id']]
            overlaps.append(len(set(dict(dense)) & set(dict(original['streams']['dense'])))/100)
            exact.append({**original, 'streams': {**original['streams'], 'dense': dense}})
        if start % 160 == 0:
            print(json.dumps({'queries_complete': start+len(selected), 'mean_ann_top100_overlap': float(np.mean(overlaps))}), flush=True)
    if sha(args.vector_db) != vector_hash_before:
        raise ValueError('Vector checkpoint changed during audit')
    before_tune, before_valid = partition_rows(baseline['rows'])
    after_tune, after_valid = partition_rows(exact)
    results = {}
    for name, before, after in [('tuning500', before_tune, after_tune), ('validation500', before_valid, after_valid),
                                ('all1000', baseline['rows'], exact)]:
        results[name] = {'ann': measure(before), 'exact': measure(after),
                         'paired_at40': paired_recall(after, [rank(r, BASELINE) for r in before], [rank(r, BASELINE) for r in after], 40)}
    report = {'evaluation': 'ANN versus exact cosine using identical cached 4B vectors',
              'external_requests': 0, 'frozen_test_run': False, 'corpus_fingerprint': corpus.fingerprint,
              'model': MODEL, 'dimensions': DIMENSIONS, 'mean_ann_top100_overlap': float(np.mean(overlaps)),
              'exact_compute_seconds': sum(times), 'batch_size': 16, 'results': results,
              'source_sha256': {'plan': sha(args.plan), 'ann_scores': sha(args.baseline_scores), 'vector_db': vector_hash_before},
              'scope': 'Reused dev, pre-rerank only. Batch CPU compute is not production latency. No runtime/default change.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, report)
    args.exact_cache.parent.mkdir(parents=True, exist_ok=True)
    args.exact_cache.write_bytes(gzip.compress(json.dumps({'metadata': {**baseline['metadata'], 'search_algorithm': 'exact_cosine'}, 'rows': exact}, ensure_ascii=False).encode(), mtime=0))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
