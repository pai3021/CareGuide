"""Offline MMR ablation over actual complete reranker responses.

Uses existing document vectors only; no new model calls or gold inference
features. Five preset weights are selected on tuning50, then evaluated once on
validation50. Both halves are reused development data, not an independent test.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np

from careguide.corpus import Corpus
from careguide.evaluation.kuake_ir_dense import ranking_metrics
from scripts.analyze_rag_failures import read_json, trace
from scripts.cache_rag_recall_scores import atomic_json
from scripts.compare_rag_embeddings import MODEL, DIMENSIONS, cache_vectors, inputs, validate
from scripts.evaluate_rag_candidate_pools import BASELINE, rank
from scripts.evaluate_rag_rank_retention import SEED
from scripts.rerank_rag_candidate_plan import paired_interval, sha
from scripts.run_careguide_kuake_rag_evaluation import CORPUS_DIR, load_cases

WEIGHTS = (1., .95, .9, .8, .6)


def mmr_order(scores, similarities, weight):
    scores, similarities = np.asarray(scores), np.asarray(similarities)
    n = len(scores)
    if (not 0 <= weight <= 1 or scores.shape != (n,) or similarities.shape != (n, n)
            or not np.isfinite(scores).all() or not np.isfinite(similarities).all()
            or not np.allclose(similarities, similarities.T, atol=1e-6)):
        raise ValueError('Invalid MMR inputs')
    if weight == 1:
        return sorted(range(n), key=lambda i: (-scores[i], i))
    selected, available = [], set(range(n))
    while available:
        best = max(available, key=lambda i: (
            weight*scores[i]-(1-weight)*(max(0., float(max(similarities[i, selected]))) if selected else 0.), -i))
        selected.append(best)
        available.remove(best)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'old-scores', 'new-scores', 'responses', 'output', 'cases-output', 'similarity-cache'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--vector-db', type=Path)
    parser.add_argument('--similarity-source', type=Path)
    args = parser.parse_args()
    if any(p.exists() for p in (args.output, args.cases_output, args.similarity_cache)):
        parser.error('Use new output paths')
    plan, old, new, response = map(read_json, (args.plan, args.old_scores, args.new_scores, args.responses))
    validate(plan, old)
    validate(plan, new)
    if response['identity']['plan_sha256'] != sha(args.plan) or sha(args.old_scores) != plan['baseline_scores_sha256']:
        raise ValueError('Experiment inputs differ')
    rows = {'old': {r['id']: r for r in old['rows']}, 'new': {r['id']: r for r in new['rows']}}
    actual = {'old': {r['case_id']: r for r in plan['baseline_rerank']},
              'new': {r['case_id']: r for r in response['completed']}}
    for name, source in actual.items():
        if set(source) != set(plan['case_ids']):
            raise ValueError('Missing or foreign actual responses')
        for key, row in source.items():
            if (len(row['ids']) != 40 or set(row['ids']) != set(row['candidate_ids'])
                    or row['candidate_ids'] != rank(rows[name][key], BASELINE)[:40]):
                raise ValueError('Requires all 40 genuine reranker scores')
    expected_identity = {'model': MODEL, 'dimensions': DIMENSIONS, 'corpus_fingerprint': plan['corpus_fingerprint']}
    if args.similarity_source:
        similarity = read_json(args.similarity_source)
        if similarity['identity'] != expected_identity or set(similarity['arms']) != set(actual):
            raise ValueError('Similarity cache identity differs')
        for name, source in actual.items():
            if set(similarity['arms'][name]) != set(source):
                raise ValueError('Missing/foreign similarity rows')
            for key, row in source.items():
                if similarity['arms'][name][key]['ids'] != row['ids']:
                    raise ValueError('Similarity matrix document order differs')
    else:
        if args.vector_db is None:
            parser.error('Existing vector-db or public similarity-source required')
        with closing(sqlite3.connect(args.vector_db.resolve().as_uri()+'?mode=ro', uri=True)) as db:
            identity = json.loads(db.execute('select identity from config').fetchone()[0])
            if identity != {'plan_sha256': sha(args.plan), 'model': MODEL, 'dimensions': DIMENSIONS,
                            'endpoint': 'https://api.siliconflow.cn/v1/embeddings'}:
                raise ValueError('Document vectors belong to another model/plan')
            _, _, _, documents = inputs()
            vectors = {key: np.frombuffer(blob, dtype='<f4') for key, (_, blob) in cache_vectors(db, documents, 'doc').items()}
        if set(vectors) != {r.id for r in Corpus(CORPUS_DIR).records}:
            raise ValueError('Incomplete document vector checkpoint')
        similarity = {'identity': expected_identity, 'arms': {}}
        for name, source in actual.items():
            similarity['arms'][name] = {}
            for key in plan['case_ids']:
                matrix = np.asarray([vectors[doc] for doc in source[key]['ids']])
                lengths = np.linalg.norm(matrix, axis=1, keepdims=True)
                if matrix.shape != (40, DIMENSIONS) or not np.isfinite(matrix).all() or np.any(lengths == 0):
                    raise ValueError('Invalid actual document vectors')
                normalized = matrix/lengths
                # Small fixed pools do not require an external BLAS DLL/runtime.
                cosine = np.einsum('ik,jk->ij', normalized, normalized, optimize=False)
                similarity['arms'][name][key] = {'ids': source[key]['ids'], 'cosine': np.clip(cosine, -1, 1).tolist()}
    keys = sorted(plan['case_ids'], key=lambda key: hashlib.sha256((SEED+key).encode()).hexdigest())
    parts = {'tuning50': keys[:50], 'validation50': keys[50:], 'all100': keys}
    results, traces = {}, []
    queries = {r['id']: r['query'] for r in load_cases('dev')[0]}
    records = {r.id: r for r in Corpus(CORPUS_DIR).records}
    for name, source in actual.items():
        rankings = {}
        for weight in WEIGHTS:
            rankings[weight] = {key: [source[key]['ids'][i] for i in mmr_order(source[key]['scores'], similarity['arms'][name][key]['cosine'], weight)] for key in keys}
        def metrics(weight, identities):
            return ranking_metrics([rankings[weight][key] for key in identities], [rows[name][key]['gold_record_id'] for key in identities], cutoff=10)
        trials = [{'weight': weight, 'tuning': metrics(weight, parts['tuning50'])} for weight in WEIGHTS]
        best = max(trials, key=lambda r: (r['tuning']['recall_at_10'], r['tuning']['mrr_at_10'], r['weight']))['weight']
        changes = [int(rows[name][key]['gold_record_id'] in rankings[best][key][:10])-int(rows[name][key]['gold_record_id'] in rankings[1.][key][:10]) for key in parts['validation50']]
        results[name] = {'selected_weight': best, 'tuning_trials': trials,
                         'baseline': {p: metrics(1., ids) for p, ids in parts.items()},
                         'selected': {p: metrics(best, ids) for p, ids in parts.items()},
                         'paired_validation': {**paired_interval(changes), 'gained': changes.count(1), 'lost': changes.count(-1)}}
        for key in plan['case_ids']:
            actual_row = source[key]
            gold = rows[name][key]['gold_record_id']
            traced = trace(rows[name][key], actual_row['candidate_ids'], actual_row['ids'][:10])
            if traced['stage'] != 'final10_hit':
                traces.append({'arm': name, 'id': key, 'query': queries[key], 'trace': traced,
                               'gold_record_id': gold, 'gold_content': records[gold].content,
                               'rerank_full_gold_rank': actual_row['ids'].index(gold)+1 if gold in actual_row['ids'] else None,
                               'top3': [{'id': doc, 'content': records[doc].content} for doc in actual_row['ids'][:3]],
                               'semantic_cause': 'not automatically adjudicated'})
    args.similarity_cache.parent.mkdir(parents=True, exist_ok=True)
    args.similarity_cache.write_bytes(gzip.compress(json.dumps(similarity, ensure_ascii=False).encode(), mtime=0))
    report = {'evaluation': 'fixed MMR diversity over actual reranker rankings', 'weights': WEIGHTS,
              'external_requests': 0, 'frozen_test_run': False, 'partition_seed': SEED,
              'results': results, 'source_sha256': {name: sha(path) for name, path in vars(args).items() if path is not None and name in {'plan', 'old_scores', 'new_scores', 'responses', 'vector_db', 'similarity_source', 'similarity_cache'}},
              'stage_counts': {name: {stage: sum(r['arm'] == name and r['trace']['stage'] == stage for r in traces) for stage in ('absent_from_retrieval_union', 'removed_at_pool_cutoff', 'inside_pool_outside_final10')} for name in actual},
              'scope': 'Exploratory reused dev. No added clinical or semantic labels; inference uses only scores and document similarities. No runtime change.'}
    atomic_json(args.output, report)
    args.cases_output.parent.mkdir(parents=True, exist_ok=True)
    args.cases_output.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in traces), encoding='utf-8')
    print(json.dumps({'results': results, 'stage_counts': report['stage_counts']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
