"""Replay dev score caches offline, tuning and validating on separate halves."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from itertools import product
from pathlib import Path

from careguide.corpus import Corpus
from careguide.evaluation.kuake_ir_dense import ranking_metrics
from careguide.retrieval.score_fusion import fuse_scores, reserve_lexical_candidates
from scripts.cache_rag_recall_scores import atomic_json, query_fingerprint
from scripts.run_careguide_kuake_rag_evaluation import CORPUS_DIR, load_cases

SEED = 'careguide-recall-fusion-dev-v1'


def weight_grid(names, units=10):
    """Simplex grid includes the individual-retriever ablations."""
    for values in product(range(units+1), repeat=len(names)):
        if sum(values) == units:
            yield dict(zip(names, (v/units for v in values)))


def rank(row, method, weights):
    return [key for key, _ in fuse_scores(row['streams'], method=method, weights=weights)]


def metrics(rows, rankings):
    gold = [row['gold_record_id'] for row in rows]
    value = ranking_metrics(rankings, gold, cutoff=10)
    value.update({f'candidate_coverage_at_{depth}': sum(
        target in ranking[:depth] for target, ranking in zip(gold, rankings))/len(gold)
                  for depth in (20, 40, 100)})
    return value


def paired_recall(rows, baseline, candidate, depth):
    wins = losses = 0
    for row, before, after in zip(rows, baseline, candidate):
        target = row['gold_record_id']
        wins += target not in before[:depth] and target in after[:depth]
        losses += target in before[:depth] and target not in after[:depth]
    return {'gained_queries': wins, 'lost_queries': losses,
            'net_queries': wins-losses, 'absolute_delta': (wins-losses)/len(rows)}


def validate_cache(cache):
    cases, manifest = load_cases('dev')
    meta, rows = cache['metadata'], cache['rows']
    if (meta['split'] != 'dev' or meta['dev_sha256'] != manifest['files']['dev.jsonl']['sha256']
            or meta['query_fingerprint'] != query_fingerprint(cases)
            or meta['corpus_fingerprint'] != Corpus(CORPUS_DIR).fingerprint
            or len(rows) != len(cases)):
        raise ValueError('Score cache is not the verified dev corpus/split')
    names = set(rows[0]['streams'])
    if not 1 <= len(names) <= 3 or names - {'dense', 'bm25', 'bge_sparse'}:
        raise ValueError('Unexpected retrieval streams')
    corpus_ids = {r.id for r in Corpus(CORPUS_DIR).records}
    if len({row['gold_record_id'] for row in rows}) != len(rows):
        raise ValueError('This partition protocol requires unique gold documents')
    for case, row in zip(cases, rows):
        if ((row['id'], row['gold_record_id']) != (case['id'], case['gold_record_id'])
                or set(row['streams']) != names):
            raise ValueError('Case identities or retrieval legs do not match')
        for stream in row['streams'].values():
            if any(key not in corpus_ids for key, _ in stream):
                raise ValueError('Unknown document in retrieval cache')
        # Also checks duplicates and non-finite numeric scores.
        fuse_scores(row['streams'], weights=dict.fromkeys(names, 1.0))
    return rows, sorted(names)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--diagnostics', type=Path, required=True)
    args = parser.parse_args()
    raw = args.cache.read_bytes()
    cache = json.loads(gzip.decompress(raw) if args.cache.suffix == '.gz' else raw)
    rows, names = validate_cache(cache)
    # Split by gold document identity, fixed independently of retrieval outcomes.
    ordered = sorted(rows, key=lambda row: hashlib.sha256(
        (SEED+row['gold_record_id']).encode()).hexdigest())
    tuning, validation = ordered[:len(rows)//2], ordered[len(rows)//2:]
    baseline_weights = {name: (4.0 if name == 'dense' else 1.0 if name == 'bm25' else 0.0)
                        for name in names}
    if not any(baseline_weights.values()):
        baseline_weights = dict.fromkeys(names, 1.0)
    configurations = [('rrf_4_1_baseline', 'rrf', baseline_weights)]
    for name in names:
        configurations.append((name+'_only', 'rrf', {n: float(n == name) for n in names}))
    searches = {}
    for method in ('rrf', 'minmax'):
        trials = []
        for weights in weight_grid(names):
            measured = metrics(tuning, [rank(row, method, weights) for row in tuning])
            trials.append({'weights': weights, 'tuning_metrics': measured})
        # Select only from tuning half: reranker-pool recall first, MRR@10 second.
        best = max(trials, key=lambda trial: (trial['tuning_metrics']['candidate_coverage_at_20'],
                                              trial['tuning_metrics']['mrr_at_10']))
        configurations.append(('tuned_'+method, method, best['weights']))
        searches[method] = {'configuration_count': len(trials), **best}
    results = {}
    for label, method, weights in configurations:
        results[label] = {'method': method, 'weights': weights, **{
            partition: metrics(selected, [rank(row, method, weights) for row in selected])
            for partition, selected in [('tuning', tuning), ('validation', validation), ('all_dev', rows)]}}
    base = [rank(row, 'rrf', baseline_weights) for row in validation]
    paired = {}
    for label, method, weights in configurations:
        candidate = [rank(row, method, weights) for row in validation]
        paired[label] = {f'at_{depth}': paired_recall(validation, base, candidate, depth)
                         for depth in (10, 20, 40)}
    coverage = {partition: {f'{name}_top_{cache["metadata"]["candidate_k"]}': sum(
        row['gold_record_id'] in dict(row['streams'][name]) for row in selected)/len(selected)
                            for name in names}
                for partition, selected in [('tuning', tuning), ('validation', validation), ('all_dev', rows)]}
    for partition, selected in [('tuning', tuning), ('validation', validation), ('all_dev', rows)]:
        coverage[partition]['union'] = sum(any(row['gold_record_id'] in dict(stream)
            for stream in row['streams'].values()) for row in selected)/len(selected)
    reservation = {}
    if 'bm25' in names:
        for partition, selected in [('tuning', tuning), ('validation', validation), ('all_dev', rows)]:
            rankings = [rank(row, 'rrf', baseline_weights) for row in selected]
            pools = [reserve_lexical_candidates(values, [key for key, _ in row['streams']['bm25']])
                     for row, values in zip(selected, rankings)]
            hits = lambda selected_pools: sum(row['gold_record_id'] in pool
                for row, pool in zip(selected, selected_pools))/len(selected)
            reservation[partition] = {
                'protected_rrf20_plus_missed_bm25_top3': hits(pools),
                'rrf20': hits([values[:20] for values in rankings]),
                'rrf23': hits([values[:23] for values in rankings]),
                'rrf_matched_per_query_pool_size': hits([values[:len(pool)] for values, pool in zip(rankings, pools)]),
                'mean_pool_size': sum(map(len, pools))/len(pools), 'maximum_pool_size': max(map(len, pools)),
                'extra_gold_queries_vs_rrf20': sum(row['gold_record_id'] in pool and row['gold_record_id'] not in values[:20]
                    for row, pool, values in zip(selected, pools, rankings))}
    args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
    tune_ids = {row['id'] for row in tuning}
    with args.diagnostics.open('w', encoding='utf-8') as handle:
        for row in rows:
            target = row['gold_record_id']
            rankings = {name: [key for key, _ in stream] for name, stream in row['streams'].items()}
            rankings.update({label: rank(row, method, weights) for label, method, weights in configurations})
            ranks = {label: values.index(target)+1 if target in values else None
                     for label, values in rankings.items()}
            handle.write(json.dumps({'id': row['id'], 'gold_record_id': target,
                                     'partition': 'tuning' if row['id'] in tune_ids else 'validation',
                                     'gold_ranks': ranks}, ensure_ascii=False)+'\n')
    report = {'evaluation': 'CareGuide dev score-fusion replay', 'split': 'dev',
              'frozen_test_run': False, 'case_count': len(rows), 'tuning_count': len(tuning),
              'validation_count': len(validation), 'partition_seed': SEED,
              'selection_objective': ['candidate_coverage_at_20', 'mrr_at_10'],
              'retrieval_legs': names, 'source_metadata': cache['metadata'],
              'cache_sha256': hashlib.sha256(args.cache.read_bytes()).hexdigest(),
              'searches': searches, 'results': results, 'candidate_recall': coverage,
              'paired_validation_vs_baseline': paired,
              'exploratory_lexical_reservation': {
                  'policy': 'Protect RRF Top20 and append BM25 Top3 entries missing from that pool; at most 23 candidates.',
                  'status': 'Additional dev exploration prompted by per-case diagnostics; not an independent validation claim. No reranker run.',
                  'partitions': reservation},
              'scope': 'Pre-rerank retrieval only. Dev has been used historically; internal validation is not a new frozen test. No answer-quality or end-to-end latency claim.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, report)
    print(json.dumps({'results': {k: {'weights': v['weights'], 'validation': v['validation']}
                                 for k, v in results.items()}, 'candidate_recall': coverage}, indent=2))


if __name__ == '__main__':
    main()
