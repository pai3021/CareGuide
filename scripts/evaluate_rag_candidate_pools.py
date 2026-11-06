"""Offline two-route pool/fusion sweep and a deterministic paid-rerank plan.

Uses existing genuine dev scores only. Does not load credentials or call a model.
The fixed 500/500 development partition is reused, not a new held-out test.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from careguide.corpus import Corpus
from careguide.evaluation.kuake_ir_dense import ranking_metrics
from careguide.retrieval.score_fusion import fuse_scores
from scripts.cache_rag_recall_scores import atomic_json
from scripts.evaluate_rag_score_fusion import SEED, paired_recall, validate_cache, weight_grid
from scripts.run_careguide_kuake_rag_evaluation import CORPUS_DIR, load_cases

DEPTHS = (20, 30, 40, 60)
BASELINE = {'method': 'rrf', 'constant': 60, 'weights': {'dense': 4.0, 'bm25': 1.0}}
SAMPLE_SEED = 'careguide-pool-rerank-pilot-v1'


def rank(row, config):
    return [key for key, _ in fuse_scores(row['streams'], **config)]


def partition_rows(rows):
    ordered = sorted(rows, key=lambda row: hashlib.sha256(
        (SEED + row['gold_record_id']).encode()).hexdigest())
    return ordered[:len(rows)//2], ordered[len(rows)//2:]


def pool_metrics(rows, rankings, depth, contents):
    pools = [values[:depth] for values in rankings]
    gold = [row['gold_record_id'] for row in rows]
    counts = [len(pool) for pool in pools]
    return {
        **ranking_metrics(pools, gold, cutoff=10),
        'candidate_coverage': sum(target in pool for target, pool in zip(gold, pools))/len(rows),
        'mean_candidates': sum(counts)/len(counts),
        'max_candidates': max(counts),
        'mean_document_characters': sum(sum(len(contents[key]) for key in pool)
                                        for pool in pools)/len(pools),
    }


def choose_trial(trials, tuning_metrics):
    # No validation values, pool size, or model cost enter the fusion selection.
    # Stable ties prefer the original RRF, then earlier prespecified trials.
    return max(range(len(trials)), key=lambda i: (
        tuning_metrics[i]['candidate_coverage'], tuning_metrics[i]['mrr_at_10'],
        trials[i] == BASELINE, -i))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--pilot-cases', type=int, default=100)
    args = parser.parse_args()
    if args.output.exists() or args.plan.exists():
        parser.error('Use new paths; refusing to overwrite an existing experiment')
    if not 1 <= args.pilot_cases <= 500:
        parser.error('pilot-cases must be between 1 and 500')
    raw = args.cache.read_bytes()
    cache = json.loads(gzip.decompress(raw) if args.cache.suffix == '.gz' else raw)
    rows, names = validate_cache(cache)
    if set(names) != {'dense', 'bm25'} or cache['metadata']['candidate_k'] != 100:
        parser.error('Requires the existing Dense/BM25 Top100 cache')
    tuning, validation = partition_rows(rows)
    contents = {record.id: record.content for record in Corpus(CORPUS_DIR).records}
    partitions = {'tuning': tuning, 'validation': validation, 'all_dev': rows}
    # Fixed modest grid: 33 RRF + 11 Min-Max configurations, plus the exact baseline.
    trials = [BASELINE]
    for method, constant in [('rrf', 10), ('rrf', 30), ('rrf', 60), ('minmax', 60)]:
        trials.extend({'method': method, 'constant': constant, 'weights': weights}
                      for weights in weight_grid(names))
    rankings = {partition: [[rank(row, config) for row in selected] for config in trials]
                for partition, selected in partitions.items()}
    results, selected_indices = {}, {}
    for depth in DEPTHS:
        tune_metrics = [pool_metrics(tuning, values, depth, contents)
                        for values in rankings['tuning']]
        best = choose_trial(trials, tune_metrics)
        selected_indices[depth] = best
        results[str(depth)] = {
            'selected_config': trials[best],
            'baseline': {partition: pool_metrics(selected, rankings[partition][0], depth, contents)
                         for partition, selected in partitions.items()},
            'tuned': {partition: pool_metrics(selected, rankings[partition][best], depth, contents)
                      for partition, selected in partitions.items()},
            'paired_validation_tuned_vs_same_budget_baseline': paired_recall(
                validation, rankings['validation'][0], rankings['validation'][best], depth),
            'tuning_trials': [{'config': config, 'metrics': measured}
                              for config, measured in zip(trials, tune_metrics)],
        }
    # Prespecified practical cap 40, with 60 as sensitivity only. No auto-promotion
    # from pre-rerank coverage. Three arms separate expansion from fusion changes.
    arms = {
        'baseline_rrf20': {**BASELINE, 'pool_size': 20},
        'baseline_rrf40': {**BASELINE, 'pool_size': 40},
        'tuning_selected40': {**trials[selected_indices[40]], 'pool_size': 40},
    }
    sampled = sorted(validation, key=lambda row: hashlib.sha256(
        (SAMPLE_SEED + row['id']).encode()).hexdigest())[:args.pilot_cases]
    jobs = []
    for row in sampled:
        # Rotate arm order deterministically to reduce provider-time order effects.
        labels = list(arms)
        offset = int(hashlib.sha256(row['id'].encode()).hexdigest(), 16) % len(labels)
        for label in labels[offset:] + labels[:offset]:
            config = dict(arms[label])
            depth = config.pop('pool_size')
            jobs.append({'case_id': row['id'], 'arm': label,
                         'candidate_ids': rank(row, config)[:depth]})
    cases, _ = load_cases('dev')
    queries = {case['id']: case['query'] for case in cases}
    report = {
        'evaluation': 'two-route candidate-pool/fusion dev sweep',
        'split': 'dev', 'frozen_test_run': False,
        'partition_seed': SEED, 'tuning_count': len(tuning), 'validation_count': len(validation),
        'cache_sha256': hashlib.sha256(raw).hexdigest(), 'source_metadata': cache['metadata'],
        'depths': list(DEPTHS), 'configuration_count': len(trials),
        'selection_objective': ['tuning candidate coverage at each fixed pool budget', 'tuning MRR@10'],
        'results': results,
        'scope': 'Exploratory reused dev. Pre-rerank only; characters are not tokens or latency. No final-quality claim or default change.',
        'external_requests_this_run': 0,
        'next_stage': {'kind': 'paired real-reranker pilot', 'arms': arms,
                       'sample_seed': SAMPLE_SEED, 'case_count': len(sampled),
                       'maximum_requests': len(jobs), 'final_top_n': 10,
                       'selection': 'fixed hash sample of validation, independent of retrieval outcomes'},
    }
    plan = {
        'version': 'careguide-rag-pool-rerank-plan-v1',
        'cache_sha256': report['cache_sha256'],
        'corpus_fingerprint': cache['metadata']['corpus_fingerprint'],
        'dev_sha256': cache['metadata']['dev_sha256'],
        'arms': arms, 'sample_seed': SAMPLE_SEED, 'case_ids': [row['id'] for row in sampled],
        'maximum_requests': len(jobs), 'final_top_n': 10, 'jobs': jobs,
        'payload_summary': {
            'query_count': len(sampled), 'request_count': len(jobs),
            'max_documents_per_request': max(len(job['candidate_ids']) for job in jobs),
            'total_document_characters': sum(sum(len(contents[key]) for key in job['candidate_ids']) for job in jobs),
            'total_query_characters': sum(len(queries[job['case_id']]) for job in jobs),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, report)
    plan['report_sha256'] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    atomic_json(args.plan, plan)
    print(json.dumps({'pools': {key: {name: value[name]['validation']['candidate_coverage']
                                     for name in ('baseline', 'tuned')}
                              for key, value in results.items()},
                      'next_stage': report['next_stage'], 'payload': plan['payload_summary']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
