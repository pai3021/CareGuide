"""Trace genuine dev failures through recall, truncation, and reranking.

Labels describe observed pipeline stages, not unverified semantic causes. The
gold ID is only used offline for evaluation, never as a retrieval feature.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from careguide.corpus import Corpus
from careguide.evaluation.kuake_ir_dense import ranking_metrics
from scripts.cache_rag_recall_scores import atomic_json
from scripts.evaluate_rag_candidate_pools import BASELINE, partition_rows, rank
from scripts.evaluate_rag_score_fusion import validate_cache
from scripts.run_careguide_kuake_rag_evaluation import CORPUS_DIR, load_cases


def read_json(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


def position(values, target):
    return values.index(target)+1 if target in values else None


def trace(row, pool, final):
    target = row['gold_record_id']
    union = {key for stream in row['streams'].values() for key, _ in stream}
    if target not in union:
        stage = 'absent_from_retrieval_union'
    elif target not in pool:
        stage = 'removed_at_pool_cutoff'
    elif target not in final:
        stage = 'inside_pool_outside_final10'
    else:
        stage = 'final10_hit'
    return {'stage': stage, 'pool_rank': position(pool, target),
            'final_rank': position(final, target),
            'leg_ranks': {name: position([key for key, _ in values], target)
                          for name, values in row['streams'].items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--responses', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases-output', type=Path, required=True)
    args = parser.parse_args()
    rows, names = validate_cache(read_json(args.cache))
    if set(names) != {'dense', 'bm25'}:
        parser.error('Requires the genuine two-route score cache')
    by_id = {row['id']: row for row in rows}
    plan, responses = read_json(args.plan), read_json(args.responses)
    if responses['identity']['plan_sha256'] != hashlib.sha256(args.plan.read_bytes()).hexdigest():
        parser.error('Response/plan identity mismatch')
    if plan['cache_sha256'] != hashlib.sha256(args.cache.read_bytes()).hexdigest():
        parser.error('Plan/score cache identity mismatch')
    corpus = {record.id: record for record in Corpus(CORPUS_DIR).records}
    cases, _ = load_cases('dev')
    queries = {case['id']: case['query'] for case in cases}
    stage_counts, metrics, records = {}, {}, []
    for arm in plan['arms']:
        actual = [value for value in responses['completed'] if value['arm'] == arm]
        if {r['case_id'] for r in actual} != set(plan['case_ids']) or len(actual) != len(plan['case_ids']):
            raise ValueError('Missing or duplicate actual responses')
        expected = {job['case_id']: job['candidate_ids'] for job in plan['jobs'] if job['arm'] == arm}
        counts = dict.fromkeys(('absent_from_retrieval_union', 'removed_at_pool_cutoff',
                                'inside_pool_outside_final10', 'final10_hit'), 0)
        for value in actual:
            if value['candidate_ids'] != expected[value['case_id']]:
                raise ValueError('Actual candidate list differs from the fixed plan')
            row = by_id[value['case_id']]
            traced = trace(row, value['candidate_ids'], value['ids'])
            counts[traced['stage']] += 1
            records.append({
                'id': value['case_id'], 'arm': arm, 'query': queries[value['case_id']],
                'gold_record_id': row['gold_record_id'], 'trace': traced,
                'gold_content': corpus[row['gold_record_id']].content,
                'top3': [{'record_id': key, 'score': score, 'content': corpus[key].content}
                         for key, score in zip(value['ids'][:3], value['scores'][:3])],
                'semantic_cause': 'not adjudicated; requires reading the query and competing passages',
            })
        gold = [by_id[v['case_id']]['gold_record_id'] for v in actual]
        before = [v['candidate_ids'][:10] for v in actual]
        after = [v['ids'] for v in actual]
        metrics[arm] = {'before': ranking_metrics(before, gold, cutoff=10),
                        'after': ranking_metrics(after, gold, cutoff=10),
                        'gained': sum(t not in b and t in a for t, b, a in zip(gold, before, after)),
                        'lost': sum(t in b and t not in a for t, b, a in zip(gold, before, after))}
        stage_counts[arm] = counts
    tuning, validation = partition_rows(rows)
    coverage = {}
    pilot_ids = set(plan['case_ids'])
    for partition, selected in [('tuning500', tuning), ('validation500', validation),
                                 ('pilot100', [r for r in rows if r['id'] in pilot_ids])]:
        ranked = [rank(row, BASELINE) for row in selected]
        coverage[partition] = {f'pool_{depth}': sum(row['gold_record_id'] in values[:depth]
                                                   for row, values in zip(selected, ranked))/len(selected)
                               for depth in (20, 40, 60, 100)}
        coverage[partition]['union_top100_each'] = sum(any(row['gold_record_id'] in dict(stream)
            for stream in row['streams'].values()) for row in selected)/len(selected)
    report = {'evaluation': 'dev RAG stage diagnostic', 'frozen_test_run': False,
              'external_requests_this_run': 0, 'stage_counts': stage_counts,
              'rerank_before_after': metrics, 'candidate_upper_bounds': coverage,
              'source_sha256': {name: hashlib.sha256(path.read_bytes()).hexdigest()
                                for name, path in [('scores', args.cache), ('plan', args.plan),
                                                   ('responses', args.responses)]},
              'scope': 'Observed loss stages only. No automatic semantic-cause or clinical adjudication. Pilot is reused development data.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.cases_output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, report)
    args.cases_output.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
