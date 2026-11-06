"""Recompute staged rerank quality, component controls and request usage offline."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from scripts.analyze_rag_failures import read_json, trace
from scripts.cache_rag_recall_scores import atomic_json
from scripts.compare_rag_rerank_scaling import ARMS, score, select_arm, validate, validate_response
from scripts.rerank_rag_candidate_plan import paired_interval, sha


def comparison(keys, before, after, rows):
    before_score, after_score = score(keys, before, rows), score(keys, after, rows)
    changes = [int(rows[key]['gold_record_id'] in after[key]['ids'][:10])-int(rows[key]['gold_record_id'] in before[key]['ids'][:10]) for key in keys]
    return {'count': len(keys), 'before': before_score, 'after': after_score,
            'paired': {**paired_interval(changes), 'gained': changes.count(1), 'lost': changes.count(-1)},
            'relative_recall_change': after_score['recall_at_10']/before_score['recall_at_10']-1 if before_score['recall_at_10'] else None}


def usage(completed):
    seconds = sorted(row['seconds'] for row in completed)
    counts = [row['input_tokens'] for row in completed]
    if not seconds or any(not math.isfinite(s) or s < 0 for s in seconds):
        raise ValueError('Missing or invalid API timings')
    return {'response_count': len(completed), 'input_tokens': sum(counts) if all(n is not None for n in counts) else None,
            'responses_with_usage': sum(n is not None for n in counts),
            'api_seconds_mean': statistics.mean(seconds), 'api_seconds_p95': seconds[math.ceil(len(seconds)*.95)-1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'scores', 'responses', 'ledger', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    plan, cache, saved, ledger = map(read_json, (args.plan, args.scores, args.responses, args.ledger))
    if sha(args.scores) != plan['source_scores_sha256']:
        raise ValueError('Score cache hash differs')
    identity = {'plan_sha256': sha(args.plan), 'endpoint': 'https://api.siliconflow.cn/v1/rerank'}
    if saved['identity'] != identity or ledger['identity'] != identity or ledger['maximum_requests'] != 205:
        raise ValueError('Provider/plan/budget identity differs')
    rows = validate(plan, cache)
    actual = saved['completed']
    selected, trials = select_arm(plan['tuning_ids'], actual, rows)
    expected = {(key, arm) for key in plan['tuning_ids'] for arm in ARMS} | {(key, selected) for key in plan['validation_ids']}
    if len(actual) != len(expected) or {(r['case_id'], r['arm']) for r in actual} != expected:
        raise ValueError('Cannot summarize incomplete or duplicate staged responses')
    for row in actual:
        validate_response(row, plan['pools'][row['case_id']])
    attempt_counts = {}
    for row in ledger['attempts']:
        key = (row['case_id'], row['arm'])
        if key not in expected:
            raise ValueError('Attempt for an unplanned query/arm')
        attempt_counts[key] = attempt_counts.get(key, 0)+1
    if set(attempt_counts) != expected or not len(expected) <= len(ledger['attempts']) <= 205:
        raise ValueError('Request ledger does not cover actual results within budget')
    sources = {arm: {r['case_id']: r for r in actual if r['arm'] == arm} for arm in ARMS}
    baseline = plan['baseline_4b40']
    tune = plan['tuning_ids']
    controls = {
        'pool40_to80_same_4b_current_instruction': comparison(tune, baseline, sources['4b80_current'], rows),
        'current_to_retrieval_instruction_same_4b80': comparison(tune, sources['4b80_current'], sources['4b80_retrieval'], rows),
        '4b_to8b_same80_retrieval_instruction': comparison(tune, sources['4b80_retrieval'], sources['8b80_retrieval'], rows),
    }
    final = {part: comparison(keys, baseline, sources[selected], rows) for part, keys in
             [('tuning50', tune), ('validation50', plan['validation_ids']), ('all100', plan['case_ids'])]}
    stages = {}
    for name, source in [('baseline_4b40', baseline), ('selected', sources[selected])]:
        counts = {}
        for key in plan['case_ids']:
            stage = trace(rows[key], source[key]['candidate_ids'], source[key]['ids'][:10])['stage']
            counts[stage] = counts.get(stage, 0)+1
        stages[name] = counts
    report = {'evaluation': 'staged pool/instruction/model reranking control', 'frozen_test_run': False,
              'selected_arm': selected, 'selected_config': ARMS[selected], 'pool_size': 80,
              'selection_scope': 'Tuning50 only; validation50 and aggregate100 are reused development evidence.',
              'tuning_trials': trials, 'component_controls_tuning50': controls, 'selected_results': final,
              'stage_counts_all100': stages,
              'usage_by_arm': {arm: usage([r for r in actual if r['arm'] == arm]) for arm in ARMS},
              'baseline_usage_all100': usage(list(baseline.values())),
              'requests': {'maximum_attempts': 205, 'attempted': len(ledger['attempts']), 'successful': len(actual),
                           'failed_without_saved_response': len(ledger['attempts'])-len(actual),
                           'successful_input_tokens': (sum(r['input_tokens'] for r in actual) if all(r['input_tokens'] is not None for r in actual) else None),
                           'responses_with_usage': sum(r['input_tokens'] is not None for r in actual)},
              'source_sha256': {name: sha(path) for name, path in vars(args).items() if name != 'output'},
              'scope': 'Actual complete rankings over frozen pools. No query filtering, no gold model inputs. API timing is not end-to-end production P95.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
