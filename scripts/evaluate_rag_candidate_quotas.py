"""Offline fixed-budget candidate quotas for an independently selected lexical leg.

Reserve lexical leaders before filling from Dense or original RRF. Select each
budget's strategy on tuning500 only; validation500 is reused dev. All inference
uses scores/ranks, never gold IDs. No API calls or frozen-test access.
"""
import argparse
import json
from pathlib import Path

from scripts.analyze_rag_failures import read_json
from scripts.cache_rag_recall_scores import atomic_json
from scripts.evaluate_rag_candidate_pools import BASELINE, partition_rows, rank
from scripts.evaluate_rag_score_fusion import validate_cache
from scripts.rerank_rag_candidate_plan import sha

DEPTHS = (40, 60, 80)
RESERVES = (0, 5, 10, 20)


def quota_pool(row, depth, reserve, fill):
    if not 0 <= reserve <= depth or fill not in {'dense', 'rrf'}:
        raise ValueError('Invalid quota')
    preferred = [key for key, _ in row['streams']['bm25'][:reserve]]
    fallback = ([key for key, _ in row['streams']['dense']] if fill == 'dense' else rank(row, BASELINE))
    selected = list(dict.fromkeys(preferred+fallback))[:depth]
    # RRF determines presentation order only, not admission to the fixed pool.
    chosen = set(selected)
    return [key for key in rank(row, BASELINE) if key in chosen]


def coverage(rows, config):
    return sum(row['gold_record_id'] in quota_pool(row, **config) for row in rows)/len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-cache', type=Path, required=True)
    parser.add_argument('--lexical-cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output path')
    base, alternate = read_json(args.baseline_cache), read_json(args.lexical_cache)
    rows, _ = validate_cache(base)
    changed, _ = validate_cache(alternate)
    if not alternate['metadata'].get('lexical_experiment'):
        raise ValueError('Requires a tuning-selected lexical experiment cache')
    if any(a['streams']['dense'] != b['streams']['dense'] for a, b in zip(rows, changed)):
        raise ValueError('Dense scores changed; cannot attribute quota/lexical effects')
    partitions = {'original_lexical': partition_rows(rows), 'selected_lexical': partition_rows(changed)}
    results = {}
    for depth in DEPTHS:
        trials = []
        for variant, (tune, _) in partitions.items():
            for fill in ('rrf', 'dense'):
                for reserve in RESERVES:
                    config = {'depth': depth, 'reserve': reserve, 'fill': fill}
                    trials.append({'variant': variant, 'config': config, 'tuning_coverage': coverage(tune, config)})
        # No validation scores enter selection; ties favor the existing strategy.
        best = max(trials, key=lambda r: (r['tuning_coverage'], r['variant'] == 'original_lexical', r['config']['reserve'] == 0, r['config']['fill'] == 'rrf'))
        tune, valid = partitions[best['variant']]
        original_valid = partitions['original_lexical'][1]
        before = [r['gold_record_id'] in rank(r, BASELINE)[:depth] for r in original_valid]
        after = [r['gold_record_id'] in quota_pool(r, **best['config']) for r in valid]
        results[str(depth)] = {'selected': best, 'tuning_trials': trials,
                              'selected_validation_coverage': sum(after)/len(after),
                              'baseline_validation_coverage': sum(before)/len(before),
                              'paired_validation': {'gained': sum(a and not b for a, b in zip(after, before)),
                                                    'lost': sum(b and not a for a, b in zip(after, before))}}
    result = {'evaluation': 'fixed candidate budget with source quotas', 'external_requests': 0,
              'frozen_test_run': False, 'tuning_count': 500, 'validation_count': 500,
              'source_sha256': {'baseline': sha(args.baseline_cache), 'lexical': sha(args.lexical_cache)},
              'results': results, 'scope': 'Reused development evidence. Pre-rerank coverage only; no final-quality or runtime claim.'}
    atomic_json(args.output, result)
    print(json.dumps({depth: {k: v for k, v in value.items() if k != 'tuning_trials'} for depth, value in results.items()}))


if __name__ == '__main__':
    main()
