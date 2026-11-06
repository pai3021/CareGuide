"""Validate/dry-run a fixed dev plan, or opt into bounded real reranker calls.

No embeddings, LLM, frozen test, or hospital tools. Each provider attempt is
persisted before sending; completed responses are checkpointed for offline replay.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import statistics
import time
from pathlib import Path
from urllib.parse import urlparse

from careguide.corpus import Corpus
from careguide.evaluation.kuake_ir_dense import ranking_metrics
from careguide.retrieval.reranker import DEFAULT_RERANK_INSTRUCTION, OpenAICompatibleReranker
from scripts.cache_rag_recall_scores import atomic_json
from scripts.evaluate_rag_candidate_pools import BASELINE, SAMPLE_SEED, partition_rows, rank
from scripts.evaluate_rag_score_fusion import validate_cache
from scripts.private_config import load_config
from scripts.run_careguide_kuake_rag_evaluation import CORPUS_DIR, load_cases


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_plan(plan, report, cache):
    rows, names = validate_cache(cache)
    if (plan['version'] != 'careguide-rag-pool-rerank-plan-v1'
            or set(names) != {'dense', 'bm25'}
            or plan['corpus_fingerprint'] != cache['metadata']['corpus_fingerprint']
            or plan['dev_sha256'] != cache['metadata']['dev_sha256']
            or plan['sample_seed'] != SAMPLE_SEED
            or plan['final_top_n'] != 10
            or plan['arms'] != report['next_stage']['arms']):
        raise ValueError('Plan identities or configuration do not match')
    _, validation = partition_rows(rows)
    selected = sorted(validation, key=lambda row: hashlib.sha256(
        (SAMPLE_SEED + row['id']).encode()).hexdigest())[:len(plan['case_ids'])]
    if not 1 <= len(selected) <= 500 or [r['id'] for r in selected] != plan['case_ids']:
        raise ValueError('Plan must use the fixed outcome-independent validation sample')
    expected_arms = {'baseline_rrf20', 'baseline_rrf40', 'tuning_selected40'}
    if (set(plan['arms']) != expected_arms
            or plan['arms']['baseline_rrf20'] != {**BASELINE, 'pool_size': 20}
            or plan['arms']['baseline_rrf40'] != {**BASELINE, 'pool_size': 40}
            or plan['arms']['tuning_selected40'] != {
                **report['results']['40']['selected_config'], 'pool_size': 40}):
        raise ValueError('Unexpected pilot arms')
    by_id = {row['id']: row for row in selected}
    seen = set()
    for job in plan['jobs']:
        key = (job['case_id'], job['arm'])
        if job['case_id'] not in by_id or job['arm'] not in expected_arms or key in seen:
            raise ValueError('Duplicate or foreign plan job')
        seen.add(key)
        config = dict(plan['arms'][job['arm']])
        depth = config.pop('pool_size')
        if job['candidate_ids'] != rank(by_id[job['case_id']], config)[:depth]:
            raise ValueError('Candidate pool does not reproduce the genuine score cache')
    if (len(seen) != len(selected)*len(expected_arms)
            or plan['maximum_requests'] != len(seen)):
        raise ValueError('Incomplete plan')
    return by_id


class AttemptLedger:
    def __init__(self, path, identity, maximum):
        if maximum < 1:
            raise ValueError('Positive explicit request budget required')
        self.path, self.maximum = path, maximum
        self.data = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {
            'identity': identity, 'maximum_requests': maximum, 'attempts': []}
        if self.data['identity'] != identity or self.data['maximum_requests'] != maximum:
            raise ValueError('Cannot reuse ledger with different plan/provider/budget')

    def reserve(self, job):
        if len(self.data['attempts']) >= self.maximum:
            raise RuntimeError('Rerank request budget exhausted')
        self.data['attempts'].append({'case_id': job['case_id'], 'arm': job['arm']})
        atomic_json(self.path, self.data)


def paired_interval(deltas):
    """Query-paired bootstrap interval for the fixed pilot, not clinical evidence."""
    rng = random.Random(20261106)
    means = sorted(sum(rng.choices(deltas, k=len(deltas)))/len(deltas) for _ in range(2000))
    return {'absolute_delta': statistics.mean(deltas),
            'bootstrap_95_interval': [means[49], means[1949]], 'bootstrap_samples': 2000}


def summarize(plan, rows, completed):
    by_key = {(r['case_id'], r['arm']): r for r in completed}
    expected = {(job['case_id'], job['arm']): job['candidate_ids'] for job in plan['jobs']}
    if len(completed) != len(expected) or set(by_key) != set(expected):
        raise ValueError('Cannot report final metrics from an incomplete run')
    for key, value in by_key.items():
        if (value['candidate_ids'] != expected[key]
                or len(value['ids']) != min(plan['final_top_n'], len(expected[key]))
                or len(set(value['ids'])) != len(value['ids'])
                or not set(value['ids']) <= set(expected[key])
                or len(value['scores']) != len(value['ids'])
                or any(not math.isfinite(score) for score in value['scores'])
                or not math.isfinite(value['seconds']) or value['seconds'] < 0
                or value['input_tokens'] is not None and value['input_tokens'] < 0):
            raise ValueError('Invalid cached rerank response')
    results = {}
    gold = [rows[key]['gold_record_id'] for key in plan['case_ids']]
    for arm in plan['arms']:
        chosen = [by_key[(key, arm)] for key in plan['case_ids']]
        rankings = [value['ids'] for value in chosen]
        results[arm] = {
            'metrics': ranking_metrics(rankings, gold, cutoff=10),
            'candidate_coverage': sum(target in value['candidate_ids']
                                      for target, value in zip(gold, chosen))/len(gold),
            'total_input_tokens': (sum(value['input_tokens'] for value in chosen)
                                   if all(value['input_tokens'] is not None for value in chosen) else None),
            'token_usage_returned_queries': sum(value['input_tokens'] is not None for value in chosen),
            'api_seconds_mean': statistics.mean(value['seconds'] for value in chosen),
            'api_seconds_observed_p95': sorted(value['seconds'] for value in chosen)[math.ceil(.95*len(chosen))-1],
        }
    base = [by_key[(key, 'baseline_rrf20')]['ids'] for key in plan['case_ids']]
    paired = {}
    for arm in plan['arms']:
        alternative = [by_key[(key, arm)]['ids'] for key in plan['case_ids']]
        deltas = [int(target in after)-int(target in before)
                  for target, before, after in zip(gold, base, alternative)]
        paired[arm] = {**paired_interval(deltas),
                       'gained_queries': deltas.count(1), 'lost_queries': deltas.count(-1)}
    transitions = {}
    for arm in plan['arms']:
        newly_covered = []
        in_pool_but_missed = 0
        outside_pool = 0
        for key, target in zip(plan['case_ids'], gold):
            before = by_key[(key, 'baseline_rrf20')]
            after = by_key[(key, arm)]
            if target in after['candidate_ids'] and target not in before['candidate_ids']:
                newly_covered.append(target in after['ids'])
            in_pool_but_missed += target in after['candidate_ids'] and target not in after['ids']
            outside_pool += target not in after['candidate_ids']
        transitions[arm] = {'newly_covered_vs_baseline_pool': len(newly_covered),
                             'newly_covered_reaching_final10': sum(newly_covered),
                             'gold_inside_pool_but_outside_final10': in_pool_but_missed,
                             'gold_outside_pool': outside_pool}
    return {'evaluation': 'fixed dev candidate-pool real-reranker pilot',
            'case_count': len(gold), 'arms': plan['arms'], 'results': results,
            'paired_recall10_vs_baseline20': paired,
            'stage_transitions': transitions, 'frozen_test_run': False,
            'scope': '100-case dev pilot by default; not independent test or clinical accuracy. API timing excludes retrieval/LLM and is not production P95.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--response-cache', type=Path, help='Readonly JSON/gzip responses for offline replay')
    parser.add_argument('--config-file', type=Path)
    parser.add_argument('--allow-remote-rerank', action='store_true')
    parser.add_argument('--max-api-requests', type=int)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    report = json.loads(args.report.read_text(encoding='utf-8'))
    raw = args.cache.read_bytes()
    cache = json.loads(gzip.decompress(raw) if args.cache.suffix == '.gz' else raw)
    if (sha(args.report) != plan['report_sha256']
            or sha(args.cache) != plan['cache_sha256']
            or report['cache_sha256'] != plan['cache_sha256']):
        parser.error('Report/cache checksum mismatch')
    rows = validate_plan(plan, report, cache)
    print(json.dumps({'mode': 'remote opt-in' if args.allow_remote_rerank else 'offline dry-run',
                      'plan_sha256': sha(args.plan), 'arms': plan['arms'],
                      'payload': plan['payload_summary']}, ensure_ascii=False), flush=True)
    if args.allow_remote_rerank and args.response_cache is not None:
        parser.error('Response cache is for offline replay only')
    if not args.allow_remote_rerank and args.output_dir is None and args.response_cache is None:
        return
    if args.allow_remote_rerank and args.output_dir is None:
        parser.error('Remote execution requires --output-dir')
    # Readonly offline replay never loads credentials or constructs an API client.
    checkpoint = args.response_cache or args.output_dir/'responses.json'
    if not args.allow_remote_rerank:
        raw_response = checkpoint.read_bytes()
        saved = json.loads(gzip.decompress(raw_response) if checkpoint.suffix == '.gz' else raw_response)
        if saved['identity']['plan_sha256'] != sha(args.plan):
            parser.error('Response cache belongs to a different plan')
        result = {**summarize(plan, rows, saved['completed']), 'identity': saved['identity'],
                  'response_cache_sha256': sha(checkpoint)}
        print(json.dumps(result, ensure_ascii=False))
        return
    if (args.max_api_requests is None or not 1 <= args.max_api_requests <= plan['maximum_requests']
            or args.config_file is None):
        parser.error('Explicit config and request cap within the fixed plan are required')
    load_config(args.config_file)
    base_url = os.environ.get('CAREGUIDE_RERANK_BASE_URL') or os.environ.get('CAREGUIDE_EMBEDDING_BASE_URL', '')
    url = urlparse(base_url.rstrip('/'))
    if ((url.scheme, url.hostname, url.path, url.port) != ('https', 'api.siliconflow.cn', '/v1', None)
            or url.query or url.fragment or url.username is not None or url.password is not None):
        parser.error('Only the explicitly scoped SiliconFlow rerank endpoint is supported')
    model = os.environ.get('CAREGUIDE_RERANK_MODEL', 'Qwen/Qwen3-Reranker-0.6B')
    if model != 'Qwen/Qwen3-Reranker-0.6B':
        parser.error('This paired plan requires the existing Qwen3-Reranker-0.6B model')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    identity = {'plan_sha256': sha(args.plan), 'model': model,
                'endpoint': 'https://api.siliconflow.cn/v1/rerank',
                'instruction': DEFAULT_RERANK_INSTRUCTION}
    saved = json.loads(checkpoint.read_text(encoding='utf-8')) if checkpoint.exists() else {
        'identity': identity, 'completed': []}
    if saved['identity'] != identity:
        parser.error('Response cache identity mismatch')
    ledger = AttemptLedger(args.output_dir/'requests.json', identity, args.max_api_requests)
    done = {(value['case_id'], value['arm']) for value in saved['completed']}
    remaining = [job for job in plan['jobs'] if (job['case_id'], job['arm']) not in done]
    if len(ledger.data['attempts']) + len(remaining) > args.max_api_requests:
        parser.error('Remaining request budget cannot complete this plan; no calls sent')
    cases, _ = load_cases('dev')
    queries = {case['id']: case['query'] for case in cases}
    contents = {record.id: record.content for record in Corpus(CORPUS_DIR).records}
    reranker = OpenAICompatibleReranker(
        api_key=os.environ.get('CAREGUIDE_RERANK_API_KEY') or os.environ.get('CAREGUIDE_EMBEDDING_API_KEY', ''),
        base_url=base_url, model=model, max_retries=0, timeout=90)
    for job in remaining:
        ledger.reserve(job)  # Includes failures/crashes; no automatic retries.
        start = time.perf_counter()
        try:
            ids, scores, meta = reranker.rerank(queries[job['case_id']],
                [(key, contents[key]) for key in job['candidate_ids']], top_n=10)
            if any(not math.isfinite(score) for score in scores):
                raise ValueError('Nonfinite provider score')
        except Exception as error:
            # Never print raw HTTP objects or private configuration.
            raise RuntimeError(f'Reranker attempt failed ({type(error).__name__}); budget already consumed') from None
        saved['completed'].append({**job, 'ids': ids, 'scores': scores,
                                   'seconds': time.perf_counter()-start,
                                   'input_tokens': (int(meta['tokens']['input_tokens'])
                                                    if 'input_tokens' in (meta.get('tokens') or {}) else None)})
        atomic_json(checkpoint, saved)
        if len(saved['completed']) % 10 == 0:
            print(json.dumps({'completed': len(saved['completed']),
                              'attempted_requests': len(ledger.data['attempts'])}), flush=True)
    result = {**summarize(plan, rows, saved['completed']), 'identity': identity,
              'attempted_requests': len(ledger.data['attempts']), 'response_cache_sha256': sha(checkpoint)}
    atomic_json(args.output_dir/'result.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
