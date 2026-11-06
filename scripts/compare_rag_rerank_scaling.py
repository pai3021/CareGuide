"""Bounded two-stage reranking experiment using an existing 4B dense cache.

Prepare/replay are offline. Real inference requires an explicit SiliconFlow
configuration, remote flag, and 205-attempt cap. Tuning50 compares three arms;
only the tuning-selected arm is requested on validation50 (200 normal calls).
No embeddings, generative LLM, hospital tools, or frozen test are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

from careguide.corpus import Corpus
from careguide.evaluation.kuake_ir_dense import ranking_metrics
from careguide.retrieval.reranker import OpenAICompatibleReranker
from scripts.analyze_rag_failures import read_json
from scripts.cache_rag_recall_scores import atomic_json
from scripts.compare_rag_embeddings import require_endpoint
from scripts.compare_rag_rerank_models import ENGLISH_INSTRUCTION
from scripts.evaluate_rag_candidate_pools import BASELINE, SAMPLE_SEED, partition_rows, rank
from scripts.evaluate_rag_rank_retention import SEED
from scripts.evaluate_rag_score_fusion import validate_cache
from scripts.private_config import load_config
from scripts.rerank_rag_candidate_plan import AttemptLedger, paired_interval, sha
from scripts.run_careguide_kuake_rag_evaluation import CORPUS_DIR, load_cases

RETRIEVAL_INSTRUCTION = ('Given a Chinese medical search query, retrieve relevant passages addressing the same medical topic. '
    'Consider partial answers and follow-up replies as relevant when they address the query; do not require a complete standalone answer.')
ARMS = {
    '4b80_current': {'model': 'Qwen/Qwen3-Reranker-4B', 'instruction': ENGLISH_INSTRUCTION},
    '4b80_retrieval': {'model': 'Qwen/Qwen3-Reranker-4B', 'instruction': RETRIEVAL_INSTRUCTION},
    '8b80_retrieval': {'model': 'Qwen/Qwen3-Reranker-8B', 'instruction': RETRIEVAL_INSTRUCTION},
}


def partition(keys):
    ordered = sorted(keys, key=lambda key: hashlib.sha256((SEED+key).encode()).hexdigest())
    return ordered[:50], ordered[50:]


def score(identities, responses, rows):
    return ranking_metrics([responses[key]['ids'][:10] for key in identities],
                           [rows[key]['gold_record_id'] for key in identities], cutoff=10)


def select_arm(tuning, completed, rows):
    trials = {}
    for arm in ARMS:
        chosen = {r['case_id']: r for r in completed if r['arm'] == arm}
        if not set(tuning) <= set(chosen):
            raise ValueError('Every tuning arm must finish before selection')
        trials[arm] = score(tuning, chosen, rows)
    # Ties prefer 4B and the original instruction, without looking at validation.
    return max(ARMS, key=lambda a: (trials[a]['recall_at_10'], trials[a]['mrr_at_10'],
                                  a == '4b80_current', a == '4b80_retrieval')), trials


def validate_response(row, pool):
    if (row['candidate_ids'] != pool or len(row['ids']) != len(pool)
            or set(row['ids']) != set(pool) or len(set(row['ids'])) != len(pool)
            or len(row['scores']) != len(pool) or any(not math.isfinite(s) for s in row['scores'])):
        raise ValueError('Actual response must rank every candidate exactly once')


def validate(plan, cache):
    rows, names = validate_cache(cache)
    if (set(names) != {'bm25', 'dense'} or cache['metadata']['embedding_model'] != 'Qwen/Qwen3-Embedding-4B'
            or cache['metadata']['embedding_dimensions'] != 1024 or plan['arms'] != ARMS
            or plan['version'] != 'careguide-rerank-scaling-v1' or plan['fusion'] != BASELINE
            or plan['pool_size'] != 80 or plan['max_attempts'] != 205):
        raise ValueError('Experiment contract differs')
    _, validation = partition_rows(rows)
    expected = sorted(validation, key=lambda row: hashlib.sha256((SAMPLE_SEED+row['id']).encode()).hexdigest())[:100]
    keys = [r['id'] for r in expected]
    if plan['case_ids'] != keys or (plan['tuning_ids'], plan['validation_ids']) != partition(keys):
        raise ValueError('Fixed sample or partition changed')
    by_id = {r['id']: r for r in rows}
    if set(plan['pools']) != set(keys) or set(plan['baseline_4b40']) != set(keys):
        raise ValueError('Missing/foreign planned queries')
    for key in keys:
        if plan['pools'][key] != rank(by_id[key], BASELINE)[:80]:
            raise ValueError('Candidate pool changed')
        validate_response(plan['baseline_4b40'][key], rank(by_id[key], BASELINE)[:40])
    return by_id


def prepare(args):
    if args.plan.exists():
        raise ValueError('Use a new plan path')
    source, cache, saved = map(read_json, (args.source_plan, args.scores, args.source_responses))
    if saved['identity'] != {'plan_sha256': sha(args.source_plan), 'model': 'Qwen/Qwen3-Reranker-4B',
                            'instruction': ENGLISH_INSTRUCTION, 'endpoint': 'https://api.siliconflow.cn/v1/rerank'}:
        raise ValueError('Actual 4B Top40 baseline identity differs')
    actual = {r['case_id']: r for r in saved['completed']}
    if len(actual) != 100 or len(saved['completed']) != 100:
        raise ValueError('Expected complete baseline')
    by_id = {r['id']: r for r in cache['rows']}
    tune, valid = partition(source['case_ids'])
    plan = {'version': 'careguide-rerank-scaling-v1', 'arms': ARMS, 'fusion': BASELINE, 'pool_size': 80,
            'max_attempts': 205, 'normal_requests': 200, 'case_ids': source['case_ids'],
            'tuning_ids': tune, 'validation_ids': valid,
            'baseline_4b40': actual, 'pools': {key: rank(by_id[key], BASELINE)[:80] for key in source['case_ids']},
            'source_scores_sha256': sha(args.scores), 'source_plan_sha256': sha(args.source_plan),
            'source_responses_sha256': sha(args.source_responses),
            'selection': ['tuning50 Recall@10', 'tuning50 MRR@10', 'prefer 4B and original instruction on ties'],
            'scope': 'Reused dev. No independent-test claim. Normal 150 tuning requests +50 selected validation; 5 failure slots, no automatic retry.'}
    rows = validate(plan, cache)
    records = {r.id: r for r in Corpus(CORPUS_DIR).records}
    queries = {r['id']: r['query'] for r in load_cases('dev')[0]}
    plan['payload'] = {'unique_queries': 100, 'max_documents_per_request': 80,
        'normal_document_characters': sum((3 if key in tune else 1)*sum(len(records[doc].content) for doc in plan['pools'][key]) for key in source['case_ids']),
        'normal_query_characters': sum((3 if key in tune else 1)*len(queries[key]) for key in source['case_ids'])}
    plan['payload']['maximum_document_characters'] = plan['payload']['normal_document_characters']+5*max(
        sum(len(records[doc].content) for doc in plan['pools'][key]) for key in source['case_ids'])
    plan['candidate_coverage'] = {name: sum(rows[key]['gold_record_id'] in plan['pools'][key] for key in keys)/len(keys)
                                   for name, keys in [('tuning50', tune), ('validation50', valid), ('all100', source['case_ids'])]}
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.plan, plan)
    print(json.dumps({key: plan[key] for key in ('normal_requests', 'max_attempts', 'payload', 'candidate_coverage', 'scope')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run', 'replay'))
    for name in ('plan', 'scores', 'source-plan', 'source-responses', 'responses', 'ledger', 'output', 'config-file'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--allow-remote', action='store_true')
    parser.add_argument('--max-requests', type=int)
    args = parser.parse_args()
    if args.mode == 'prepare':
        if not all((args.plan, args.scores, args.source_plan, args.source_responses)):
            parser.error('Prepare requires explicit verified sources')
        prepare(args)
        return
    if not all((args.plan, args.scores, args.responses, args.output)):
        parser.error('Plan/scores/responses/output required')
    plan, cache = read_json(args.plan), read_json(args.scores)
    if sha(args.scores) != plan['source_scores_sha256']:
        raise ValueError('Source score checksum changed')
    rows = validate(plan, cache)
    identity = {'plan_sha256': sha(args.plan), 'endpoint': 'https://api.siliconflow.cn/v1/rerank'}
    saved = read_json(args.responses) if args.responses.exists() else {'identity': identity, 'completed': []}
    if saved['identity'] != identity:
        raise ValueError('Checkpoint belongs to another plan')
    completed = saved['completed']
    seen = set()
    for row in completed:
        key = (row['case_id'], row['arm'])
        if key in seen or row['case_id'] not in plan['case_ids'] or row['arm'] not in ARMS:
            raise ValueError('Duplicate or foreign response')
        validate_response(row, plan['pools'][row['case_id']])
        seen.add(key)
    remote = args.mode == 'run' and args.allow_remote and args.max_requests == 205 and args.config_file and args.ledger
    if args.mode == 'run' and not remote:
        print(json.dumps({'mode': 'offline dry-run', 'normal_requests': 200, 'max_attempts': 205, 'payload': plan['payload']}))
        return
    clients = {}
    if remote:
        load_config(args.config_file)
        base = os.environ.get('CAREGUIDE_RERANK_BASE_URL') or os.environ.get('CAREGUIDE_EMBEDDING_BASE_URL', '')
        require_endpoint(base)
        args.ledger.parent.mkdir(parents=True, exist_ok=True)
        args.responses.parent.mkdir(parents=True, exist_ok=True)
        ledger = AttemptLedger(args.ledger, identity, 205)
        queries = {r['id']: r['query'] for r in load_cases('dev')[0]}
        records = {r.id: r for r in Corpus(CORPUS_DIR).records}

    def ensure(keys, arms):
        for key in keys:
            ordered = list(arms)
            offset = int(hashlib.sha256(key.encode()).hexdigest(), 16) % len(ordered)
            for arm in ordered[offset:]+ordered[:offset]:
                if (key, arm) in seen:
                    continue
                if not remote:
                    raise ValueError('Missing actual responses; remote execution requires explicit authorization')
                if arm not in clients:
                    clients[arm] = OpenAICompatibleReranker(api_key=os.environ.get('CAREGUIDE_RERANK_API_KEY') or os.environ.get('CAREGUIDE_EMBEDDING_API_KEY', ''),
                        base_url=base, model=ARMS[arm]['model'], instruction=ARMS[arm]['instruction'], timeout=90, max_retries=0)
                pool = plan['pools'][key]
                ledger.reserve({'case_id': key, 'arm': arm})
                started = time.perf_counter()
                try:
                    ids, scores, meta = clients[arm].rerank(queries[key], [(doc, records[doc].content) for doc in pool], top_n=len(pool))
                except Exception as error:
                    raise RuntimeError(f'Provider request failed ({type(error).__name__}); attempt counted, no automatic retry') from None
                row = {'case_id': key, 'arm': arm, 'candidate_ids': pool, 'ids': ids, 'scores': scores,
                       'seconds': time.perf_counter()-started, 'input_tokens': (meta.get('tokens') or {}).get('input_tokens')}
                validate_response(row, pool)
                completed.append(row)
                seen.add((key, arm))
                atomic_json(args.responses, saved)
                if len(completed) % 5 == 0:
                    print(json.dumps({'completed': len(completed), 'attempts': len(ledger.data['attempts'])}), flush=True)

    ensure(plan['tuning_ids'], ARMS)
    selected, trials = select_arm(plan['tuning_ids'], completed, rows)
    print(json.dumps({'selected_from_tuning': selected, 'tuning_trials': trials}), flush=True)
    allowed = {(key, arm) for key in plan['tuning_ids'] for arm in ARMS} | {(key, selected) for key in plan['validation_ids']}
    if not seen <= allowed:
        raise ValueError('Validation responses exist for unselected arms')
    ensure(plan['validation_ids'], [selected])
    chosen = {r['case_id']: r for r in completed if r['arm'] == selected}
    report = {'evaluation': 'pool, retrieval instruction and 8B reranker staged pilot', 'selected_arm': selected,
              'tuning_trials': trials, 'source_plan_sha256': sha(args.plan), 'frozen_test_run': False,
              'sample_scope': plan['scope'], 'results': {}}
    for name, keys in [('tuning50', plan['tuning_ids']), ('validation50', plan['validation_ids']), ('all100', plan['case_ids'])]:
        before, after = score(keys, plan['baseline_4b40'], rows), score(keys, chosen, rows)
        changes = [int(rows[key]['gold_record_id'] in chosen[key]['ids'][:10])-int(rows[key]['gold_record_id'] in plan['baseline_4b40'][key]['ids'][:10]) for key in keys]
        report['results'][name] = {'baseline': before, 'selected': after, 'paired': {**paired_interval(changes), 'gained': changes.count(1), 'lost': changes.count(-1)}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
