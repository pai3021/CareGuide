"""Prepare/replay or explicitly execute a bounded instruction/model/pool ablation.

Reuses the 100 fixed public dev queries and existing actual Chinese 0.6B Top40
baseline. Three new arms isolate instruction, model capacity, and pool expansion.
No embedding calls, generative LLM calls, hospital tools, or frozen-test runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path
from urllib.parse import urlparse

from careguide.corpus import Corpus
from careguide.evaluation.kuake_ir_dense import ranking_metrics
from careguide.retrieval.reranker import DEFAULT_RERANK_INSTRUCTION, OpenAICompatibleReranker
from scripts.analyze_rag_failures import read_json
from scripts.cache_rag_recall_scores import atomic_json
from scripts.evaluate_rag_candidate_pools import BASELINE, SAMPLE_SEED, partition_rows, rank
from scripts.evaluate_rag_score_fusion import validate_cache
from scripts.private_config import load_config
from scripts.rerank_rag_candidate_plan import AttemptLedger, paired_interval, sha, summarize, validate_plan
from scripts.run_careguide_kuake_rag_evaluation import CORPUS_DIR, load_cases

ENGLISH_INSTRUCTION = (
    "Rank the candidate passages by whether they can directly answer the Chinese medical query."
)
ARMS = {
    'cached_cn_06b40': {'model': 'Qwen/Qwen3-Reranker-0.6B', 'instruction': DEFAULT_RERANK_INSTRUCTION,
                       'pool_size': 40, 'cached': True},
    'english_06b40': {'model': 'Qwen/Qwen3-Reranker-0.6B', 'instruction': ENGLISH_INSTRUCTION,
                      'pool_size': 40, 'cached': False},
    'english_4b40': {'model': 'Qwen/Qwen3-Reranker-4B', 'instruction': ENGLISH_INSTRUCTION,
                     'pool_size': 40, 'cached': False},
    'english_4b100': {'model': 'Qwen/Qwen3-Reranker-4B', 'instruction': ENGLISH_INSTRUCTION,
                      'pool_size': 100, 'cached': False},
}


def prepare(args):
    if args.plan.exists():
        raise ValueError('Use a new plan path; refusing to overwrite')
    source_plan, source_report = read_json(args.source_plan), read_json(args.source_report)
    cache, responses = read_json(args.scores), read_json(args.source_responses)
    if (sha(args.source_report) != source_plan['report_sha256']
            or sha(args.scores) != source_plan['cache_sha256']
            or responses['identity']['plan_sha256'] != sha(args.source_plan)):
        raise ValueError('Original plan/report/cache identities mismatch')
    rows = validate_plan(source_plan, source_report, cache)
    summarize(source_plan, rows, responses['completed'])  # Require the full genuine baseline cache.
    sampled = source_plan['case_ids']
    jobs = []
    for key in sampled:
        ranking = rank(rows[key], BASELINE)
        labels = [name for name, arm in ARMS.items() if not arm['cached']]
        offset = int(hashlib.sha256(key.encode()).hexdigest(), 16) % len(labels)
        for label in labels[offset:]+labels[:offset]:
            jobs.append({'case_id': key, 'arm': label, 'candidate_ids': ranking[:ARMS[label]['pool_size']]})
    baseline = [{**value, 'arm': 'cached_cn_06b40'} for value in responses['completed']
                if value['arm'] == 'baseline_rrf40']
    cases, _ = load_cases('dev')
    queries = {case['id']: case['query'] for case in cases}
    contents = {record.id: record.content for record in Corpus(CORPUS_DIR).records}
    plan = {
        'version': 'careguide-reranker-ablation-dev-v1', 'sample_seed': SAMPLE_SEED,
        'case_ids': sampled, 'arms': ARMS, 'jobs': jobs, 'baseline': baseline,
        'cache_sha256': sha(args.scores), 'corpus_fingerprint': cache['metadata']['corpus_fingerprint'],
        'maximum_requests': len(jobs), 'final_cutoff': 10,
        'return_all_scores': True,
        'source_sha256': {name: sha(path) for name, path in [('plan', args.source_plan),
                          ('report', args.source_report), ('responses', args.source_responses)]},
        'payload': {'query_count': len(sampled), 'request_count': len(jobs),
                    'max_documents_per_request': 100,
                    'document_characters': sum(sum(len(contents[key]) for key in job['candidate_ids']) for job in jobs),
                    'query_characters': sum(len(queries[job['case_id']]) for job in jobs)},
        'scope': 'Fixed reused dev pilot, not independent test. No query filtering or gold-derived retrieval features. All candidates use original RRF 4:1.'}
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.plan, plan)
    print(json.dumps({'plan_sha256': sha(args.plan), 'arms': ARMS, 'payload': plan['payload']}, ensure_ascii=False))


def validate(plan, cache):
    rows, names = validate_cache(cache)
    if (plan['version'] != 'careguide-reranker-ablation-dev-v1' or set(names) != {'dense', 'bm25'}
            or plan['arms'] != ARMS or plan['sample_seed'] != SAMPLE_SEED
            or plan['final_cutoff'] != 10 or plan['maximum_requests'] != 300
            or plan['return_all_scores'] is not True
            or plan['corpus_fingerprint'] != cache['metadata']['corpus_fingerprint']):
        raise ValueError('Unexpected model experiment contract')
    _, validation = partition_rows(rows)
    expected = sorted(validation, key=lambda row: hashlib.sha256(
        (SAMPLE_SEED+row['id']).encode()).hexdigest())[:100]
    if plan['case_ids'] != [row['id'] for row in expected]:
        raise ValueError('Model comparison must retain the original outcome-independent 100-query sample')
    by_id = {row['id']: row for row in expected}
    seen = set()
    for job in plan['jobs']:
        key = (job['case_id'], job['arm'])
        if (key in seen or job['case_id'] not in by_id or job['arm'] not in ARMS
                or ARMS[job['arm']]['cached']):
            raise ValueError('Duplicate or foreign model job')
        seen.add(key)
        if job['candidate_ids'] != rank(by_id[job['case_id']], BASELINE)[:ARMS[job['arm']]['pool_size']]:
            raise ValueError('Model job does not match real original fusion scores')
    if len(seen) != 300:
        raise ValueError('Incomplete model comparison')
    if len(plan['baseline']) != 100 or {r['case_id'] for r in plan['baseline']} != set(by_id):
        raise ValueError('Incomplete/duplicate cached baseline')
    for value in plan['baseline']:
        if (value['arm'] != 'cached_cn_06b40' or value['case_id'] not in by_id
                or value['candidate_ids'] != rank(by_id[value['case_id']], BASELINE)[:40]
                or len(value['ids']) != 10 or len(set(value['ids'])) != 10
                or not set(value['ids']) <= set(value['candidate_ids'])
                or len(value['scores']) != 10 or any(not math.isfinite(s) for s in value['scores'])):
            raise ValueError('Invalid cached baseline')
    return by_id


def summarize_models(plan, rows, completed):
    combined = plan['baseline'] + completed
    by_key = {(r['case_id'], r['arm']): r for r in combined}
    if len(combined) != 400 or len(by_key) != 400:
        raise ValueError('Incomplete/duplicate paired model run')
    results = {}
    for arm, config in ARMS.items():
        values = [by_key[(key, arm)] for key in plan['case_ids']]
        for key, value in zip(plan['case_ids'], values):
            expected = rank(rows[key], BASELINE)[:config['pool_size']]
            count = 10 if config['cached'] else len(expected)
            if (value['candidate_ids'] != expected or len(value['ids']) != count
                    or len(set(value['ids'])) != count or not set(value['ids']) <= set(expected)
                    or len(value['scores']) != count or any(not math.isfinite(s) for s in value['scores'])):
                raise ValueError('Invalid actual model response')
        gold = [rows[key]['gold_record_id'] for key in plan['case_ids']]
        results[arm] = {
            'metrics': ranking_metrics([value['ids'][:10] for value in values], gold, cutoff=10),
            'candidate_coverage': sum(target in value['candidate_ids'] for target, value in zip(gold, values))/100,
            'api_seconds_mean': statistics.mean(value['seconds'] for value in values),
            'input_tokens': (sum(value['input_tokens'] for value in values)
                             if all(value['input_tokens'] is not None for value in values) else None),
            'reused_baseline': config['cached'],
        }
    pairs = {}
    for before, after in [('cached_cn_06b40', 'english_06b40'), ('english_06b40', 'english_4b40'),
                           ('cached_cn_06b40', 'english_4b40'),
                           ('english_4b40', 'english_4b100'), ('cached_cn_06b40', 'english_4b100')]:
        changes = [int(rows[key]['gold_record_id'] in by_key[(key, after)]['ids'][:10])
                   - int(rows[key]['gold_record_id'] in by_key[(key, before)]['ids'][:10]) for key in plan['case_ids']]
        pairs[before+'__to__'+after] = {**paired_interval(changes),
                                        'gained': changes.count(1), 'lost': changes.count(-1)}
    score_errors = []
    same_order = 0
    for key in plan['case_ids']:
        narrow,broad = by_key[(key,'english_4b40')],by_key[(key,'english_4b100')]
        ids = set(narrow['ids'])
        same_order += [identity for identity in broad['ids'] if identity in ids] == narrow['ids']
        a,b = dict(zip(narrow['ids'],narrow['scores'])),dict(zip(broad['ids'],broad['scores']))
        score_errors.extend(abs(a[identity]-b[identity]) for identity in ids)
    return {'evaluation': 'fixed dev instruction/model/pool comparison', 'case_count': 100,
            'results': results, 'paired': pairs, 'arms': ARMS,
            'cross_pool_score_diagnostic': {'shared_query_document_pairs':len(score_errors),
                'same_relative_order_queries':same_order,
                'maximum_score_difference':max(score_errors),
                'mean_score_difference':statistics.mean(score_errors),
                'interpretation':'Changing candidate sets was accompanied by score variation; do not assume exact per-pair score invariance or infer one pool result from another.'},
            'frozen_test_run': False,
            'scope': 'Exploratory reused dev pilot. No clinical-quality or production-latency claim.'}


def execute(args):
    plan, cache = read_json(args.plan), read_json(args.scores)
    if sha(args.scores) != plan['cache_sha256']:
        raise ValueError('Score cache checksum mismatch')
    rows = validate(plan, cache)
    print(json.dumps({'mode': 'remote opt-in' if args.allow_remote_rerank else 'offline',
                      'plan_sha256': sha(args.plan), 'payload': plan['payload']}, ensure_ascii=False), flush=True)
    if not args.allow_remote_rerank:
        if args.responses:
            saved = read_json(args.responses)
            if saved['identity']['plan_sha256'] != sha(args.plan):
                raise ValueError('Responses belong to another model plan')
            result = summarize_models(plan, rows, saved['completed'])
            if args.output:
                atomic_json(args.output, result)
            print(json.dumps(result, ensure_ascii=False))
        return
    if args.responses or not args.config_file or not args.output_dir or args.max_api_requests != 300:
        raise ValueError('Remote run requires explicit config/output directory and exact cap 300')
    load_config(args.config_file)
    base = os.environ.get('CAREGUIDE_RERANK_BASE_URL') or os.environ.get('CAREGUIDE_EMBEDDING_BASE_URL', '')
    endpoint = urlparse(base.rstrip('/'))
    if ((endpoint.scheme, endpoint.hostname, endpoint.path, endpoint.port) != ('https','api.siliconflow.cn','/v1',None)
            or endpoint.username is not None or endpoint.password is not None or endpoint.query or endpoint.fragment):
        raise ValueError('Only scoped SiliconFlow rerank endpoint allowed')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    identity = {'plan_sha256': sha(args.plan), 'endpoint': 'https://api.siliconflow.cn/v1/rerank', 'arms': ARMS}
    checkpoint = args.output_dir/'responses.json'
    saved = read_json(checkpoint) if checkpoint.exists() else {'identity': identity, 'completed': []}
    if saved['identity'] != identity:
        raise ValueError('Checkpoint model/plan identity mismatch')
    budget = AttemptLedger(args.output_dir/'requests.json', identity, 300)
    done = {(row['case_id'], row['arm']) for row in saved['completed']}
    remaining = [job for job in plan['jobs'] if (job['case_id'],job['arm']) not in done]
    if len(budget.data['attempts']) + len(remaining) > 300:
        raise ValueError('Remaining budget cannot complete the plan; no calls sent')
    cases, _ = load_cases('dev')
    queries = {case['id']: case['query'] for case in cases}
    contents = {r.id:r.content for r in Corpus(CORPUS_DIR).records}
    adapters = {label: OpenAICompatibleReranker(api_key=os.environ.get('CAREGUIDE_RERANK_API_KEY')
        or os.environ.get('CAREGUIDE_EMBEDDING_API_KEY',''), base_url=base,
        model=arm['model'], instruction=arm['instruction'], max_retries=0, timeout=90)
                for label, arm in ARMS.items() if not arm['cached']}
    for job in remaining:
        budget.reserve(job)
        start = time.perf_counter()
        try:
            ids,scores,meta = adapters[job['arm']].rerank(queries[job['case_id']],
                [(key,contents[key]) for key in job['candidate_ids']], top_n=len(job['candidate_ids']))
            if any(not math.isfinite(score) for score in scores):
                raise ValueError('Nonfinite provider score')
        except Exception as error:
            raise RuntimeError(f'Attempt failed ({type(error).__name__}); request budget consumed') from None
        saved['completed'].append({**job, 'ids':ids,'scores':scores,'seconds':time.perf_counter()-start,
                                  'input_tokens':(int(meta['tokens']['input_tokens'])
                                                   if 'input_tokens' in (meta.get('tokens') or {}) else None)})
        atomic_json(checkpoint,saved)
        if len(saved['completed']) % 10 == 0:
            print(json.dumps({'completed':len(saved['completed']),'attempted':len(budget.data['attempts'])}),flush=True)
    result = {**summarize_models(plan,rows,saved['completed']), 'identity':identity,
              'attempted_requests':len(budget.data['attempts']), 'response_sha256':sha(checkpoint)}
    atomic_json(args.output_dir/'result.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','run'))
    parser.add_argument('--scores',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True)
    for name in ('source-plan','source-report','source-responses','config-file','output-dir','responses','output'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--allow-remote-rerank',action='store_true')
    parser.add_argument('--max-api-requests',type=int)
    args=parser.parse_args()
    if args.mode=='prepare':
        if not all((args.source_plan,args.source_report,args.source_responses)):
            parser.error('Prepare requires the verified original plan/report/responses')
        prepare(args)
    else:
        execute(args)


if __name__=='__main__':
    main()
