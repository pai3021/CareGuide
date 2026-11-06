"""Offline, development-only fusion of original and complete reranker ranks.

Uses full actual provider rankings, never assigns missing reranker scores. The
50/50 pilot split and five weights are fixed before inspecting those outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from careguide.evaluation.kuake_ir_dense import ranking_metrics
from careguide.retrieval.score_fusion import fuse_scores
from scripts.analyze_rag_failures import read_json
from scripts.cache_rag_recall_scores import atomic_json
from scripts.compare_rag_rerank_models import ARMS, summarize_models, validate
from scripts.rerank_rag_candidate_plan import paired_interval, sha

SEED = 'careguide-rerank-rank-retention-dev-v1'
WEIGHTS = (0.0, .25, .5, .75, 1.0)


def retain_ranking(original, reranked, weight):
    if (len(original) != len(set(original)) or len(reranked) != len(set(reranked))
            or set(original) != set(reranked)):
        raise ValueError('Rank retention needs two complete rankings over the same pool')
    if not 0 <= weight <= 1:
        raise ValueError('Reranker weight must be between zero and one')
    return [key for key, _ in fuse_scores(
        {'retrieval': [(key,0.) for key in original], 'rerank': [(key,0.) for key in reranked]},
        weights={'retrieval':1-weight, 'rerank':weight}, method='rrf',constant=60)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--responses',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    plan,cache,saved = read_json(args.plan),read_json(args.scores),read_json(args.responses)
    if (plan['cache_sha256'] != sha(args.scores)
            or saved['identity']['plan_sha256'] != sha(args.plan)):
        raise ValueError('Rank-retention cache identities mismatch')
    rows = validate(plan,cache)
    summarize_models(plan,rows,saved['completed'])
    identities = sorted(plan['case_ids'],key=lambda key:hashlib.sha256((SEED+key).encode()).hexdigest())
    parts = {'tuning':identities[:50], 'validation':identities[50:], 'all_pilot':identities}
    by_key = {(r['case_id'],r['arm']):r for r in saved['completed']}
    results = {}
    for arm,config in ARMS.items():
        if config['cached']:
            continue  # Old baseline has only Top10 scores, not a complete ranking.
        def metrics(keys,weight):
            rankings = [retain_ranking(by_key[(key,arm)]['candidate_ids'],by_key[(key,arm)]['ids'],weight)[:10]
                        for key in keys]
            return ranking_metrics(rankings,[rows[key]['gold_record_id'] for key in keys],cutoff=10)
        trials = [{'weight':weight,'tuning':metrics(parts['tuning'],weight)} for weight in WEIGHTS]
        best = max(trials,key=lambda v:(v['tuning']['recall_at_10'],v['tuning']['mrr_at_10'],v['weight']))
        paired = []
        for key in parts['validation']:
            row = by_key[(key,arm)]
            before = row['ids'][:10]
            after = retain_ranking(row['candidate_ids'],row['ids'],best['weight'])[:10]
            target = rows[key]['gold_record_id']
            paired.append(int(target in after)-int(target in before))
        results[arm] = {
            'selected_weight':best['weight'], 'tuning_trials':trials,
            'reranker_only':{name:metrics(keys,1.) for name,keys in parts.items()},
            'selected':{name:metrics(keys,best['weight']) for name,keys in parts.items()},
            'paired_validation':{**paired_interval(paired),'gained':paired.count(1),'lost':paired.count(-1)},
        }
    report = {'evaluation':'offline original-rank retention after actual reranking',
              'partition_seed':SEED, 'tuning_count':50,'validation_count':50,'weights':WEIGHTS,
              'frozen_test_run':False,'external_requests':0,'results':results,
              'source_sha256':{'plan':sha(args.plan),'scores':sha(args.scores),'responses':sha(args.responses)},
              'scope':'Exploratory reused development pilot, not new independent evaluation. Full actual provider ranks required. No new model calls.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(args.output,report)
    print(json.dumps({key:{'weight':value['selected_weight'],
                          'before_validation':value['reranker_only']['validation'],
                          'after_validation':value['selected']['validation']}
                      for key,value in results.items()},ensure_ascii=False))


if __name__=='__main__':
    main()
