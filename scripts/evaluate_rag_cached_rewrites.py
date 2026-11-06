"""Offline lexical expansion with a verified historical rewrite cache.

Never runs a rewriting or embedding model. Original query/dense scores remain
unchanged; three preset weights are selected on tuning500, then validation500.
Numeric preservation is a conservative guard, not a semantic-equivalence proof.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import unicodedata
from pathlib import Path

from careguide.corpus import Corpus
from careguide.retrieval.knowledge_base import MedicalKnowledgeBase
from careguide.retrieval.score_fusion import fuse_scores
from scripts.analyze_rag_failures import read_json
from scripts.cache_rag_recall_scores import atomic_json
from scripts.evaluate_rag_candidate_pools import BASELINE, partition_rows, rank
from scripts.evaluate_rag_lexical_variants import PostingBM25, check_baseline
from scripts.evaluate_rag_score_fusion import metrics, paired_recall, validate_cache
from scripts.rerank_rag_candidate_plan import sha
from scripts.run_careguide_kuake_rag_evaluation import CORPUS_DIR, load_cases

WEIGHTS = (0., .25, .5, 1.)
KNOWN_REWRITE_SHA256 = '1ac83efb63c8dec989841ae1fb438cd2a519311a940c627e7e0d0f917411dd62'


def numbers(text):
    return set(re.findall(r'\d+(?:\.\d+)?', unicodedata.normalize('NFKC', text)))


def guarded_rewrites(query, values):
    if not isinstance(values, list) or len(values) not in {0, 2} or any(not isinstance(v, str) for v in values):
        raise ValueError('Expected zero or two cached string rewrites')
    return list(dict.fromkeys(v.strip() for v in values if v.strip() and len(v) <= 200 and v.strip() != query
                             and numbers(v) == numbers(query)))


def expand_lexical(original, rewritten, weight):
    if weight not in WEIGHTS:
        raise ValueError('Weight outside the fixed experiment')
    if not weight or not rewritten:
        return original
    streams = {'original': original, **{f'rewrite{i}': values for i, values in enumerate(rewritten)}}
    weights = {'original': 1., **{f'rewrite{i}': weight/len(rewritten) for i in range(len(rewritten))}}
    return fuse_scores(streams, weights=weights, constant=60)[:100]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('scores', 'rewrites', 'output', 'selected-cache'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.selected_cache.exists():
        parser.error('New result paths required')
    if sha(args.rewrites) != KNOWN_REWRITE_SHA256:
        raise ValueError('Expected the already generated historical rewrite cache')
    cache = read_json(args.scores)
    rows, names = validate_cache(cache)
    if set(names) != {'dense', 'bm25'}:
        raise ValueError('Requires original two-route scores')
    cases, _ = load_cases('dev')
    by_id = {c['id']: c for c in cases}
    saved = [json.loads(s) for s in args.rewrites.read_text(encoding='utf-8').splitlines()]
    if len(saved) != len(cases) or len({r['id'] for r in saved}) != len(saved) or {r['id'] for r in saved} != set(by_id):
        raise ValueError('Duplicate, missing or foreign rewrite IDs')
    if any(r['query'] != by_id[r['id']]['query'] or r['model'] != 'mimo-v2.5' for r in saved):
        raise ValueError('Cached query/model identity mismatch')
    rewrite_texts = {r['id']: guarded_rewrites(r['query'], r['rewrites']) for r in saved}
    scorer = PostingBM25({r.id: MedicalKnowledgeBase.retrieval_text(r) for r in Corpus(CORPUS_DIR).records}, '2')
    rewritten = {}
    for row in rows:
        check_baseline(row['streams']['bm25'], scorer.search(by_id[row['id']]['query']))
        rewritten[row['id']] = [scorer.search(text) for text in rewrite_texts[row['id']]]
    results, variants = [], {}
    for weight in WEIGHTS:
        changed = [{**row, 'streams': {**row['streams'], 'bm25': expand_lexical(row['streams']['bm25'], rewritten[row['id']], weight)}} for row in rows]
        variants[weight] = changed
        tune, valid = partition_rows(changed)
        measured = {'rewrite_total_weight': weight}
        for name, part in [('tuning500', tune), ('validation500', valid)]:
            measured[name] = metrics(part, [rank(row, BASELINE) for row in part])
            measured[name]['lexical_top100'] = sum(row['gold_record_id'] in dict(row['streams']['bm25']) for row in part)/len(part)
            measured[name]['union_top100_each'] = sum(any(row['gold_record_id'] in dict(v) for v in row['streams'].values()) for row in part)/len(part)
        results.append(measured)
    chosen = max(results, key=lambda r: (r['tuning500']['candidate_coverage_at_40'], r['tuning500']['mrr_at_10'], r['rewrite_total_weight'] == 0))
    selected = variants[chosen['rewrite_total_weight']]
    before, after = partition_rows(rows)[1], partition_rows(selected)[1]
    report = {'evaluation': 'cached query rewriting for lexical retrieval', 'external_requests': 0, 'frozen_test_run': False,
              'source_sha256': {'scores': sha(args.scores), 'rewrites': sha(args.rewrites)},
              'baseline_reproduced_queries': len(rows), 'weights': WEIGHTS, 'results': results,
              'selected_weight': chosen['rewrite_total_weight'],
              'usable_rewrites': sum(map(len, rewrite_texts.values())),
              'cached_rewrites': sum(len(r['rewrites']) for r in saved),
              'paired_validation_at40': paired_recall(after, [rank(r, BASELINE) for r in before], [rank(r, BASELINE) for r in after], 40),
              'scope': 'Historical model outputs reused without API calls. Numeric preservation does not prove semantic equivalence. Reused dev, pre-rerank only.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, report)
    args.selected_cache.parent.mkdir(parents=True, exist_ok=True)
    args.selected_cache.write_bytes(gzip.compress(json.dumps({'metadata': {**cache['metadata'],
        'lexical_query_expansion': {'rewrite_weight': chosen['rewrite_total_weight'], 'rewrite_model': 'mimo-v2.5',
                                    'cache_sha256': KNOWN_REWRITE_SHA256, 'merge': 'query-local RRF, original kept'}},
        'rows': selected}, ensure_ascii=False).encode(), mtime=0))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
