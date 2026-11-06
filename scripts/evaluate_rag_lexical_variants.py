"""Offline Chinese n-gram/BM25 ablation, with unchanged cached dense scores.

No API client, credentials, model download, or frozen test. The baseline must
reproduce the existing BM25 Top100 before any alternative is evaluated. Select
on the fixed tuning half only; validation is reused development evidence.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from careguide.corpus import Corpus
from careguide.retrieval.bm25_index import tokenize_knowledge_text
from careguide.retrieval.knowledge_base import MedicalKnowledgeBase
from scripts.analyze_rag_failures import read_json
from scripts.cache_rag_recall_scores import atomic_json
from scripts.evaluate_rag_candidate_pools import BASELINE, partition_rows, rank
from scripts.evaluate_rag_score_fusion import metrics, paired_recall, validate_cache
from scripts.run_careguide_kuake_rag_evaluation import CORPUS_DIR, load_cases

TRIALS = tuple({'name': f'{grams}_b{b}', 'grams': grams, 'b': b, 'k1': 1.5}
               for grams in ('2', '12', '23', '123') for b in (.75, .3))
BASELINE_NAME = '2_b0.75'


def tokens(text, grams):
    if grams not in {'2', '12', '23', '123'}:
        raise ValueError('Unknown n-gram configuration')
    result = tokenize_knowledge_text(text)
    for run in re.findall(r'[\u3400-\u9fff]+', str(text).casefold()):
        # Existing tokenizer already emits single isolated characters and all
        # bigrams. Add other n-grams without duplicating those isolated tokens.
        if '1' in grams and len(run) > 1:
            result.extend(run)
        if '3' in grams:
            result.extend(run[i:i+3] for i in range(len(run)-2))
    return result


class PostingBM25:
    """NumPy posting scorer equivalent to native BM25's match/IDF contract."""

    def __init__(self, documents, grams):
        self.grams = grams
        ordered = sorted(documents)
        self.ids = np.asarray(ordered)
        self.lengths = np.zeros(len(ordered), dtype=float)
        raw = defaultdict(list)
        for i, key in enumerate(ordered):
            counts = Counter(tokens(documents[key], grams))
            self.lengths[i] = sum(counts.values())
            for term, count in counts.items():
                raw[term].append((i, count))
        self.average = float(self.lengths.mean()) or 1.
        self.postings = {}
        for term, values in raw.items():
            array = np.asarray(values)
            idf = math.log(1 + (len(ordered)-len(values)+.5)/(len(values)+.5))
            self.postings[term] = (array[:, 0].astype(int), array[:, 1].astype(float), idf)

    def search(self, query, *, b=.75, k1=1.5, top_k=100):
        terms = set(tokens(query, self.grams))
        scores = np.zeros(len(self.ids))
        matches = np.zeros(len(self.ids), dtype=int)
        identifier_matches = np.zeros(len(self.ids), dtype=bool)
        normalization = k1*(1-b+b*self.lengths/self.average)
        for term in sorted(terms):
            if term not in self.postings:
                continue
            indices, frequencies, idf = self.postings[term]
            scores[indices] += idf*frequencies*(k1+1)/(frequencies+normalization[indices])
            matches[indices] += 1
            if any(c.isdigit() for c in term) or '-' in term or '_' in term:
                identifier_matches[indices] = True
        eligible = np.flatnonzero((scores > 0) & ((matches >= (1 if len(terms) == 1 else 2)) | identifier_matches))
        selected = eligible[np.lexsort((self.ids[eligible], -scores[eligible]))[:top_k]]
        return [(str(self.ids[i]), float(scores[i])) for i in selected]


def check_baseline(expected, actual):
    if [v[0] for v in expected] != [v[0] for v in actual]:
        raise ValueError('Posting scorer does not reproduce original BM25 Top100 IDs')
    if not np.allclose([v[1] for v in expected], [v[1] for v in actual], atol=1e-10, rtol=1e-10):
        raise ValueError('Posting scorer does not reproduce original BM25 scores')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--selected-cache', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.selected_cache.exists():
        parser.error('New output paths required')
    cache = read_json(args.cache)
    rows, names = validate_cache(cache)
    if set(names) != {'dense', 'bm25'}:
        parser.error('Requires genuine two-route scores')
    cases, _ = load_cases('dev')
    queries = {case['id']: case['query'] for case in cases}
    documents = {r.id: MedicalKnowledgeBase.retrieval_text(r) for r in Corpus(CORPUS_DIR).records}
    results, variants = [], {}
    for grams in ('2', '12', '23', '123'):
        scorer = PostingBM25(documents, grams)
        for trial in (v for v in TRIALS if v['grams'] == grams):
            changed = []
            for row in rows:
                lexical = scorer.search(queries[row['id']], b=trial['b'], k1=trial['k1'])
                if trial['name'] == BASELINE_NAME:
                    check_baseline(row['streams']['bm25'], lexical)
                changed.append({**row, 'streams': {**row['streams'], 'bm25': lexical}})
            tune, valid = partition_rows(changed)
            result = {'config': trial}
            for name, selected in (('tuning', tune), ('validation', valid)):
                rankings = [rank(row, BASELINE) for row in selected]
                result[name] = metrics(selected, rankings)
                result[name]['bm25_top100'] = sum(row['gold_record_id'] in dict(row['streams']['bm25']) for row in selected)/len(selected)
                result[name]['union_top100_each'] = sum(any(row['gold_record_id'] in dict(stream) for stream in row['streams'].values()) for row in selected)/len(selected)
            results.append(result)
            variants[trial['name']] = changed
            print(json.dumps({'completed': trial['name'], 'tuning': result['tuning']}), flush=True)
    best = max(results, key=lambda r: (r['tuning']['candidate_coverage_at_40'], r['tuning']['mrr_at_10'], r['config']['name'] == BASELINE_NAME))
    selected = variants[best['config']['name']]
    _, before = partition_rows(rows)
    _, after = partition_rows(selected)
    paired = paired_recall(after, [rank(r, BASELINE) for r in before], [rank(r, BASELINE) for r in after], 40)
    report = {'evaluation': 'offline Chinese lexical ablation with fixed dense/fusion',
              'external_requests': 0, 'frozen_test_run': False, 'baseline_reproduced_queries': len(rows),
              'source_cache_sha256': hashlib.sha256(args.cache.read_bytes()).hexdigest(),
              'source_metadata': cache['metadata'], 'trial_count': len(TRIALS), 'results': results,
              'selection': ['tuning Top40 coverage', 'tuning MRR@10', 'prefer baseline on ties'],
              'selected_config': best['config'], 'selected_paired_validation_at40': paired,
              'scope': 'Reused dev, no final rerank claim. Validation not used to select. No runtime change.'}
    atomic_json(args.output, report)
    args.selected_cache.parent.mkdir(parents=True, exist_ok=True)
    saved = {'metadata': {**cache['metadata'], 'lexical_experiment': best['config']}, 'rows': selected}
    with args.selected_cache.open('wb') as handle:
        handle.write(gzip.compress(json.dumps(saved, ensure_ascii=False).encode(), mtime=0))
    print(json.dumps({'selected': best, 'paired_validation': paired}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
