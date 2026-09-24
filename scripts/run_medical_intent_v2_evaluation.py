"""Explicit V2 fine-intent evaluation; no evaluation rows enter recognizer templates."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections import Counter
from pathlib import Path

from careguide.intent_v2_dataset import (BOUNDARY_LABELS, DATASET_ROOT, DOMAIN_BY_FINE,
                                         FINE_LABELS, load_v2_development_cases,
                                         load_v2_evaluation_cases)


def metrics(cases, predictions):
    labels = [*FINE_LABELS, *BOUNDARY_LABELS]
    gold = [c.primary_fine_intent.value if c.primary_fine_intent else c.disposition.value for c in cases]
    predicted = [r.fine_intent if r.disposition == 'route' else r.disposition for r in predictions]
    confusion = {label: dict(sorted(Counter(p for g, p in zip(gold, predicted) if g == label).items()))
                 for label in labels}
    by_label = {}
    for label in labels:
        tp = sum(g == p == label for g, p in zip(gold, predicted))
        fp = sum(g != label and p == label for g, p in zip(gold, predicted))
        fn = sum(g == label and p != label for g, p in zip(gold, predicted))
        precision = tp/(tp+fp) if tp+fp else 0
        recall = tp/(tp+fn) if tp+fn else 0
        by_label[label] = {'precision': precision, 'recall': recall,
                           'f1': 2*precision*recall/(precision+recall) if precision+recall else 0,
                           'support': sum(g == label for g in gold)}
    domain_total = sum(c.primary_domain in {'triage', 'medication', 'appointment'} for c in cases)
    coarse_by_domain = {'triage': 'symptom_query', 'medication': 'medication_query',
                        'appointment': 'appointment'}
    domain_correct = sum(c.primary_domain in coarse_by_domain and r.intent is not None and
                         r.intent.value == coarse_by_domain[c.primary_domain]
                         for c, r in zip(cases, predictions))
    return {'count': len(cases), 'fine_accuracy': sum(g == p for g, p in zip(gold, predicted))/len(cases),
            'fine_macro_f1': sum(v['f1'] for v in by_label.values())/len(labels),
            'per_label': by_label, 'confusion': confusion,
            'disposition_accuracy': sum(c.disposition.value == r.disposition for c, r in zip(cases, predictions))/len(cases),
            'domain_mapping_accuracy': domain_correct/domain_total if domain_total else None,
            'domain_mapping_count': domain_total,
            'multi_intent_exact_match': sum(set(i.value for i in c.fine_intents) == set(r.fine_intents)
                                            for c, r in zip(cases, predictions))/len(cases),
            'branch_failures': {branch: sum(r.source_scores.get(branch, {}).get('failed', False)
                                            for r in predictions) for branch in ('llm', 'embedding', 'pattern')},
            'missing_entity_annotation_coverage': sum(bool(c.missing_entities) for c in cases)/len(cases),
            'source_coverage_annotation': 'not_assessed'}


async def run(args):
    from scripts.private_config import load_config
    load_config(args.config_file)
    from careguide.intent_v2_recognizer import V2IntentRecognizer
    cases = load_v2_evaluation_cases() if args.split == 'heldout' else load_v2_development_cases()
    client = None
    if not args.offline:
        from careguide.llm_utils import create_llm_client
        import os
        key = os.environ.get('CAREGUIDE_LLM_API_KEY', '')
        if not key or key.startswith('YOUR_'):
            raise ValueError('Configure the real LLM provider or explicitly select --offline')
        client = create_llm_client(key, os.environ.get('CAREGUIDE_LLM_BASE_URL') or None)
    import os
    mode = 'local_ngram' if args.offline else os.environ.get('CAREGUIDE_INTENT_EMBEDDING_MODE', 'local_ngram')
    embedding = None
    if mode == 'remote':
        from careguide.runtime import semantic_embedding
        embedding = semantic_embedding()
    recognizer = V2IntentRecognizer(client, os.environ.get('CAREGUIDE_LLM_MODEL', 'gpt-4o-mini'),
                                    embedding=embedding, embedding_mode=mode)
    semaphore = asyncio.Semaphore(args.concurrency)
    async def predict(case):
        async with semaphore:
            return await recognizer.recognize(case.message, [t.model_dump() for t in case.history])
    try:
        predictions = await asyncio.gather(*(predict(case) for case in cases))
    finally:
        if client is not None:
            await client.close()
    report = {'version': 'careguide-medical-intent-v2', 'split': args.split,
              'dataset_sha256': hashlib.sha256((DATASET_ROOT / f'{args.split}.jsonl').read_bytes()).hexdigest(),
              'synthetic': True, 'clinical_validity_evaluated': False,
              'runtime_mode': 'offline_local_ngram' if args.offline else 'configured_provider',
              'metrics': metrics(cases, predictions),
              'predictions': [{'id': case.id,
                               'gold': case.primary_fine_intent.value if case.primary_fine_intent else case.disposition.value,
                               'predicted': result.fine_intent if result.disposition == 'route' else result.disposition,
                               'fine_intents': result.fine_intents,
                               'disposition': result.disposition,
                               'confidence': result.confidence} for case, result in zip(cases, predictions)]}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(json.dumps({k: report['metrics'][k] for k in ('count', 'fine_accuracy', 'fine_macro_f1',
                        'disposition_accuracy', 'domain_mapping_accuracy')}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['dev', 'heldout'], required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config-file')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--concurrency', type=int, choices=range(1, 9), default=2)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise SystemExit('Refusing to overwrite an evaluation report')
    asyncio.run(run(args))


if __name__ == '__main__':
    main()
