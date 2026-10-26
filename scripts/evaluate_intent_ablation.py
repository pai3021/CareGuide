"""Controlled candidate-source comparison; plans are validated but never executed.

Offline mode scores strict local understanding only. Live mode compares rules+
LLM, adding the existing RoBERTa, then adding business embedding references.
No .env loading, hospital calls, writes, or automatic score-based routing.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from careguide.corpus import Corpus, DEFAULT_CORPUS
from careguide.llm_utils import create_llm_client, track_token_usage
from careguide.orchestration.budget import BudgetLedger, BudgetLimits
from careguide.orchestration.understanding import LocalClassifier, LocalUnderstanding, compose_plan
from careguide.orchestration.understanding_planner import UnderstandingPlanner, build_context
from careguide.runtime import intent_semantic_retriever

CASES = ROOT/'careguide'/'evaluation'/'intent_cases.json'
VARIANTS = ('rules_llm', 'rules_classifier_llm', 'rules_classifier_embedding_llm')


def score(case, plan):
    tasks = plan.tasks if plan else []
    actions = {task.action for task in tasks}
    expected = set(case['actions'])
    precision = len(actions & expected)/len(actions) if actions else 0
    recall = len(actions & expected)/len(expected)
    arguments = all(any(task.action == action and all(task.arguments.get(k) == v for k, v in values.items())
                        for task in tasks) for action, values in case.get('arguments', {}).items())
    return {'action_precision': precision, 'action_recall': recall,
        'action_f1': 2*precision*recall/(precision+recall) if precision+recall else 0,
        'argument_match': arguments, 'forbidden_actions': sorted(actions & set(case.get('forbidden_actions', []))),
        'complete': actions == expected and arguments and not actions & set(case.get('forbidden_actions', []))}


async def evaluate(mode, limit=6):
    dataset = json.loads(CASES.read_text(encoding='utf-8'))
    corpus, rows = Corpus(DEFAULT_CORPUS), []
    client = classifier = semantic = None
    model = os.environ.get('CAREGUIDE_LLM_MODEL', 'mimo-v2.6-flash')
    warmup = {'status': 'not_requested'}
    if mode == 'live':
        key = os.environ.get('CAREGUIDE_LLM_API_KEY', '')
        if not key or key.startswith('YOUR_'):
            raise ValueError('Live evaluation requires explicitly configured model credentials')
        client = create_llm_client(key, os.environ.get('CAREGUIDE_LLM_BASE_URL') or 'https://api.xiaomimimo.com/v1')
        path = os.environ.get('CAREGUIDE_QIC_MODEL_PATH')
        if path:
            classifier = LocalClassifier(path, os.environ.get('CAREGUIDE_QIC_MODEL_DEVICE', 'cpu'))
            started = time.monotonic()
            try:
                await asyncio.to_thread(classifier.predict, '查消化内科医生')
                warmup = {'status': 'ready', 'seconds': round(time.monotonic()-started, 3), 'included_in_query_latency': False}
            except Exception as exc:
                warmup = {'status': 'unavailable', 'diagnostic_type': type(exc).__name__}
                classifier = None
        try:
            semantic = intent_semantic_retriever('remote')[0]
        except ValueError:
            pass
    try:
        with tempfile.TemporaryDirectory(prefix='careguide-ablation-') as folder, track_token_usage() as usage:
            for variant in VARIANTS if client else ('rules_only',):
                if 'classifier' in variant and classifier is None:
                    rows.append({'variant': variant, 'status': 'not_run_classifier_unavailable'})
                    continue
                if 'embedding' in variant and semantic is None:
                    rows.append({'variant': variant, 'status': 'not_run_embedding_unavailable'})
                    continue
                understanding = LocalUnderstanding(corpus, classifier if 'classifier' in variant else None,
                    semantic=semantic if 'embedding' in variant else None)
                for case in dataset['cases'][:limit]:
                    started, at = time.monotonic(), datetime.now(timezone.utc).isoformat()
                    budget = BudgetLedger(Path(folder)/'budget.sqlite3', variant+'-'+case['id'], 'eval', 'eval',
                                          BudgetLimits(model=4, tool=0))
                    row = {'variant': variant, 'id': case['id'], 'status': 'ok'}
                    plan, events = None, []
                    try:
                        result = await understanding.understand(case['text'], message_id='eval-message', at=at, classify=False)
                        # Remove char-neighbours in every arm to isolate the two added sources.
                        result.candidates = []
                        result.recognition_source = ['rules']
                        plan, _ = compose_plan(case['text'], result)
                        if plan is None and client:
                            await understanding.enrich(case['text'], result, budget=budget, scope=('eval', variant+'-'+case['id']))
                            context = build_context(case['text'], 'eval-message', at, case.get('history', []), understanding=result)
                            decision, events = await UnderstandingPlanner(client, model).decide(context, budget=budget)
                            plan = decision.plan
                            row['decision'] = decision.kind
                        row['plan_source'] = 'local' if not events else 'llm'
                        row['candidate_sources'] = sorted({candidate['source'] for candidate in result.candidates})
                        row['candidate_diagnostics'] = result.diagnostics
                        row['semantic_retrieval'] = result.semantic_retrieval
                    except Exception as exc:
                        row.update(status='error', diagnostic_type=type(exc).__name__)
                    row.update(score(case, plan))
                    row['budget'] = budget.summary()
                    row['elapsed_ms'] = round((time.monotonic()-started)*1000, 2)
                    rows.append(row)
            usage_report = usage.to_dict()
    finally:
        if semantic:
            await semantic.close()
        if client:
            await client.close()
    summary = {}
    for variant in sorted({row['variant'] for row in rows}):
        samples = [row for row in rows if row['variant'] == variant and 'id' in row]
        summary[variant] = {'cases': len(samples), 'complete': sum(row['complete'] for row in samples),
            'errors': sum(row['status'] == 'error' for row in samples),
            'classifier_candidate_cases': sum('classifier' in row.get('candidate_sources', []) for row in samples),
            'embedding_candidate_cases': sum('semantic_embedding' in row.get('candidate_sources', []) for row in samples),
            'model_requests': sum(row['budget']['used']['model'] for row in samples),
            'mean_latency_ms': round(sum(row['elapsed_ms'] for row in samples)/len(samples), 2) if samples else None}
    return {'kind': 'intent_candidate_ablation', 'mode': mode, 'model': model if client else None,
        'dataset_version': dataset['version'], 'dataset_sha256': hashlib.sha256(CASES.read_bytes()).hexdigest(),
        'classifier_warmup': warmup, 'summary': summary, 'rows': rows, 'chat_usage': usage_report,
        'interpretation': 'Six selected annotated cases, one run per arm. Canonical case is a local control. Character candidates removed equally. No tools executed. Cold embedding index cost included; classifier warmup excluded. Not a representative accuracy estimate or proof of improvement.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['offline', 'live'], default='offline')
    parser.add_argument('--limit', type=int, choices=range(1, 7), default=6)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = asyncio.run(evaluate(args.mode, args.limit))
    data = json.dumps(report, ensure_ascii=False, indent=2)+'\n'
    if args.output:
        with args.output.open('x', encoding='utf-8') as handle:
            handle.write(data)
        print(json.dumps({'output': str(args.output), 'summary': report['summary'], 'chat_usage': report['chat_usage']}))
    else:
        print(data)
