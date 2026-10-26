"""Offline labeled probes or explicit small paid answer/judge smoke. No .env loading.

Live mode uses the configured model, including a separate judge call for a known
bad counterexample. Same-model judging is provisional, not independent evidence.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from careguide.evaluation.faithfulness import judge_answer, quantity_risks
from careguide.llm_utils import create_llm_client, track_token_usage
from careguide.orchestration.answering import generate_answer, validate_answer
from careguide.orchestration.budget import BudgetLedger, BudgetLimits, use_budget
from careguide.orchestration.contracts import TaskStatus

CASES = ROOT/'careguide'/'evaluation'/'answer_cases.json'


def probe_case(case, text):
    try:
        validate_answer(text, case['sources'])
        accepted = True
    except ValueError:
        accepted = False
    return {'contract_accepted': accepted, 'quantity_check': quantity_risks(text, case['sources']),
            'labeled_probe_coverage': sum(bool(re.search(p, text)) for p in case['probes'])/len(case['probes'])}


def fixture_state(case):
    tasks = {}
    for source in case['sources']:
        key = source['task_id']
        if key not in tasks:
            tasks[key] = SimpleNamespace(spec=SimpleNamespace(fragment=case['question'], action='search_care_guidance'),
                status=TaskStatus.SUCCEEDED, failure=None, result={'evidence': []})
        tasks[key].result['evidence'].append({'text': source['text']})
    return SimpleNamespace(tasks=tasks, status='completed')


async def evaluate(mode, limit=4, ids=None):
    cases = json.loads(CASES.read_text(encoding='utf-8'))
    model = os.environ.get('CAREGUIDE_LLM_MODEL', 'mimo-v2.6-flash')
    client = None
    if mode == 'live':
        key = os.environ.get('CAREGUIDE_LLM_API_KEY', '')
        if not key or key.startswith('YOUR_'):
            raise ValueError('Live evaluation requires explicitly configured model credentials')
        client = create_llm_client(key, os.environ.get('CAREGUIDE_LLM_BASE_URL') or 'https://api.xiaomimimo.com/v1')
    rows = []
    try:
        with tempfile.TemporaryDirectory(prefix='careguide-answer-eval-') as folder, track_token_usage() as usage:
            selected = [case for case in cases['cases'] if ids is None or case['id'] in ids][:limit]
            for case in selected:
                row = {'id': case['id'], 'known_good': probe_case(case, case['good']), 'known_bad': probe_case(case, case['bad'])}
                if client:
                    started = time.monotonic()
                    budget = BudgetLedger(Path(folder)/'budget.sqlite3', case['id'], 'eval', 'eval', BudgetLimits(model=4, tool=0))
                    try:
                        answer, diagnostic = await generate_answer(client, model, fixture_state(case), budget, case['question'])
                        row['generation'] = {k: v for k, v in diagnostic.items() if k != 'body'}
                        body = diagnostic.get('body', '')
                        row['answer'] = body
                        if answer is not None:
                            row['generated_probes'] = probe_case(case, body)
                            with use_budget(budget):
                                row['judge'] = await judge_answer(client, model, body, case['sources'])
                        with use_budget(budget):
                            row['known_bad_judge'] = await judge_answer(client, model, case['bad'], case['sources'])
                    except Exception as exc:
                        row['error'] = type(exc).__name__
                    row['budget'] = budget.summary()
                    row['elapsed_ms'] = round((time.monotonic()-started)*1000, 2)
                rows.append(row)
            token_usage = usage.to_dict()
    finally:
        if client:
            await client.close()
    return {'kind': 'labeled_answer_faithfulness_probes', 'version': cases['version'], 'mode': mode,
        'model': model if client else None, 'cases': len(rows), 'rows': rows, 'usage': token_usage,
        'provider_requests': sum(row.get('budget', {}).get('used', {}).get('model', 0) for row in rows),
        'requests_without_provider_usage': sum(row.get('budget', {}).get('tokens', {}).get('calls_without_usage', 0) for row in rows),
        'interpretation': 'Contract checks and hand-labeled probes are narrow diagnostics. Live judge is the same model family, with exact-quote validation and known-bad controls; not independent clinical or semantic accuracy.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['offline', 'live'], default='offline')
    parser.add_argument('--limit', type=int, choices=range(1, 5), default=4)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--ids', nargs='+', choices=['quantity', 'negation', 'conditions', 'source_match'])
    args = parser.parse_args()
    report = asyncio.run(evaluate(args.mode, args.limit, args.ids))
    data = json.dumps(report, ensure_ascii=False, indent=2)+'\n'
    if args.output:
        with args.output.open('x', encoding='utf-8') as handle:
            handle.write(data)
        print(json.dumps({'output': str(args.output), 'cases': report['cases'], 'usage': report['usage']}))
    else:
        print(data)
