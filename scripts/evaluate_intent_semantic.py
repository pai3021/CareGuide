"""Independent business-example retrieval evaluation. Default is an offline lexical baseline.

--mode remote explicitly uses the configured embedding API and can incur fees.
No .env loading, LLM planning, hospital actions, or frozen dataset writes.
"""
import argparse
import asyncio
import json
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from careguide.orchestration.budget import BudgetLedger
from careguide.orchestration.semantic_intent import load_bank
from careguide.orchestration.understanding import local_candidates
from careguide.runtime import intent_semantic_retriever

CASES = Path(__file__).resolve().parents[1]/'careguide/orchestration/intent_semantic_cases.json'


async def evaluate(mode):
    cases = json.loads(CASES.read_text(encoding='utf-8'))
    bank = load_bank()
    assert not {e['text'] for e in bank['examples']} & {c['text'] for c in cases['cases']}
    retriever = intent_semantic_retriever('remote')[0] if mode == 'remote' else None
    rows = []
    try:
        with tempfile.TemporaryDirectory(prefix='careguide-intent-eval-') as folder:
            for case in cases['cases']:
                started = time.monotonic()
                if retriever:
                    budget = BudgetLedger(Path(folder)/'eval.sqlite3', case['id'], 'eval', 'eval')
                    candidates, diagnostic = await retriever.retrieve(case['text'], budget=budget, scope=('eval', case['id']))
                    charged = budget.summary()['used']['model']
                else:
                    candidates, diagnostic, charged = local_candidates(case['text']), {'status': 'local_char_ngram'}, 0
                retrieved = {action for c in candidates for action in c.get('actions', [c.get('action')]) if action}
                expected = set(case['actions'])
                rows.append({'id': case['id'], 'category': case['category'], 'diagnostic': diagnostic,
                    'expected_actions': sorted(expected), 'retrieved_actions': sorted(retrieved),
                    'action_coverage': len(expected & retrieved)/len(expected) if expected else None,
                    'forbidden_candidate_actions': sorted(retrieved & set(case.get('forbidden_actions', []))),
                    'remote_calls': charged, 'latency_ms': (time.monotonic()-started)*1000})
    finally:
        if retriever:
            await retriever.close()
    scored = [r['action_coverage'] for r in rows if r['action_coverage'] is not None]
    return {'kind': 'business_example_retrieval_only', 'mode': mode, 'case_version': cases['version'],
        'bank_version': bank['version'], 'cases': len(rows), 'scored_cases': len(scored),
        'mean_action_coverage': sum(scored)/len(scored), 'remote_calls': sum(r['remote_calls'] for r in rows),
        'direct_acceptance': False, 'rows': rows,
        'interpretation': 'Candidate action coverage only, not goal extraction, calibrated routing, clinical accuracy, or execution success. Near neighbours may include negated/forbidden actions; the planner must preserve the current request.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['local_char_ngram', 'remote'], default='local_char_ngram')
    args = parser.parse_args()
    print(json.dumps(asyncio.run(evaluate(args.mode)), ensure_ascii=False, indent=2))
