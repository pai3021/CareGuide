"""Explicit evaluation entrypoint; never implicitly opens heldout or promotes baseline."""
import argparse
import asyncio
import json
from pathlib import Path

from careguide.evaluator import IntentEvaluator, IntentTestCase
from careguide.intent_dataset import load_development_cases, load_evaluation_cases
from careguide.runtime import get_chat_runtime


async def run(args):
    from scripts.private_config import load_config
    load_config(args.config_file)
    from scripts.validate_real_provider import source_hashes
    before = source_hashes()
    cases = load_evaluation_cases() if args.split == 'heldout' else load_development_cases()
    runtime = get_chat_runtime()
    if runtime[0].client is None:
        raise ValueError('Configure the real LLM provider before reporting recognition results')
    def progress(done, total):
        if done % 20 == 0 or done == total:
            print(f'Evaluated {done}/{total}', flush=True)
    try:
        results = await IntentEvaluator(runtime[0].recognizer).evaluate([
            IntentTestCase(c.message, c.primary_intent.value if c.primary_intent else c.disposition.value,
                           {'history': [t.model_dump() for t in c.history]}) for c in cases],
            concurrency=args.concurrency, progress=progress)
    finally:
        await runtime[1].close()
        await runtime[0].client.close()
    output = Path(args.output)
    assert source_hashes() == before, 'Frozen corpus or dataset changed during evaluation'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as file:
        json.dump({'split': args.split, 'synthetic': True, 'clinical_validity_evaluated': False,
                   'metrics_scope': 'primary_intent_or_boundary_state', 'results': results}, file, ensure_ascii=False, indent=2)
    print(f"Accuracy={results['accuracy']} Macro-F1={results['macro_f1']}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['dev', 'heldout'], required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config-file')
    parser.add_argument('--concurrency', type=int, choices=range(1, 9), default=1)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise SystemExit('Refusing to overwrite an evaluation report')
    asyncio.run(run(args))


if __name__ == '__main__':
    main()
