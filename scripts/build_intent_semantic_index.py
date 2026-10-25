"""Explicitly build a business-example index using the configured embedding API (may incur fees)."""
import argparse
import asyncio
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from careguide.orchestration.budget import BudgetLedger
from careguide.runtime import intent_semantic_retriever


async def build(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Refusing to overwrite an existing intent index')
    if not output.parent.is_dir():
        raise FileNotFoundError('Create the output directory explicitly before building')
    retriever = intent_semantic_retriever('remote')[0]
    try:
        with tempfile.TemporaryDirectory(prefix='careguide-intent-index-') as folder:
            budget = BudgetLedger(Path(folder)/'build.sqlite3', 'index', 'index', 'index')
            artifact = await retriever.export_index(budget=budget)
        with output.open('x', encoding='utf-8') as handle:
            json.dump(artifact, handle, ensure_ascii=False, allow_nan=False)
        return {'status': 'built', 'examples': len(artifact['vectors']), 'fingerprint': artifact['fingerprint']}
    finally:
        await retriever.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    print(json.dumps(asyncio.run(build(parser.parse_args().output))))
