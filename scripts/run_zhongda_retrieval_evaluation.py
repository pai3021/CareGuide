"""Evaluate Zhongda pilot record ranking and scoped read-only tool decisions."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

from careguide.corpus import Corpus
from careguide.domain import AgentType
from careguide.evaluation.retrieval import evaluate_retrieval_cases
from careguide.tools import MedicalToolRegistry
from scripts.run_medical_framework_evaluation import retrieval_runtime

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / 'data' / 'eval' / 'zhongda_retrieval_dev'
CORPUS = ROOT / 'data' / 'medical_zhongda_v2'


def load_cases(path):
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    identities = [row['case_id'] for row in rows]
    if not rows or len(identities) != len(set(identities)):
        raise ValueError('Evaluation cases must have unique, nonempty identities')
    return rows


def evaluate_tools(registry, cases, today):
    rows = []
    for case in cases:
        answer = registry.query(AgentType(case['agent']), case['tool'], case['arguments'], today=today)
        ids = [item['record_id'] for item in answer['evidence']]
        scope_leak = any(item['scope'].get('hospital_id') != 'zhongda' or
                         item['scope'].get('campus') != '本部' for item in answer['evidence'])
        passed = (answer['status'] == case['expected_status'] and
                  set(case['required_evidence_ids']) <= set(ids) and
                  not scope_leak and not answer['side_effects'] and
                  (answer['status'] == 'found' or not ids))
        rows.append({'case_id': case['case_id'], 'expected_status': case['expected_status'],
                     'actual_status': answer['status'], 'required_evidence_ids': case['required_evidence_ids'],
                     'returned_ids': ids, 'scope_leak': scope_leak, 'passed': passed})
    return {'total': len(rows), 'passed': sum(row['passed'] for row in rows),
            'scope_leaks': sum(row['scope_leak'] for row in rows), 'cases': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus-dir', type=Path, default=CORPUS)
    parser.add_argument('--retrieval-mode', choices=('bm25', 'dense', 'hybrid'), default='bm25')
    parser.add_argument('--config-file', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refusing to overwrite an evaluation report')
    from scripts.private_config import load_config
    load_config(args.config_file)
    corpus = Corpus(args.corpus_dir)
    if corpus.version != 'careguide-medical-zhongda-v2':
        raise ValueError('This contract targets the Zhongda V2 corpus only')
    paths = {name: CASES / f'{name}_cases.jsonl' for name in ('retrieval', 'tool')}
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    cases = {name: load_cases(path) for name, path in paths.items()}
    knowledge = retrieval_runtime(corpus, args.retrieval_mode)
    report = {'evaluation_version': 'careguide-zhongda-retrieval-dev-1',
              'split': 'dev', 'heldout_used': False, 'clinical_validity_evaluated': False,
              'retrieval_mode': args.retrieval_mode, 'corpus_version': corpus.version,
              'corpus_fingerprint': corpus.fingerprint, 'fixture_sha256': hashes,
              'retrieval': evaluate_retrieval_cases(knowledge, cases['retrieval'], today=date.today()),
              'scoped_tools': evaluate_tools(MedicalToolRegistry(knowledge), cases['tool'], date.today())}
    if hashes != {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()} or \
            corpus.fingerprint != Corpus(args.corpus_dir).fingerprint:
        raise RuntimeError('Evaluation input changed during run')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write('\n')
    print(json.dumps({'retrieval': {key: report['retrieval'][key] for key in
                      ('total', 'positive_count', 'hard_negative_count', 'recall_at_1',
                       'recall_at_3', 'hard_negative_accuracy')},
                      'scoped_tools': {key: report['scoped_tools'][key] for key in
                       ('total', 'passed', 'scope_leaks')}, 'report': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
