"""Run separate medical execution, Skill, and retrieval development contracts.

Execution uses the explicitly labeled offline rule fallback unless a prepared
trace bundle is supplied. Neither path uses the 120-case intent holdout.
"""
import argparse
import asyncio
import hashlib
import json
import os
import tempfile
from datetime import date
from pathlib import Path

from careguide.agent_orchestrator import AgentOrchestrator
from careguide.business_store import BusinessStore
from careguide.corpus import Corpus
from careguide.domain import AGENTS
from careguide.evaluation.execution import ExecutionCase, evaluate_execution, summarize_execution
from careguide.evaluation.retrieval import evaluate_retrieval_cases
from careguide.evaluation.skill_selection import evaluate_skill_cases
from careguide.intent_recognizer import IntentRecognizer
from careguide.retrieval.knowledge_base import MedicalKnowledgeBase, versioned_collection_name, COLLECTION_NAME
from careguide.skill_registry import SkillRegistry
from careguide.tools import MedicalToolRegistry


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT/'data'/'eval'/'medical_framework_dev'


def load_cases(path):
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    identities = [row['case_id'] for row in rows]
    if len(identities) != len(set(identities)) or not rows:
        raise ValueError('Evaluation cases must be nonempty with unique identities')
    return rows


async def execute_offline(cases, registry, skills):
    with tempfile.TemporaryDirectory(prefix='careguide-contract-') as directory:
        engine = AgentOrchestrator(IntentRecognizer(embedding_mode='local_ngram'), registry,
            skills=skills, business_store=BusinessStore(Path(directory)/'handoffs.sqlite3'))
        rows = []
        for case in cases:
            reply = await engine.process(case.message, 'synthetic-contract-user', case.case_id)
            rows.append(evaluate_execution(case, engine.traces.get(reply['request_id'])))
    return summarize_execution(rows)


def evaluate_trace_bundle(cases, path):
    traces = load_cases(path)
    by_id = {row['case_id']: row['trace'] for row in traces}
    expected = {case.case_id for case in cases}
    if set(by_id) != expected:
        raise ValueError('Trace bundle and contract case IDs must match exactly')
    return summarize_execution([evaluate_execution(case, by_id[case.case_id]) for case in cases])


def retrieval_runtime(corpus, mode):
    if mode == 'bm25':
        return MedicalKnowledgeBase(corpus)
    from careguide.runtime import chroma_client, semantic_embedding
    client = chroma_client()
    names = {item.name for item in client.list_collections()}
    name = versioned_collection_name(corpus) if versioned_collection_name(corpus) in names else COLLECTION_NAME
    if name not in names:
        raise ValueError('Explicit medical vector index is missing; run index_medical first')
    return MedicalKnowledgeBase(corpus, mode=mode,
        collection=client.get_collection(name, embedding_function=None), embedding=semantic_embedding(),
        min_dense_score=float(os.environ.get('CAREGUIDE_MIN_DENSE_SCORE', '0.5')))


def run(args):
    from scripts.private_config import load_config
    load_config(args.config_file)
    files = {name: FIXTURES/(name+'_cases.jsonl') for name in ('execution', 'skill', 'retrieval')}
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}
    corpus = Corpus()
    corpus_fingerprint = corpus.fingerprint
    execution_cases = [ExecutionCase.from_dict(row) for row in load_cases(files['execution'])]
    skill_cases = load_cases(files['skill'])
    retrieval_cases = load_cases(files['retrieval'])
    skills = SkillRegistry(tool_scopes={agent.value: spec.tools for agent, spec in AGENTS.items()})
    skills.load()
    if args.trace_bundle:
        execution = evaluate_trace_bundle(execution_cases, args.trace_bundle)
        execution_mode = 'provided_trace_bundle'
    else:
        registry = MedicalToolRegistry(MedicalKnowledgeBase(corpus))
        execution = asyncio.run(execute_offline(execution_cases, registry, skills))
        execution_mode = 'offline_rule_fallback'
    skill = evaluate_skill_cases(skills, skill_cases)
    retrieval = evaluate_retrieval_cases(retrieval_runtime(corpus, args.retrieval_mode), retrieval_cases,
        today=date.today())
    if corpus.fingerprint != corpus_fingerprint or hashes != {
            name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}:
        raise RuntimeError('Evaluation source changed during run')
    report = {'evaluation_version': 'careguide-medical-framework-dev-1',
        'split': 'dev', 'heldout_used': False, 'medical_answering_enabled': False,
        'clinical_validity_evaluated': False, 'execution_mode': execution_mode,
        'retrieval_mode': args.retrieval_mode, 'corpus_fingerprint': corpus_fingerprint,
        'fixture_sha256': hashes, 'execution': execution, 'skill': skill, 'retrieval': retrieval}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write('\n')
    print(json.dumps({'execution': {k: execution[k] for k in ('total','passed','task_success_rate')},
        'skill': {k: skill[k] for k in ('total','correct','exact_selection_accuracy')},
        'retrieval': {k: retrieval[k] for k in ('total','recall_at_1','recall_at_3','mrr',
                                               'hard_negative_accuracy','scope_violations')},
        'report': str(output)}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-file', type=Path)
    parser.add_argument('--retrieval-mode', choices=('bm25','dense','hybrid'), default='bm25')
    parser.add_argument('--trace-bundle', type=Path, help='JSONL case_id + trace pairs; never runs agents')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refusing to overwrite an evaluation report')
    if args.trace_bundle and not args.trace_bundle.is_file():
        parser.error('Trace bundle does not exist')
    run(args)


if __name__ == '__main__':
    main()
