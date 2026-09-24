import asyncio
from copy import deepcopy
from datetime import date

from careguide.agent_orchestrator import AgentOrchestrator
from careguide.corpus import Corpus
from careguide.evaluation.execution import ExecutionCase, evaluate_execution
from careguide.evaluation.retrieval import evaluate_retrieval_cases
from careguide.evaluation.skill_selection import evaluate_skill_cases
from careguide.intent_recognizer import IntentRecognizer
from careguide.retrieval.knowledge_base import MedicalKnowledgeBase
from careguide.skill_registry import SkillRegistry
from careguide.tools import MedicalToolRegistry
from careguide.domain import AGENTS
from scripts.run_medical_framework_evaluation import FIXTURES, load_cases


def _engine():
    skills = SkillRegistry(tool_scopes={agent.value: spec.tools for agent, spec in AGENTS.items()})
    skills.load()
    return AgentOrchestrator(IntentRecognizer(), MedicalToolRegistry(MedicalKnowledgeBase(Corpus())), skills=skills)


def test_trace_contract_detects_wrong_tool_args_evidence_and_side_effect():
    case = ExecutionCase.from_dict(load_cases(FIXTURES/'execution_cases.jsonl')[1])
    engine = _engine()
    reply = asyncio.run(engine.process(case.message, 'synthetic-test-user', case.case_id))
    original = engine.traces.get(reply['request_id'])
    assert evaluate_execution(case, original)['task_success']

    changed = deepcopy(original)
    changed['execution']['tool_calls'][0]['tool_name'] = 'lookup_hospital_guide'
    changed['execution']['tool_calls'][0]['input']['specification'] = '0.4g'
    changed['evidence'] = []
    changed['side_effects'] = [{'type': 'unexpected_write'}]
    score = evaluate_execution(case, changed)
    assert not score['task_success']
    assert {'MISSING_REQUIRED_TOOL', 'UNNECESSARY_TOOL', 'FORBIDDEN_TOOL',
            'MISSING_EVIDENCE', 'SIDE_EFFECT_FAILURE'} <= set(score['failure_labels'])
    assert score['earliest_failure_layer'] == 'tool_selection'


def test_trace_contract_detects_argument_error_without_tool_name_change():
    case = ExecutionCase.from_dict(load_cases(FIXTURES/'execution_cases.jsonl')[1])
    engine = _engine()
    reply = asyncio.run(engine.process(case.message, 'synthetic-test-user', case.case_id))
    changed = deepcopy(engine.traces.get(reply['request_id']))
    changed['execution']['tool_calls'][0]['input']['specification'] = '0.4g'
    score = evaluate_execution(case, changed)
    assert score['failure_labels'] == ['TOOL_ARGUMENT_FAILURE']
    assert score['earliest_failure_layer'] == 'tool_arguments'


def test_skill_dev_contract_covers_positive_and_no_skill_routes():
    skills = SkillRegistry(tool_scopes={agent.value: spec.tools for agent, spec in AGENTS.items()})
    skills.load()
    report = evaluate_skill_cases(skills, load_cases(FIXTURES/'skill_cases.jsonl'))
    assert report['total'] == 6 and report['no_skill_accuracy'] == 1
    assert report['external_llm_calls'] == 0


def test_retrieval_dev_contract_separates_positive_recall_and_negative_rejection():
    corpus = Corpus()
    report = evaluate_retrieval_cases(MedicalKnowledgeBase(corpus),
        load_cases(FIXTURES/'retrieval_cases.jsonl'), today=date(2026, 9, 19))
    assert report['positive_count'] == 9 and report['hard_negative_count'] == 2
    assert report['recall_at_1'] == 1 and report['hard_negative_accuracy'] == 0
    assert report['scope_violations'] == 0
