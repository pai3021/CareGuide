import asyncio
from collections import Counter
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from careguide.evaluation.business_intents import BusinessCase, DATASET, load_cases
from careguide.orchestration.contracts import PlanProposal, TaskSpec
from careguide.domain import AgentType
from careguide.orchestration.semantic_intent import load_bank
from scripts.evaluate_business_intents import evaluate, score_local
from tests.test_layered_understanding import prediction


def test_business_evaluation_is_frozen_action_labeled_and_reference_disjoint():
    data, cases, digest = load_cases()
    lock = json.loads((Path(__file__).resolve().parents[1]/'docs/evaluation/business-intent-eval-v1.lock.json').read_text())
    assert digest == lock['dataset_sha256'] and len(cases) == 64
    assert data['split'] == 'evaluation_only'
    assert len(Counter(c.group for c in cases)) == 8
    assert set(Counter(c.group for c in cases).values()) == {8}
    texts = {c.text for c in cases}
    assert not texts & {r['text'] for r in load_bank()['examples']}
    old = json.loads(DATASET.with_name('intent_cases.json').read_text(encoding='utf-8'))
    assert not texts & {r['text'] for r in old['cases']}
    from scripts.evaluate_candidate_routing import CASES
    assert not texts & {text for _, text, _ in CASES}
    assert {c.operation for c in cases} == {'new', 'change', 'answer', 'cancel', 'clarify', 'direct', 'boundary'}


@pytest.mark.parametrize('fields', [
    {'actions': ['appointment_action']},
    {'actions': ['search_doctors', 'search_doctors']},
    {'forbidden_actions': ['search_doctors']},
    {'arguments': {'search_doctors': {'specification': '0.2g'}}},
    {'arguments': {'search_doctors': {'department': 123}}},
    {'local_eligible': True, 'operation': 'change'},
    {'local_eligible': True, 'forbid_local_plan': True},
])
def test_business_labels_validate_catalog_and_orthogonal_operations(fields):
    with pytest.raises((ValidationError, ValueError)):
        BusinessCase.model_validate({'id': 'case', 'group': 'doctor', 'text': '查询医生',
            'operation': 'new', 'actions': ['search_doctors'], **fields})


def proposal(department='神经内科'):
    return PlanProposal(goal='query', tasks=[TaskSpec(id='doctor', fragment='query',
        agent=AgentType.APPOINTMENT, action='search_doctors', arguments={'department': department}, completion='items')])


def test_scoring_checks_arguments_and_unhandled_conditions_not_just_action_names():
    case = BusinessCase(id='case', group='doctor', text='query', operation='new', actions=['search_doctors'],
                        arguments={'search_doctors': {'department': '神经内科'}}, local_eligible=True)
    assert score_local(case, proposal())['raw_annotation_match']
    assert score_local(case, proposal('消化内科'))['raw_annotation_mismatch']
    constrained = case.model_copy(update={'local_eligible': False, 'forbid_local_plan': True})
    assert score_local(constrained, proposal())['unsafe_local_plan']
    unresolved = score_local(constrained, None)
    assert not unresolved['raw_annotation_mismatch'] and not unresolved['raw_annotation_match']


def test_inference_receives_only_queries_and_labels_do_not_generate_candidates(tmp_path):
    text = '神经内科坐诊医生的信息请展示一下'
    case = {'id': 'case', 'group': 'doctor', 'text': text, 'operation': 'new', 'actions': ['search_doctors'],
            'arguments': {'search_doctors': {'department': '神经内科'}}, 'local_eligible': True}
    target = tmp_path/'cases.json'
    target.write_text(json.dumps({'version': 'business-intent-eval-v1', 'split': 'evaluation_only',
        'source_at': '2026-11-05T10:00:00+08:00', 'cases': [case]}, ensure_ascii=False), encoding='utf-8')
    class Classifier:
        def __init__(self, label):
            self.label, self.inputs = label, []

        def predict(self, query):
            assert isinstance(query, str) and query == text
            self.inputs.append(query)
            return prediction(label=self.label, confidence=.99)
    good, wrong = Classifier('appointment_action'), Classifier('其他')
    positive = asyncio.run(evaluate(good, dataset_path=target))
    negative = asyncio.run(evaluate(wrong, dataset_path=target))
    assert good.inputs == wrong.inputs == [text]
    assert positive['summary']['rules_classifier']['candidate_plans'] == 1
    assert negative['summary']['rules_classifier']['candidate_plans'] == 0
    assert positive['summary']['rules_only']['local_plans'] == negative['summary']['rules_only']['local_plans'] == 0
    assert positive['provider_requests'] == positive['planner_calls'] == positive['tools_executed'] == 0
    assert positive['summary']['rules_semantic']['status'] == 'not_run_embedding_unavailable'


def test_no_model_runs_rules_only_and_does_not_fabricate_classifier_metrics():
    report = asyncio.run(evaluate())
    assert report['summary']['rules_classifier']['status'] == 'not_run_classifier_unavailable'
    assert report['classifier_diagnostic']['raw_family_target_covered'] is None
    assert report['classifier_diagnostic']['rows'] == []


def test_live_ablation_has_a_separate_semantic_only_arm_without_offline_network(monkeypatch):
    from scripts import evaluate_intent_ablation as script
    assert 'rules_embedding_llm' in script.VARIANTS
    def forbidden(*args, **kwargs):
        pytest.fail('Offline evaluation must not construct a provider client')
    monkeypatch.setattr(script, 'create_llm_client', forbidden)
    monkeypatch.setattr(script, 'intent_semantic_retriever', forbidden)
    report = asyncio.run(script.evaluate('offline'))
    assert set(report['summary']) == {'rules_only'}


def test_runtime_can_disable_classifier_for_semantic_only_comparison(monkeypatch, tmp_path):
    import os
    from careguide import runtime as factory
    for key in tuple(os.environ):
        if key.startswith('CAREGUIDE_'):
            monkeypatch.delenv(key)
    monkeypatch.setenv('CAREGUIDE_BUSINESS_DB', str(tmp_path/'factory.sqlite3'))
    monkeypatch.setenv('CAREGUIDE_CORPUS_DIR', str(DATASET.parents[2]/'data/medical_v1'))
    monkeypatch.setenv('CAREGUIDE_HOSPITAL_BACKEND', 'memory')
    try:
        for mode, missing in [('disabled', True), ('auto', False)]:
            monkeypatch.setenv('CAREGUIDE_INTENT_CLASSIFIER_MODE', mode)
            factory.get_registry.cache_clear()
            factory.get_chat_runtime.cache_clear()
            local = factory.get_chat_runtime()[0].understanding
            assert (local.classifier is None) == missing
        monkeypatch.setenv('CAREGUIDE_INTENT_CLASSIFIER_MODE', 'typo')
        factory.get_chat_runtime.cache_clear()
        with pytest.raises(ValueError, match='classifier mode'):
            factory.get_chat_runtime()
    finally:
        factory.get_registry.cache_clear()
        factory.get_chat_runtime.cache_clear()
