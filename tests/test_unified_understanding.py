import asyncio
from datetime import datetime, timezone

import pytest

from careguide.corpus import Corpus, DEFAULT_CORPUS
from careguide.orchestration.decisions import RequestDecision
from careguide.orchestration.understanding import LocalUnderstanding, compose_plan


def understand(text, classifier=None):
    return asyncio.run(LocalUnderstanding(Corpus(DEFAULT_CORPUS), classifier).understand(
        text, message_id='message-1', at='2026-10-24T10:00:00+00:00'))


@pytest.mark.parametrize('text,source,count', [
    ('查消化内科医生', 'single', 1),
    ('查消化内科医生，同时查询精氨酸布洛芬片0.2g说明书', 'compose', 2),
    ('查消化内科医生，查询精氨酸布洛芬片0.2g说明书', 'compose', 2),
    ('查消化内科医生的号源并预约', 'template', 3),
    ('查消化内科医生和明天下午的号源', 'template', 2),
    ('查消化内科医生的号源，先别预约', 'template', 2),
    ('查消化内科医生然后告诉我结果', 'single', 1),
])
def test_local_plans_cover_complete_known_requests(text, source, count):
    understanding = understand(text)
    plan, metadata = compose_plan(text, understanding)
    assert metadata['plan_source'] == source
    assert len(plan.tasks) == count
    assert all(t.fragment in text for t in plan.tasks)
    if source == 'compose':
        doctor, drug = plan.tasks
        assert doctor.arguments == {'department': '消化内科'}
        assert drug.arguments['drug_name'] == '精氨酸布洛芬片'
        assert drug.arguments['specification'] == '0.2g'
        assert not doctor.depends_on and not drug.depends_on
    if source == 'template':
        assert plan.tasks[1].inputs['practitioner_id'].mode == 'selected'


@pytest.mark.parametrize('text', [
    '查消化内科医生，同时分析我的检查结果',
    '查消化内科医生和药品的关系',
    '查精氨酸布洛芬片0.2g和酮替芬的禁忌',
    '那个医生，换成下午，再查同一种药',
    '查消化内科医生，同时查询精氨酸布洛芬片0.2g说明书，再帮我预约',
    '查消化内科医生，然后不要查这个科室',
])
def test_unknown_clauses_and_relationships_never_dispatch_partial_plan(text):
    result = understand(text)
    plan, metadata = compose_plan(text, result)
    assert plan is None
    assert metadata['plan_source'] == 'llm'


def test_classifier_is_a_hint_not_multiple_goals_or_execution_permission():
    class Classifier:
        def predict(self, text):
            return {'label': 'appointment_action', 'scores': {'appointment_action': .99, '其他': .01},
                    'margin': .98, 'confidence': .99, 'truncated': True}
    result = understand('这段话不属于任何已知流程，但是有很多想法', Classifier())
    assert result.relation == 'unknown'
    assert result.candidates[-1]['source'] == 'classifier'
    assert result.candidates[-1]['calibrated'] is False
    assert 'classifier_input_truncated' in result.diagnostics
    assert compose_plan('这段话不属于任何已知流程，但是有很多想法', result)[0] is None


def test_classifier_failure_is_explicit_and_local_candidate_fallback_remains():
    class Broken:
        def predict(self, text):
            raise RuntimeError('missing local weights')
    result = understand('有没有别的安排建议', Broken())
    assert result.candidates
    assert result.diagnostics == ['classifier_unavailable:RuntimeError']


def test_incoherent_decision_shapes_are_rejected():
    with pytest.raises(ValueError):
        RequestDecision(kind='answer')
    with pytest.raises(ValueError):
        RequestDecision(kind='clarify', workflow_id='other')


def test_missing_specification_is_preserved_in_local_understanding():
    result = understand('查询精氨酸布洛芬片说明书')
    assert result.goals[0].missing_fields == ['specification']


def test_mismatched_classifier_labels_remain_diagnostic_only():
    class Classifier:
        def predict(self, text):
            return {'label': 'foreign_taxonomy', 'scores': {'foreign_taxonomy': 1}}
    result = understand('我想了解别的安排', Classifier())
    assert 'classifier_unavailable:ValueError' in result.diagnostics
    assert not any(c['source'] == 'classifier' for c in result.candidates)
    assert compose_plan('我想了解别的安排', result)[0] is None
