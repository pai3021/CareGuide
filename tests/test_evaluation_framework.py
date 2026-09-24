import asyncio
from dataclasses import asdict

from careguide.agent_orchestrator import AgentOrchestrator
from careguide.domain import IntentCategory
from careguide.evaluator import EndToEndEvaluator, IntentEvaluator, IntentTestCase, QualityScores
from careguide.intent_recognizer import IntentResult
from careguide.runtime import get_registry


class ContextRecognizer:
    def __init__(self):
        self.histories = []

    async def recognize(self, message, history=None):
        self.histories.append(history or [])
        if message == '范围外':
            return IntentResult(None, 0, disposition='out_of_scope')
        return IntentResult(IntentCategory.APPOINTMENT, 0.9, [IntentCategory.APPOINTMENT], 'route')


def test_evaluator_handles_boundary_states_and_case_context():
    recognizer = ContextRecognizer()
    cases = [IntentTestCase('范围外', 'out_of_scope'),
             IntentTestCase('材料呢', 'appointment', {'history': [{'role': 'user', 'content': '挂号'}]})]
    report = asyncio.run(IntentEvaluator(recognizer).evaluate(cases))
    assert report['correct'] == 2
    assert recognizer.histories[1][0]['content'] == '挂号'


def test_parallel_evaluation_preserves_case_alignment():
    class DelayedRecognizer(ContextRecognizer):
        async def recognize(self, message, history=None):
            if message == '范围外':
                await asyncio.sleep(0.02)
            return await super().recognize(message, history)
    progress = []
    cases = [IntentTestCase('范围外', 'out_of_scope'), IntentTestCase('挂号', 'appointment')]
    report = asyncio.run(IntentEvaluator(DelayedRecognizer()).evaluate(cases, concurrency=2,
        progress=lambda completed, total: progress.append((completed, total))))
    assert report['correct'] == 2
    assert [item['message'] for item in report['cases']] == ['范围外', '挂号']
    assert progress == [(1, 2), (2, 2)]


def test_multiturn_evaluation_retains_trace_layer_without_fake_judge_score():
    recognizer = ContextRecognizer()
    engine = AgentOrchestrator(recognizer, get_registry())
    evaluator = EndToEndEvaluator(engine, recognizer, '', judge_enabled=False)
    report = asyncio.run(evaluator.run(dialog_cases=[{'turns': ['挂号', '材料呢']}]))
    assert report.total == 2
    assert report.scored == 0 and report.unscored == 2
    assert report.passed == 0 and report.pass_rate is None
    assert not any('所有指标均达标' in item for item in report.recommendations)
    assert recognizer.histories[1]
    for item in report.results:
        assert item.passed is None and item.scores == {}
        assert item.metadata['evaluation_layers']['llm_judge']['failed']
        assert item.metadata['evaluation_layers']['deterministic_trace']['complete']


def test_unscored_dialog_does_not_change_scored_pass_rate():
    recognizer = ContextRecognizer()
    engine = AgentOrchestrator(recognizer, get_registry())
    evaluator = EndToEndEvaluator(engine, recognizer, '', judge_enabled=False)
    report = asyncio.run(evaluator.run(
        intent_cases=[IntentTestCase('挂号', 'appointment')],
        dialog_cases=[{'question': '挂号'}]))
    assert (report.total, report.scored, report.unscored, report.passed) == (2, 1, 1, 1)
    assert report.pass_rate == 1.0
    assert any('未评分' in item for item in report.recommendations)
    assert report.results[1].passed is None
    restored = evaluator._report_from_dict(asdict(report))
    assert restored.results[1].passed is None and restored.pass_rate == 1.0


def test_scored_dialog_still_contributes_to_pass_rate():
    recognizer = ContextRecognizer()
    engine = AgentOrchestrator(recognizer, get_registry())
    evaluator = EndToEndEvaluator(engine, recognizer, '', judge_enabled=False)
    async def scored(*args, **kwargs):
        return QualityScores(0.9, 0.9, 0.9, 0.9)
    evaluator._judge.judge = scored
    report = asyncio.run(evaluator.run(dialog_cases=[{'question': '挂号'}]))
    assert (report.total, report.scored, report.unscored, report.passed) == (1, 1, 0, 1)
    assert report.pass_rate == 1.0 and report.results[0].passed is True
