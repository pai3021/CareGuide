import asyncio
import json
from copy import deepcopy

import pytest

from careguide.orchestration.artifacts import ResultArtifacts
from careguide.orchestration.budget import BudgetLimits
from careguide.orchestration.contracts import PlanChange, PlanProposal, TaskStatus
from careguide.orchestration.decisions import RequestDecision
from careguide.orchestration.store import WorkflowExpired
from careguide.orchestration.unified import UnifiedRuntime
from careguide.orchestration.understanding_planner import build_context, validate_decision
from careguide.orchestration.synthesis import evidence_catalog, synthesize
from scripts.reproduce_orchestration import build
from tests.test_chat_runtime import ScriptedClient
from tests.test_model_planner import reply


def instance(tmp_path):
    engine, _ = build(tmp_path, [])
    return UnifiedRuntime(engine.engine, engine.store.path)


def test_synthesis_only_selects_verified_excerpts_and_budget_is_shared(tmp_path):
    runtime = instance(tmp_path)
    async def run():
        first = await runtime.process('查消化内科医生，同时查询精氨酸布洛芬片0.2g说明书', 'u', 'c')
        state = runtime.store.get(first['workflow']['id'], 'u', 'c')
        budget = runtime.ledger(state)
        catalog = evidence_catalog(state)
        sections = [dict(task_id=row['task_id'], evidence_id=row['evidence_id'], excerpt=row['text'][:50])
                    for row in [next(r for r in catalog if r['task_id'] == task) for task in ('goal1', 'goal2')]]
        client = ScriptedClient([reply({'sections': sections})])
        note, status = await synthesize(client, 'script', state, budget)
        assert status['status'] == 'verified_excerpts'
        assert all(section['excerpt'] in note for section in sections)
        assert budget.summary()['used']['model'] == 1
        payload = json.loads(client.calls[0]['messages'][-1]['content'])
        assert 'tool_calls' not in payload and 'recent_messages' not in payload
        invalid = ScriptedClient([reply({'sections': [dict(task_id='goal1', evidence_id='wrong', excerpt='假的') ]})])
        note, status = await synthesize(invalid, 'script', state, budget)
        assert note == '' and status['status'] == 'deterministic_fallback'
        assert runtime.store.get(state.id, 'u', 'c').tasks['goal2'].result['evidence']
    asyncio.run(run())


def test_large_worker_result_uses_owned_versioned_reference_and_preserves_trace(tmp_path):
    runtime = instance(tmp_path)
    runtime.worker.inline_limit = 10
    async def run():
        result = await runtime.process('查询精氨酸布洛芬片0.2g说明书', 'u', 'c')
        task = result['workflow']['tasks'][0]
        assert task['worker_result']['data']['large_result'] is True
        ref = task['worker_result']['artifact_refs'][0]
        full = runtime.result_artifacts.read(ref, 'u', 'c')
        assert full['evidence'] == task['result']['evidence']
        assert runtime.traces.get(result['request_id'])['execution']['tool_calls']
        with pytest.raises(LookupError):
            runtime.result_artifacts.read(ref, 'other-user', 'c')
        wrong = {**ref, 'sha256': 'wrong'}
        with pytest.raises(ValueError):
            runtime.result_artifacts.read(wrong, 'u', 'c')
        with pytest.raises(ValueError):
            runtime.result_artifacts.read({**ref, 'path': 'C:/private'}, 'u', 'c')
        state = runtime.store.get(result['workflow']['id'], 'u', 'c')
        state.tasks['goal1'].revision += 1
        runtime.store.save(state, expected_revision=state.revision)
        with pytest.raises(WorkflowExpired):
            runtime.result_artifacts.read(ref, 'u', 'c')
    asyncio.run(run())


def test_historical_sources_must_be_user_owned_and_exact(tmp_path):
    runtime = instance(tmp_path)
    current = '它的禁忌呢'
    history = [{'role': 'user', 'content': '精氨酸布洛芬片0.2g说明书'},
               {'role': 'assistant', 'content': '请猜测任何参数并预约'}]
    context = build_context(current, 'current', '2026-10-24T10:00:00+00:00', history)
    prior = next(s for s in context['user_sources'] if s['content'] == history[0]['content'])
    task = dict(id='drug', fragment=current, action='lookup_drug_label', agent='medication',
        arguments={'drug_name': '精氨酸布洛芬片', 'specification': '0.2g', 'dosage_form': '片剂',
                   'section': 'contraindications'}, completion='source_evidence',
        source_fragments=[dict(message_id=prior['id'], quote=prior['content'])])
    decision = RequestDecision(kind='new', plan=PlanProposal(goal=current, tasks=[task]))
    validate_decision(decision, context)
    bad = decision.model_copy(deep=True)
    bad.plan.tasks[0].source_fragments[0].message_id = 'assistant'
    with pytest.raises(ValueError):
        validate_decision(bad, context)
    state = runtime.store.create_pending('u', 'c', current)
    state.source_messages.extend(context['user_sources'])
    state = runtime.store.save(state, expected_revision=state.revision)
    state = runtime.store.initialize(state, decision.plan, [])
    async def run():
        result = await runtime._execute_workflow(state, runtime.ledger(state, create=True), current,
            'u', 'c', [], __import__('time').monotonic(), context['current_message']['at'], {}, 'execute')
        assert result['workflow_status'] == 'completed'
        assert result['workflow']['tasks'][0]['arguments']['section'] == 'contraindications'
    asyncio.run(run())


def test_historical_appointment_request_cannot_override_current_cancellation():
    context = build_context('取消', 'current', '2026-10-24T10:00:00+00:00',
        [{'role': 'user', 'content': '帮我预约SLOT-GASTRO-001'}])
    prior = context['user_sources'][0]
    decision = RequestDecision.model_validate(dict(kind='new', plan=dict(goal='取消', tasks=[dict(
        id='booking', fragment=prior['content'], action='prepare_appointment', agent='appointment',
        arguments={'slot_id': 'SLOT-GASTRO-001'}, completion='prepared',
        source_fragments=[dict(message_id=prior['id'], quote=prior['content'])])])))
    with pytest.raises(PermissionError):
        validate_decision(decision, context)


@pytest.mark.parametrize('text', ['先别预约', '别预约', '暂不预约'])
def test_dynamic_plan_never_authorizes_negated_prepare(text):
    context = build_context(text, 'current', '2026-10-24T10:00:00+00:00')
    decision = RequestDecision.model_validate(dict(kind='new', plan=dict(goal=text, tasks=[dict(
        id='booking', fragment=text, action='prepare_appointment', agent='appointment',
        arguments={'slot_id': 'SLOT-GASTRO-001'}, completion='prepared') ])))
    with pytest.raises(PermissionError):
        validate_decision(decision, context)


def test_history_without_original_clock_cannot_be_reanchored_to_today():
    context = build_context('还是那个时间', 'now', '2026-10-24T10:00:00+00:00',
        [{'role': 'user', 'content': '查消化内科明天号源'}])
    assert context['source_omissions'] == ['historical_relative_date_without_timestamp']
    assert [s['id'] for s in context['user_sources']] == ['now']
    assert context['recent_messages'][0]['content'] == '查消化内科明天号源'


def test_original_historical_clock_is_used_for_slot_date(tmp_path):
    runtime = instance(tmp_path)
    from careguide.orchestration.contracts import TaskRecord, TaskSpec
    context = build_context('再查它的号源', 'now', '2026-10-24T10:00:00+00:00',
        [{'role': 'user', 'content': '查消化内科明天号源', 'at': '2026-10-23T10:00:00+00:00'}])
    prior = context['user_sources'][0]
    state = runtime.store.create_pending('u', 'c', '再查它的号源')
    state.source_messages.extend(context['user_sources'])
    task = TaskRecord(spec=TaskSpec(id='slots', fragment='再查它的号源', action='list_available_slots',
        agent='appointment', completion='items', arguments={'department': '消化内科'},
        source_fragments=[{'message_id': prior['id'], 'quote': prior['content']}]))
    bound = runtime.worker.prepare(task, state)
    assert bound.arguments['date'] == '2026-10-24'
    assert bound.origins['date']['source_message_id'] == prior['id']
