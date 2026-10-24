import asyncio
from datetime import timedelta

import pytest

from careguide.action_store import ActionStore
from careguide.agent_orchestrator import AgentOrchestrator
from careguide.appointment_service import AppointmentService
from careguide.corpus import Corpus
from careguide.hospital_workflow import InMemoryHospitalBackend
from careguide.intent_recognizer import IntentRecognizer
from careguide.orchestration.budget import BudgetLedger, BudgetLimits
from careguide.orchestration.contracts import PlanProposal, TaskStatus
from careguide.orchestration.replies import compose_workflow
from careguide.orchestration.scheduler import DAGScheduler
from careguide.orchestration.store import WorkflowStore
from careguide.orchestration.store import utc_now
from careguide.orchestration.worker import TaskWorker
from careguide.retrieval.knowledge_base import MedicalKnowledgeBase
from careguide.tools import MedicalToolRegistry


def setup(tmp_path, text, tasks, *, backend=None, limits=None):
    path = tmp_path/'runtime.sqlite3'
    store = WorkflowStore(path)
    state = store.create('u', 'c', PlanProposal(goal=text, tasks=tasks), source_text=text)
    state = store.begin_run(state.id, 'u', 'c', expected_version=1)
    registry = MedicalToolRegistry(MedicalKnowledgeBase(Corpus()))
    engine = AgentOrchestrator(IntentRecognizer(), registry,
        appointment_service=AppointmentService(backend or InMemoryHospitalBackend(), ActionStore(path)))
    budget = BudgetLedger(path, state.id, 'u', 'c', limits)
    return store, state, engine, budget


def drug(fragment='查询精氨酸布洛芬片0.2g说明书', *, specification=True):
    args = {'drug_name': '精氨酸布洛芬片', 'dosage_form': '片剂'}
    if specification:
        args['specification'] = '0.2g'
    return {'id': 'drug', 'fragment': fragment, 'action': 'lookup_drug_label',
            'agent': 'medication', 'arguments': args, 'completion': 'source_evidence'}


def doctors(fragment='查消化内科医生'):
    return {'id': 'doctors', 'fragment': fragment, 'action': 'search_doctors',
            'agent': 'appointment', 'arguments': {'department': '消化内科'}, 'completion': 'items'}


def test_downstream_starts_after_its_own_parent_without_waiting_for_slow_independent_task(tmp_path):
    text = '查消化内科所有医生和号源，同时查询精氨酸布洛芬片0.2g说明书'
    parent = doctors('查消化内科所有医生和号源')
    slots = {'id': 'slots', 'fragment': parent['fragment'], 'action': 'list_available_slots',
        'agent': 'appointment', 'depends_on': ['doctors'], 'completion': 'items',
        'inputs': {'practitioner_id': {'task_id': 'doctors', 'path': ['items'],
                                      'mode': 'each', 'item_field': 'id'}}}

    async def run():
        downstream_started = asyncio.Event()
        queried_ids = []
        class Backend(InMemoryHospitalBackend):
            async def list_available_slots(self, **kwargs):
                queried_ids.append(kwargs['practitioner_id'])
                downstream_started.set()
                return await super().list_available_slots(**kwargs)
        backend = Backend()
        store, state, engine, budget = setup(tmp_path, text, [parent, slots, drug()], backend=backend)
        tool = engine.tool_manager._tools['medication:lookup_drug_label']
        original = tool.handler
        async def slow_independent(params, context):
            await asyncio.wait_for(downstream_started.wait(), timeout=2)
            return await original(params, context)
        tool.handler = slow_independent
        final = await DAGScheduler(store, TaskWorker(engine)).run(state, budget)
        assert all(task.status == TaskStatus.SUCCEEDED for task in final.tasks.values())
        expected = {item['id'] for item in final.tasks['doctors'].result['items']}
        assert set(queried_ids) == expected
        assert {item['practitioner_id'] for item in final.tasks['slots'].result['items']} <= expected
    asyncio.run(run())


def test_wait_then_reopen_and_resume_does_not_repeat_independent_completed_query(tmp_path):
    text = '查消化内科医生，同时查询精氨酸布洛芬片说明书'
    store, state, engine, budget = setup(tmp_path, text,
        [doctors(), drug('查询精氨酸布洛芬片说明书', specification=False)])
    async def run():
        scheduler = DAGScheduler(store, TaskWorker(engine))
        first = await scheduler.run(state, budget)
        assert first.tasks['doctors'].status == TaskStatus.SUCCEEDED
        assert first.tasks['drug'].status == TaskStatus.WAITING_INPUT
        assert first.tasks['drug'].missing_fields == ['specification']
        store.end_run(first.id, 'u', 'c', first.active_run, status='waiting_user')
        reopened = WorkflowStore(store.path)
        answered = reopened.answer(first.id, 'u', 'c', 'drug', {'specification': '0.2g'},
            plan_version=1, task_revision=1, event_id='reply-spec')
        resumed = reopened.begin_run(answered.id, 'u', 'c', expected_version=1)
        final = await DAGScheduler(reopened, TaskWorker(engine)).run(resumed, budget)
        assert all(task.status == TaskStatus.SUCCEEDED for task in final.tasks.values())
        assert final.tasks['doctors'].attempts == 1
        assert final.tasks['drug'].attempts == 1
        assert budget.summary()['used']['tool'] == 2
        repeated = reopened.answer(final.id, 'u', 'c', 'drug', {'specification': '0.2g'},
            plan_version=1, task_revision=1, event_id='reply-spec')
        assert repeated.tasks['drug'].status == TaskStatus.SUCCEEDED
    asyncio.run(run())


def test_transient_failure_retries_only_failed_branch_and_delivers_other_results(tmp_path):
    text = '查消化内科医生，同时查询精氨酸布洛芬片0.2g说明书'
    store, state, engine, budget = setup(tmp_path, text, [doctors(), drug()])
    async def unavailable(params, context):
        raise OSError('synthetic tool outage')
    engine.tool_manager._tools['medication:lookup_drug_label'].handler = unavailable
    final = asyncio.run(DAGScheduler(store, TaskWorker(engine)).run(state, budget))
    assert final.tasks['doctors'].status == TaskStatus.SUCCEEDED
    assert final.tasks['doctors'].attempts == 1
    assert final.tasks['drug'].status == TaskStatus.FAILED
    assert final.tasks['drug'].attempts == 2
    assert budget.summary()['used']['retry'] == 1
    reply = compose_workflow(final, budget)
    assert reply['workflow_status'] == 'partial_completed'
    assert reply['workflow']['tasks'][0]['result']['items']
    assert reply['errors'][0]['code'] == 'tool_unavailable'


def test_parallel_branches_share_hard_tool_limit_and_keep_completed_result(tmp_path):
    text = '查消化内科医生，同时查询精氨酸布洛芬片0.2g说明书'
    store, state, engine, budget = setup(tmp_path, text, [doctors(), drug()], limits=BudgetLimits(tool=1))
    final = asyncio.run(DAGScheduler(store, TaskWorker(engine)).run(state, budget))
    assert budget.summary()['used']['tool'] == 1
    assert sum(task.status == TaskStatus.SUCCEEDED for task in final.tasks.values()) == 1
    assert any(task.failure and task.failure['code'] == 'budget_exhausted' for task in final.tasks.values())
    assert compose_workflow(final, budget)['workflow_status'] == 'partial_completed'


@pytest.mark.parametrize('refresh_budget', [0, 1])
def test_source_expiring_after_answer_never_dispatches_old_selected_doctor(tmp_path, refresh_budget):
    text = '查消化内科医生，再选择医生查号源'
    parent = doctors(text)
    slots = {'id': 'slots', 'fragment': text, 'action': 'list_available_slots',
        'agent': 'appointment', 'depends_on': ['doctors'], 'completion': 'items',
        'inputs': {'practitioner_id': {'task_id': 'doctors', 'path': ['items'],
                                      'mode': 'selected', 'item_field': 'id'}}}
    slot_calls = []
    class Backend(InMemoryHospitalBackend):
        async def list_available_slots(self, **kwargs):
            slot_calls.append(kwargs)
            return await super().list_available_slots(**kwargs)
    store, state, engine, budget = setup(tmp_path, text, [parent, slots],
        backend=Backend(), limits=BudgetLimits(retry=refresh_budget))
    async def run():
        first = await DAGScheduler(store, TaskWorker(engine)).run(state, budget)
        assert first.tasks['slots'].status == TaskStatus.WAITING_INPUT
        selected = first.tasks['doctors'].result['items'][0]['id']
        store.end_run(first.id, 'u', 'c', first.active_run, status='waiting_user')
        answered = store.answer(first.id, 'u', 'c', 'slots', {'practitioner_id': selected},
            plan_version=1, task_revision=1, event_id='selected-before-expiry')
        assert answered.tasks['slots'].status == TaskStatus.READY
        answered.tasks['doctors'].result_valid_until = (utc_now()-timedelta(seconds=1)).isoformat()
        store.save(answered, expected_revision=answered.revision)
        resumed = store.begin_run(first.id, 'u', 'c', expected_version=1)
        final = await asyncio.wait_for(DAGScheduler(store, TaskWorker(engine)).run(resumed, budget), 3)
        assert slot_calls == []
        assert final.tasks['slots'].status == (TaskStatus.WAITING_INPUT if refresh_budget else TaskStatus.BLOCKED)
        assert budget.summary()['used']['retry'] == refresh_budget
        if refresh_budget:
            assert final.tasks['doctors'].revision == 2
            assert final.tasks['slots'].revision == 2
            assert final.tasks['slots'].missing_fields == ['practitioner_id']
            assert 'practitioner_id' not in final.tasks['slots'].arguments
        else:
            assert final.tasks['slots'].failure['code'] == 'budget_exhausted'
    asyncio.run(run())
