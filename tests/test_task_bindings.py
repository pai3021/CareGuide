import asyncio
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace

import pytest

from careguide.agent_orchestrator import AgentOrchestrator
from careguide.corpus import Corpus
from careguide.intent_recognizer import IntentRecognizer
from careguide.orchestration.bindings import InvalidResultReference, resolve_inputs
from careguide.orchestration.budget import BudgetLedger
from careguide.orchestration.contracts import PlanProposal, TaskStatus
from careguide.orchestration.store import WorkflowStore
from careguide.orchestration.worker import TaskWorker
from careguide.retrieval.knowledge_base import MedicalKnowledgeBase
from careguide.tools import MedicalToolRegistry


TEXT = '查消化内科医生，再选择医生查号源'


def waiting_selection(tmp_path):
    doctor = {'id': 'doctor', 'fragment': TEXT, 'action': 'search_doctors',
              'agent': 'appointment', 'completion': 'items', 'arguments': {'department': '消化内科'}}
    slot = {'id': 'slot', 'fragment': TEXT, 'action': 'list_available_slots',
            'agent': 'appointment', 'completion': 'items', 'depends_on': ['doctor'],
            'inputs': {'practitioner_id': {'task_id': 'doctor', 'path': ['items'],
                'mode': 'selected', 'item_field': 'id', 'result_version': 1}}}
    store = WorkflowStore(tmp_path/'workflow.sqlite3')
    state = store.create('u', 'c', PlanProposal(goal=TEXT, tasks=[doctor, slot]), source_text=TEXT)
    parent = state.tasks['doctor']
    parent.status = TaskStatus.SUCCEEDED
    parent.result = {'items': [{'id': 'PRA-1'}, {'id': 'PRA-2'}]}
    return state, state.tasks['slot']


def test_candidate_list_is_not_a_user_selection_even_if_only_one_candidate(tmp_path):
    state, task = waiting_selection(tmp_path)
    state.tasks['doctor'].result['items'] = [{'id': 'PRA-1'}]
    bound = resolve_inputs(task, state)
    assert bound.missing_fields == ['practitioner_id']
    assert 'practitioner_id' not in bound.arguments
    assert bound.options[0]['values'] == [{'id': 'PRA-1'}]


def test_explicit_selection_is_bound_to_current_candidate_membership(tmp_path):
    state, task = waiting_selection(tmp_path)
    task.arguments = {'practitioner_id': 'PRA-2'}
    task.origins = {'practitioner_id': {'kind': 'user_selection', 'event_id': 'user-event'}}
    bound = resolve_inputs(task, state)
    assert not bound.missing_fields
    assert bound.arguments['practitioner_id'] == 'PRA-2'
    assert bound.origins['practitioner_id']['result_version'] == 1
    task.arguments['practitioner_id'] = 'PRA-NOT-IN-RESULT'
    assert resolve_inputs(task, state).missing_fields == ['practitioner_id']


def test_old_result_version_cannot_bind_a_new_task(tmp_path):
    state, task = waiting_selection(tmp_path)
    state.tasks['doctor'].revision = 2
    with pytest.raises(InvalidResultReference, match='revision changed'):
        resolve_inputs(task, state)


def test_expired_slot_or_doctor_result_requires_refresh(tmp_path):
    state, task = waiting_selection(tmp_path)
    state.tasks['doctor'].result_valid_until = (
        datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()
    with pytest.raises(InvalidResultReference, match='expired'):
        resolve_inputs(task, state)


def test_upstream_candidate_collection_cannot_bind_as_scalar(tmp_path):
    state, task = waiting_selection(tmp_path)
    task.spec.inputs['practitioner_id'].mode = 'value'
    with pytest.raises(InvalidResultReference, match='not a confirmed scalar'):
        resolve_inputs(task, state)


def test_model_suggestion_does_not_become_confirmed_scope(tmp_path):
    state, task = waiting_selection(tmp_path)
    task.spec.inputs = {}
    task.spec.arguments = {'practitioner_id': 'PRA-2'}
    bound = resolve_inputs(task, state)
    assert bound.missing_fields == ['practitioner_id']
    assert 'practitioner_id' not in bound.arguments
    assert bound.origins['practitioner_id']['confirmed'] is False


def test_two_medication_tasks_keep_independent_context_and_evidence(tmp_path):
    products = [
        ('精氨酸布洛芬片', '片剂', '0.2g'),
        ('富马酸酮替芬鼻喷雾剂', '鼻喷雾剂', '15ml:16.7mg')]
    fragments = [name+spec+'说明书' for name, form, spec in products]
    text = '，同时'.join(fragments)
    tasks = [{'id': 'drug'+str(index), 'fragment': fragment, 'action': 'lookup_drug_label',
        'agent': 'medication', 'completion': 'source_evidence', 'arguments': {
            'drug_name': name, 'dosage_form': form, 'specification': spec}}
        for index, ((name, form, spec), fragment) in enumerate(zip(products, fragments))]
    state = WorkflowStore(tmp_path/'workflow.sqlite3').create('u', 'c',
        PlanProposal(goal=text, tasks=tasks), source_text=text)
    calls = []
    async def create(**kwargs):
        calls.append(kwargs)
        if kwargs['messages'][-1]['role'] == 'tool':
            return {'choices': [{'message': {'content': 'untrusted narrative is ignored'}}]}
        payload = json.loads(kwargs['messages'][-1]['content'])
        assert len(kwargs['tools']) == 1
        return {'choices': [{'message': {'tool_calls': [{'id': 'call-'+str(len(calls)),
            'function': {'name': 'lookup_drug_label', 'arguments': json.dumps(payload['entities'])}}]}}]}
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    registry = MedicalToolRegistry(MedicalKnowledgeBase(Corpus()))
    engine = AgentOrchestrator(IntentRecognizer(), registry, client=client)
    worker = TaskWorker(engine)
    budget = BudgetLedger(tmp_path/'workflow.sqlite3', state.id, 'u', 'c')
    async def run():
        return await asyncio.gather(*(worker.run(task, state, budget) for task in state.tasks.values()))
    outcomes = asyncio.run(run())
    assert all(outcome.status == TaskStatus.SUCCEEDED for outcome in outcomes)
    for outcome, (name, _, _) in zip(outcomes, products):
        assert outcome.data['evidence']
        assert {e['scope']['drug_name'] for e in outcome.data['evidence']} == {name}
        assert outcome.traces[0]['failure'] is None
    assert budget.summary()['used']['tool'] == 2
    assert budget.summary()['used']['model'] == 4
    initial_payloads = [json.loads(call['messages'][-1]['content']) for call in calls
                        if call['messages'][-1]['role'] == 'user']
    assert len(initial_payloads) == 2
    assert all(payload['message'] in fragments for payload in initial_payloads)
