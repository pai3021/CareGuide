import pytest
from pydantic import ValidationError

from careguide.orchestration.contracts import (PlanProposal, TaskRecord,
    TaskStatus, validate_plan)


TEXT = '查消化内科全部医生和号源，并查询两个药品说明书'


def doctor(identity='doctors'):
    return {'id': identity, 'fragment': '查消化内科全部医生和号源',
            'action': 'search_doctors', 'agent': 'appointment',
            'arguments': {'department': '消化内科'}, 'completion': 'items'}


def slot():
    return {'id': 'slots', 'fragment': '查消化内科全部医生和号源',
            'action': 'list_available_slots', 'agent': 'appointment',
            'completion': 'items', 'depends_on': ['doctors'], 'inputs': {
                'practitioner_id': {'task_id': 'doctors', 'path': ['items'],
                                    'mode': 'each', 'item_field': 'id'}}}


def proposal(tasks):
    return PlanProposal(goal=TEXT, tasks=tasks)


def test_dag_preserves_explicit_output_binding_and_is_detached_from_model_input():
    original = proposal([doctor(), slot()])
    validated = validate_plan(original, TEXT)
    assert validated.tasks[1].inputs['practitioner_id'].task_id == 'doctors'
    original.tasks[0].arguments['department'] = '模型随后篡改的条件'
    assert validated.tasks[0].arguments['department'] == '消化内科'


def test_duplicate_task_ids_are_rejected_before_execution():
    with pytest.raises(ValueError, match='Duplicate task'):
        validate_plan(proposal([doctor(), doctor()]), TEXT)


def test_unknown_dependency_is_rejected():
    task = doctor()
    task['depends_on'] = ['missing']
    with pytest.raises(ValueError, match='Unknown task dependency'):
        validate_plan(proposal([task]), TEXT)


def test_dependency_cycle_is_rejected():
    task = doctor()
    task['depends_on'] = ['slots']
    with pytest.raises(ValueError, match='cycle'):
        validate_plan(proposal([task, slot()]), TEXT)


def test_task_cannot_use_another_roles_tool():
    task = doctor()
    task['agent'] = 'medication'
    with pytest.raises(PermissionError):
        validate_plan(proposal([task]), TEXT)


def test_planner_cannot_request_direct_confirmation_or_write_execution():
    task = doctor()
    task['action'] = 'confirm_appointment'
    with pytest.raises(PermissionError):
        validate_plan(proposal([task]), TEXT)


def test_input_must_declare_its_dependency():
    task = slot()
    task['depends_on'] = []
    with pytest.raises(ValueError, match='explicit dependency'):
        validate_plan(proposal([doctor(), task]), TEXT)


def test_model_cannot_treat_first_candidate_as_user_selection():
    task = slot()
    task['inputs']['practitioner_id'] = {'task_id': 'doctors', 'path': ['items', '0', 'id']}
    with pytest.raises(ValueError, match='candidate indices'):
        validate_plan(proposal([doctor(), task]), TEXT)


def test_fanout_requires_explicit_user_scope():
    task = slot()
    task['fragment'] = '查医生'
    parent = doctor()
    parent['fragment'] = '查医生'
    with pytest.raises(PermissionError, match='authorization'):
        validate_plan(proposal([parent, task]), '查医生和号源')


def test_all_drug_queries_cannot_authorize_all_doctor_slot_fanout():
    task, parent = slot(), doctor()
    text = '查所有药品说明书，同时查消化内科医生的号源'
    task['fragment'] = parent['fragment'] = text
    with pytest.raises(PermissionError, match='authorization'):
        validate_plan(proposal([parent, task]), text)


def test_missing_user_argument_can_become_a_waiting_task():
    task = {'id': 'drug', 'fragment': '查询两个药品说明书', 'action': 'lookup_drug_label',
            'agent': 'medication', 'arguments': {}, 'completion': 'source_evidence'}
    assert validate_plan(proposal([task]), TEXT).tasks[0].arguments == {}


@pytest.mark.parametrize('bad_arguments', [{'patient_id': 'another-user'}, {'department': 17}])
def test_parameter_names_and_types_are_checked(bad_arguments):
    task = doctor()
    task['arguments'] = bad_arguments
    with pytest.raises((ValueError, ValidationError)):
        validate_plan(proposal([task]), TEXT)


def test_changed_task_fragment_cannot_add_a_new_user_goal():
    task = doctor()
    task['fragment'] = '改查其他医院'
    with pytest.raises(ValueError, match='original user request'):
        validate_plan(proposal([task]), TEXT)


def test_plan_scale_is_checked_against_runtime_budget():
    with pytest.raises(ValueError, match='budget'):
        validate_plan(proposal([doctor(), slot()]), TEXT, max_tasks=1)


def test_task_state_machine_preserves_result_when_invalidated():
    spec = validate_plan(proposal([doctor()]), TEXT).tasks[0]
    record = TaskRecord(spec=spec)
    record.transition(TaskStatus.READY, at='t1')
    record.transition(TaskStatus.RUNNING, at='t2')
    record.result = {'items': [{'id': 'PRA-OLD'}]}
    record.transition(TaskStatus.SUCCEEDED, at='t3')
    record.transition(TaskStatus.INVALIDATED, at='t4', reason='department changed')
    assert record.result['items'][0]['id'] == 'PRA-OLD'
    assert len(record.history) == 4
    with pytest.raises(ValueError, match='Invalid task transition'):
        record.transition(TaskStatus.RUNNING, at='t5')
