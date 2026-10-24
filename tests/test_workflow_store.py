from datetime import timedelta

import pytest

from careguide.orchestration.contracts import PlanChange, PlanProposal, TaskStatus
from careguide.orchestration.store import VersionConflict, WorkflowStore, utc_now
from tests.test_orchestration_contracts import TEXT, doctor, slot


def create(tmp_path):
    store = WorkflowStore(tmp_path/'workflow.sqlite3')
    drug = {'id': 'drug', 'fragment': '查询两个药品说明书', 'agent': 'medication',
            'action': 'lookup_drug_label', 'completion': 'source_evidence'}
    state = store.create('alice', 'conversation', PlanProposal(
        goal=TEXT, tasks=[doctor(), slot(), drug]), source_text=TEXT)
    return store, store.begin_run(state.id, 'alice', 'conversation', expected_version=1)


def start_task(store, state, task_id):
    task = state.tasks[task_id]
    task.transition(TaskStatus.READY, at=utc_now().isoformat())
    task.transition(TaskStatus.RUNNING, at=utc_now().isoformat())
    return store.save(state, expected_revision=state.revision)


def complete(store, state, task_id, result):
    state = start_task(store, state, task_id)
    assert store.commit_result(state.id, state.actor, state.conversation,
        task_id, result, TaskStatus.SUCCEEDED, plan_version=state.version,
        task_revision=state.tasks[task_id].revision, run_id=state.active_run)
    return store.get(state.id, state.actor, state.conversation)


def change_department():
    changed = doctor()
    changed['arguments'] = {'department': '心内科'}
    changed['fragment'] = '改成心内科'
    return PlanChange(base_version=1, reason='用户修改科室', update=[changed])


def test_reopening_store_preserves_completed_result_and_identity(tmp_path):
    store, state = create(tmp_path)
    state = complete(store, state, 'drug', {'evidence': [{'record_id': 'saved'}]})
    reopened = WorkflowStore(store.path).get(state.id, 'alice', 'conversation')
    assert reopened.tasks['drug'].status == TaskStatus.SUCCEEDED
    assert reopened.tasks['drug'].result['evidence'][0]['record_id'] == 'saved'
    assert store.latest('bob', 'conversation') is None
    with pytest.raises(LookupError):
        store.get(state.id, 'bob', 'conversation')
    with pytest.raises(LookupError):
        store.get(state.id, 'alice', 'other-conversation')


def test_optimistic_state_save_prevents_lost_updates(tmp_path):
    store, state = create(tmp_path)
    first = store.get(state.id, 'alice', 'conversation')
    second = store.get(state.id, 'alice', 'conversation')
    first.events.append({'event': 'first'})
    store.save(first, expected_revision=first.revision)
    second.events.append({'event': 'second'})
    with pytest.raises(VersionConflict):
        store.save(second, expected_revision=second.revision)
    assert store.get(state.id, 'alice', 'conversation').events == [{'event': 'first'}]


def test_local_replanning_invalidates_only_affected_results_and_descendants(tmp_path):
    store, state = create(tmp_path)
    state = complete(store, state, 'drug', {'evidence': ['kept']})
    state = complete(store, state, 'doctors', {'items': [{'id': 'old-doctor'}]})
    updated = store.apply_change(state.id, 'alice', 'conversation',
        change_department(), user_text='改成心内科')
    assert updated.version == 2
    assert updated.tasks['drug'].revision == 1
    assert updated.tasks['drug'].result == {'evidence': ['kept']}
    assert updated.tasks['doctors'].revision == 2
    assert updated.tasks['doctors'].result is None
    assert updated.tasks['slots'].revision == 2
    assert updated.tasks['slots'].spec.inputs['practitioner_id'].result_version == 2
    archived = next(item for item in updated.archived_results if item['task_id'] == 'doctors')
    assert archived['record']['result']['items'][0]['id'] == 'old-doctor'


def test_late_old_result_is_archived_without_unlocking_new_dependencies(tmp_path):
    store, state = create(tmp_path)
    state = start_task(store, state, 'doctors')
    old_run = state.active_run
    store.apply_change(state.id, 'alice', 'conversation', change_department(), user_text='改成心内科')
    committed = store.commit_result(state.id, 'alice', 'conversation', 'doctors',
        {'items': [{'id': 'late-old-doctor'}]}, TaskStatus.SUCCEEDED,
        plan_version=1, task_revision=1, run_id=old_run)
    assert committed is False
    updated = store.get(state.id, 'alice', 'conversation')
    assert updated.tasks['doctors'].result is None
    assert updated.tasks['slots'].status == TaskStatus.WAITING_DEPENDENCIES
    assert updated.archived_results[-1]['discarded']


def test_stale_change_is_rejected_without_modifying_current_plan(tmp_path):
    store, state = create(tmp_path)
    updated = store.apply_change(state.id, 'alice', 'conversation',
        change_department(), user_text='改成心内科')
    with pytest.raises(VersionConflict):
        store.apply_change(state.id, 'alice', 'conversation',
            change_department(), user_text='改成心内科')
    assert store.get(state.id, 'alice', 'conversation').revision == updated.revision


def test_repeated_identical_plan_is_no_progress(tmp_path):
    store, state = create(tmp_path)
    with pytest.raises(ValueError, match='No progress'):
        store.apply_change(state.id, 'alice', 'conversation', PlanChange(
            base_version=1, reason='重复计划', update=[doctor()]), user_text=TEXT)


def test_cancellation_propagates_to_dependent_branch_only(tmp_path):
    store, state = create(tmp_path)
    state = complete(store, state, 'drug', {'evidence': ['kept']})
    updated = store.apply_change(state.id, 'alice', 'conversation', PlanChange(
        base_version=1, reason='用户取消医生查询', cancel=['doctors']), user_text='取消医生查询')
    assert updated.tasks['doctors'].status == TaskStatus.CANCELLED
    assert updated.tasks['slots'].status == TaskStatus.CANCELLED
    assert updated.tasks['drug'].status == TaskStatus.SUCCEEDED


def test_completed_write_cannot_be_erased_by_cancel_or_replanning(tmp_path):
    store = WorkflowStore(tmp_path/'workflow.sqlite3')
    task = {'id': 'booking', 'fragment': '预约 SLOT-1', 'agent': 'appointment',
        'action': 'prepare_appointment', 'arguments': {'slot_id': 'SLOT-1'}, 'completion': 'prepared'}
    state = store.create('alice', 'c', PlanProposal(goal='预约 SLOT-1', tasks=[task]),
                         source_text='预约 SLOT-1')
    state = store.begin_run(state.id, 'alice', 'c', expected_version=1)
    state = complete(store, state, 'booking', {'status': 'completed', 'appointment_id': 'APT-1'})
    with pytest.raises(PermissionError, match='new confirmed action'):
        store.apply_change(state.id, 'alice', 'c', PlanChange(
            base_version=1, reason='删除任务', cancel=['booking']), user_text='删除任务')
    assert store.get(state.id, 'alice', 'c').tasks['booking'].result['appointment_id'] == 'APT-1'


def test_live_lease_prevents_parallel_request_takeover(tmp_path):
    store, state = create(tmp_path)
    with pytest.raises(VersionConflict, match='already active'):
        store.begin_run(state.id, 'alice', 'conversation', expected_version=1)
    store.end_run(state.id, 'alice', 'conversation', state.active_run, status='waiting_user')
    reopened = store.begin_run(state.id, 'alice', 'conversation', expected_version=1)
    assert reopened.active_run != state.active_run


@pytest.mark.parametrize('replacement', ['removed', 'each'])
def test_replanning_cannot_broaden_selected_doctor_without_specific_authorization(tmp_path, replacement):
    store, state = create(tmp_path)
    state.tasks['slots'].spec.inputs['practitioner_id'].mode = 'selected'
    state = store.save(state, expected_revision=state.revision)
    proposed = state.tasks['slots'].spec.model_copy(deep=True)
    if replacement == 'removed':
        proposed.inputs = {}
    else:
        proposed.inputs['practitioner_id'].mode = 'each'
    with pytest.raises(PermissionError, match='authorization'):
        store.apply_change(state.id, 'alice', 'conversation', PlanChange(
            base_version=1, reason='扩大查询范围', update=[proposed]),
            user_text='请重新查询所有药品资料')
    current = store.get(state.id, 'alice', 'conversation')
    assert current.version == 1
    assert current.tasks['slots'].spec.inputs['practitioner_id'].mode == 'selected'


def test_expired_run_cannot_commit_or_be_resumed(tmp_path, monkeypatch):
    store, state = create(tmp_path)
    state = start_task(store, state, 'doctors')
    later = utc_now()+timedelta(hours=1)
    monkeypatch.setattr('careguide.orchestration.store.utc_now', lambda: later)
    assert not store.commit_result(state.id, 'alice', 'conversation', 'doctors',
        {'items': ['too-late']}, TaskStatus.SUCCEEDED, plan_version=1,
        task_revision=1, run_id=state.active_run)
    with pytest.raises(ValueError, match='expired'):
        store.begin_run(state.id, 'alice', 'conversation', expected_version=1)


def test_new_dependent_task_binds_the_changed_parent_revision(tmp_path):
    store, state = create(tmp_path)
    changed = change_department()
    additional = slot()
    additional.update(id='new_slots', fragment='新增心内科所有医生号源')
    additional['inputs']['practitioner_id']['result_version'] = 1
    changed = PlanChange(base_version=changed.base_version, reason=changed.reason,
                         update=changed.update, add=[additional])
    updated = store.apply_change(state.id, 'alice', 'conversation', changed,
                                 user_text='改成心内科，新增心内科所有医生号源')
    assert updated.tasks['doctors'].revision == 2
    assert updated.tasks['new_slots'].spec.inputs['practitioner_id'].result_version == 2
