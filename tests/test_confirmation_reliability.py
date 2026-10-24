import asyncio

import pytest
import httpx

from careguide.action_store import ActionStore
from careguide.appointment_service import AppointmentService
from careguide.hospital_workflow import FHIRHospitalBackend, InMemoryHospitalBackend
from careguide.orchestration.contracts import PlanChange, PlanProposal, TaskStatus
from careguide.orchestration.store import WorkflowStore
from careguide.orchestration.budget import BudgetLedger
from tests.test_workflow_store import start_task


def prepared(tmp_path):
    path = tmp_path/'runtime.sqlite3'
    store = WorkflowStore(path)
    text = '预约 SLOT-CARD-001；查心血管内科医生'
    tasks = [dict(id='booking', fragment='预约 SLOT-CARD-001', agent='appointment',
        action='prepare_appointment', arguments={'slot_id': 'SLOT-CARD-001'}, completion='prepared'),
        dict(id='doctors', fragment='查心血管内科医生', agent='appointment',
             action='search_doctors', arguments={'department': '心血管内科'}, completion='items')]
    state = store.create('alice', 'c', PlanProposal(goal=text, tasks=tasks), source_text=text)
    BudgetLedger(path, state.id, 'alice', 'c')
    state = store.begin_run(state.id, 'alice', 'c', expected_version=1)
    state = start_task(store, state, 'booking')
    task = state.tasks['booking']
    task.arguments = dict(task.spec.arguments)
    state = store.save(state, expected_revision=state.revision)
    service = AppointmentService(InMemoryHospitalBackend(), ActionStore(path))
    action = service.prepare('appointment_create', 'alice', 'c', task.arguments,
        workflow_binding={'workflow_id': state.id, 'plan_version': state.version,
            'task_id': 'booking', 'task_revision': task.revision, 'run_id': state.active_run})
    assert store.commit_result(state.id, 'alice', 'c', 'booking', action,
        TaskStatus.WAITING_CONFIRMATION, plan_version=1, task_revision=1,
        run_id=state.active_run, arguments=task.arguments, action_id=action['id'])
    return store, state, service, action


def test_changed_slot_expires_old_confirmation_before_backend_write(tmp_path):
    store, state, service, action = prepared(tmp_path)
    changed = state.tasks['booking'].spec.model_copy(update={
        'fragment': '改为预约 SLOT-GASTRO-001', 'arguments': {'slot_id': 'SLOT-GASTRO-001'}})
    store.apply_change(state.id, 'alice', 'c', PlanChange(base_version=1,
        reason='用户改号源', update=[changed]), user_text='改为预约 SLOT-GASTRO-001')
    assert service.actions.get(action['id'], 'alice')['status'] == 'expired'
    with pytest.raises(ValueError, match='expired'):
        asyncio.run(service.confirm(action['id'], 'alice', 'c'))
    assert service.backend.appointments == {}


def test_unrelated_plan_change_keeps_confirmation_and_repeat_is_idempotent(tmp_path):
    store, state, service, action = prepared(tmp_path)
    changed = state.tasks['doctors'].spec.model_copy(update={
        'fragment': '改查消化内科医生', 'arguments': {'department': '消化内科'}})
    store.apply_change(state.id, 'alice', 'c', PlanChange(base_version=1,
        reason='修改独立查询', update=[changed]), user_text='改查消化内科医生')
    first = asyncio.run(service.confirm(action['id'], 'alice', 'c'))
    again = asyncio.run(service.confirm(action['id'], 'alice', 'c'))
    assert first['status'] == again['status'] == 'completed'
    assert len(service.backend.appointments) == 1
    synced = store.sync_actions(state.id, 'alice', 'c')
    assert synced.tasks['booking'].status == TaskStatus.SUCCEEDED
    with pytest.raises(PermissionError):
        store.apply_change(state.id, 'alice', 'c', PlanChange(base_version=2,
            reason='取消', cancel=['booking']), user_text='取消预约任务')


def test_confirmation_cannot_be_claimed_by_another_user_or_conversation(tmp_path):
    _, _, service, action = prepared(tmp_path)
    for actor, conversation in [('bob', 'c'), ('alice', 'other')]:
        with pytest.raises(LookupError):
            asyncio.run(service.confirm(action['id'], actor, conversation))
    assert service.actions.get(action['id'], 'alice')['status'] == 'pending_confirmation'


def test_committed_write_with_lost_response_is_reconciled_without_resubmit(tmp_path):
    class LostResponse(InMemoryHospitalBackend):
        submissions = 0
        async def execute(self, *args):
            self.submissions += 1
            await super().execute(*args)
            raise TimeoutError('response lost after commit')
    backend = LostResponse()
    service = AppointmentService(backend, ActionStore(tmp_path/'actions.sqlite3'))
    action = service.prepare('appointment_create', 'alice', 'c', {'slot_id': 'SLOT-CARD-001'})
    first = asyncio.run(service.confirm(action['id'], 'alice', 'c'))
    second = asyncio.run(service.confirm(action['id'], 'alice', 'c'))
    assert first['status'] == second['status'] == 'completed'
    assert backend.submissions == len(backend.appointments) == 1


def test_unknown_outcome_is_persisted_across_service_restart_and_never_reexecuted(tmp_path):
    class UnknownBackend(InMemoryHospitalBackend):
        submissions = 0
        async def execute(self, *args):
            self.submissions += 1
            raise TimeoutError('unknown write')
    path = tmp_path/'actions.sqlite3'
    backend = UnknownBackend()
    service = AppointmentService(backend, ActionStore(path))
    action = service.prepare('appointment_create', 'alice', 'c', {'slot_id': 'SLOT-CARD-001'})
    assert asyncio.run(service.confirm(action['id'], 'alice', 'c'))['status'] == 'reconciling'
    reopened = AppointmentService(backend, ActionStore(path))
    assert asyncio.run(reopened.confirm(action['id'], 'alice', 'c'))['status'] == 'reconciling'
    assert backend.submissions == 1


def test_concurrent_confirmations_claim_exactly_one_write(tmp_path):
    async def run():
        class SlowBackend(InMemoryHospitalBackend):
            submissions = 0
            async def execute(self, *args):
                self.submissions += 1
                await asyncio.sleep(.02)
                return await super().execute(*args)
        backend = SlowBackend()
        service = AppointmentService(backend, ActionStore(tmp_path/'actions.sqlite3'))
        action = service.prepare('appointment_create', 'alice', 'c', {'slot_id': 'SLOT-CARD-001'})
        results = await asyncio.gather(*(service.confirm(action['id'], 'alice', 'c') for _ in range(4)))
        assert any(item['status'] == 'completed' for item in results)
        assert backend.submissions == len(backend.appointments) == 1
    asyncio.run(run())


@pytest.mark.parametrize('responses', [[], [{'response': {'status': '201 Created'}},
                                         {'response': {'status': '500 Server Error'}}]])
def test_fhir_http_200_without_complete_transaction_proof_is_unknown(responses):
    async def run():
        posts = []
        def handle(request):
            if request.method == 'POST':
                posts.append(request)
                return httpx.Response(200, json={'resourceType': 'Bundle', 'type': 'transaction-response', 'entry': responses})
            return httpx.Response(200, json={'resourceType': 'Slot', 'id': 'SLOT-TEST', 'status': 'free',
                'start': '2030-01-02T10:00:00+08:00', 'end': '2030-01-02T10:15:00+08:00'})
        backend = FHIRHospitalBackend('http://synthetic.test/fhir')
        await backend.client.aclose()
        backend.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        result = await backend.execute('appointment_create', {'patient_id': 'alice', 'slot_id': 'SLOT-TEST'}, 'fixture-operation')
        assert result['status'] == 'unknown'
        assert len(posts) == 1
        assert backend._results == {}
        await backend.close()
    asyncio.run(run())


def test_plan_cannot_erase_write_that_was_claimed_concurrently(tmp_path):
    async def run():
        store, state, service, action = prepared(tmp_path)
        entered, release = asyncio.Event(), asyncio.Event()
        original = service.backend.execute
        async def slow(*args):
            entered.set()
            await release.wait()
            return await original(*args)
        service.backend.execute = slow
        running = asyncio.create_task(service.confirm(action['id'], 'alice', 'c'))
        await asyncio.wait_for(entered.wait(), 2)
        try:
            with pytest.raises(PermissionError):
                store.apply_change(state.id, 'alice', 'c', PlanChange(base_version=1,
                    reason='取消任务', cancel=['booking']), user_text='取消任务')
        finally:
            release.set()
        assert (await running)['status'] == 'completed'
        synced = store.sync_actions(state.id, 'alice', 'c')
        assert synced.tasks['booking'].status == TaskStatus.SUCCEEDED
        assert len(service.backend.appointments) == 1
    asyncio.run(run())
