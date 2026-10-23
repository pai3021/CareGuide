import asyncio
from datetime import datetime, timedelta, timezone

import pytest
import httpx

from careguide.hospital_workflow import InMemoryHospitalBackend
from careguide.intent_recognizer import IntentRecognizer
from careguide.orchestration.budget import BudgetLimits
from careguide.orchestration.contracts import PlanChange, TaskStatus
from careguide.orchestration.store import WorkflowExpired
from careguide.orchestration.runtime import needs_plan
from careguide.corpus import Corpus, DEFAULT_CORPUS
from scripts.reproduce_orchestration import build, task_plan
from careguide.api import app
from careguide.runtime import get_chat_runtime
from careguide.task_context import SessionMemory


ACTOR, CONVERSATION = 'example-user', 'example-conversation'


def test_no_slots_cannot_silently_change_date_but_explicit_user_change_preserves_drug_result(tmp_path):
    today = datetime.now(timezone(timedelta(hours=8))).date()
    tomorrow, later = [(today+timedelta(days=value)).isoformat() for value in (1, 2)]
    text = '查消化内科明天号源，同时查询精氨酸布洛芬片0.2g说明书'
    _, plan = task_plan()
    plan['goal'] = text
    plan['tasks'][0].update(id='slots', fragment='查消化内科明天号源', action='list_available_slots',
        arguments={'department': '消化内科', 'date': tomorrow})
    unauthorized = {**plan['tasks'][0], 'arguments': {'department': '消化内科', 'date': later}}
    user_change = {**unauthorized, 'fragment': '同意改查后天的消化内科号源'}
    calls = []
    class Backend(InMemoryHospitalBackend):
        async def list_available_slots(self, **kwargs):
            calls.append(kwargs)
            return await super().list_available_slots(**kwargs)
    backend = Backend(slots=[dict(id='SLOT-LATER', practitioner_id='PRA-GASTRO-01',
        department='消化内科', status='free', start=later+'T10:00:00+08:00', end=later+'T10:15:00+08:00')])
    runtime, provider = build(tmp_path, [plan,
        dict(base_version=1, reason='无结果后擅自换日期', update=[unauthorized]),
        dict(base_version=1, reason='用户同意换日期', update=[user_change])], backend=backend)
    async def run():
        first = await runtime.process(text, ACTOR, CONVERSATION)
        assert first['workflow_status'] == 'partial_completed'
        assert [call['date'] for call in calls] == [tomorrow]
        assert [failure['code'] for failure in first['errors']] == ['permission_denied', 'not_covered']
        identity = first['workflow']['id']
        evidence = next(task for task in first['workflow']['tasks'] if task['id'] == 'drug')['result']['evidence']
        changed = await runtime.process('同意改查后天的消化内科号源', ACTOR, CONVERSATION)
        assert changed['workflow']['id'] == identity
        assert changed['workflow_status'] == 'completed'
        assert [call['date'] for call in calls] == [tomorrow, later]
        state = runtime.store.get(identity, ACTOR, CONVERSATION)
        assert state.tasks['drug'].attempts == state.tasks['drug'].revision == 1
        assert state.tasks['drug'].result['evidence'] == evidence
        assert changed['budget']['used']['replan'] == 2
        assert len(provider.calls) == 3
    asyncio.run(run())


def test_repeated_identical_replan_stops_after_one_without_extra_tool_dispatch(tmp_path):
    text, plan = task_plan()
    plan['tasks'][0]['arguments'] = {'department': '未收录科室'}
    plan['tasks'][0]['fragment'] = '查未收录科室医生'
    text = plan['goal'] = text.replace('消化内科', '未收录科室')
    runtime, provider = build(tmp_path, [plan, dict(base_version=1, reason='重复计划', update=[plan['tasks'][0]])])
    response = asyncio.run(runtime.process(text, ACTOR, CONVERSATION))
    assert response['workflow_status'] == 'partial_completed'
    assert response['budget']['used']['replan'] == 1
    assert response['budget']['used']['tool'] == 2
    assert len(provider.calls) == 2
    assert 'invalid_parameters' in [failure['code'] for failure in response['errors']]


def test_multiple_waits_do_not_bind_ambiguous_free_text_or_call_model(tmp_path):
    text, plan = task_plan(waiting=True)
    second = {**plan['tasks'][1], 'id': 'second', 'fragment': '查询富马酸酮替芬滴鼻液说明书',
              'arguments': {'drug_name': '富马酸酮替芬滴鼻液', 'dosage_form': '滴鼻剂'}}
    text += '，另外'+second['fragment']
    plan['goal'] = text
    plan['tasks'].append(second)
    runtime, provider = build(tmp_path, [plan])
    async def run():
        first = await runtime.process(text, ACTOR, CONVERSATION)
        assert first['workflow_status'] == 'waiting_user'
        second_response = await runtime.process('0.2g', ACTOR, CONVERSATION)
        assert '本次回答属于哪项任务' in second_response['response']
        state = runtime.store.get(first['workflow']['id'], ACTOR, CONVERSATION)
        assert state.tasks['drug'].status == state.tasks['second'].status == TaskStatus.WAITING_INPUT
        assert state.tasks['drug'].arguments.get('specification') is None
        assert len(provider.calls) == 1
    asyncio.run(run())


def test_expired_wait_rejects_late_answer_and_preserves_completed_query(tmp_path, monkeypatch):
    text, plan = task_plan(waiting=True)
    runtime, _ = build(tmp_path, [plan])
    async def run():
        first = await runtime.process(text, ACTOR, CONVERSATION)
        identity = first['workflow']['id']
        later = datetime.now(timezone.utc)+timedelta(minutes=6)
        monkeypatch.setattr('careguide.orchestration.store.utc_now', lambda: later)
        with pytest.raises(WorkflowExpired):
            await runtime.process('0.2g', ACTOR, CONVERSATION, workflow_event=dict(
                kind='answer', workflow_id=identity, plan_version=1,
                task_id='drug', task_revision=1, values={'specification': '0.2g'}))
        state = runtime.store.get(identity, ACTOR, CONVERSATION)
        assert state.tasks['drug'].status == TaskStatus.EXPIRED
        assert state.tasks['doctors'].status == TaskStatus.SUCCEEDED
    asyncio.run(run())


def test_user_cancel_and_add_are_local_and_preserve_completed_doctor_query(tmp_path):
    text, plan = task_plan(waiting=True)
    change = dict(base_version=1, reason='取消旧查询并加查新科室', cancel=['drug'], add=[
        dict(id='new_doctors', fragment='另外查心血管内科医生', action='search_doctors', agent='appointment',
             arguments={'department': '心血管内科'}, completion='items')])
    runtime, provider = build(tmp_path, [plan, change])
    async def run():
        first = await runtime.process(text, ACTOR, CONVERSATION)
        identity = first['workflow']['id']
        response = await runtime.process('不用查药品，另外查心血管内科医生', ACTOR, CONVERSATION)
        assert response['workflow_status'] == 'completed'
        state = runtime.store.get(identity, ACTOR, CONVERSATION)
        assert state.tasks['drug'].status == TaskStatus.CANCELLED
        assert state.tasks['doctors'].attempts == 1
        assert state.tasks['new_doctors'].status == TaskStatus.SUCCEEDED
        assert response['budget']['used']['tool'] == 2
        assert len(provider.calls) == 2
    asyncio.run(run())


def test_emergency_followup_preempts_waiting_workflow_before_any_ordinary_call(tmp_path):
    text, plan = task_plan(waiting=True)
    runtime, provider = build(tmp_path, [plan])
    async def run():
        first = await runtime.process(text, ACTOR, CONVERSATION)
        runtime.engine.recognizer = IntentRecognizer(embedding_mode='disabled')
        response = await runtime.process('现在家人叫不醒，继续查药', ACTOR, CONVERSATION)
        assert response['workflow_status'] == 'urgent_handoff'
        assert response['tools_used'] == []
        state = runtime.store.get(first['workflow']['id'], ACTOR, CONVERSATION)
        assert state.status == 'urgent_handoff'
        assert state.tasks['drug'].status == TaskStatus.CANCELLED
        assert state.tasks['doctors'].status == TaskStatus.SUCCEEDED
        assert runtime.ledger(state).summary()['used']['tool'] == 1
        assert len(provider.calls) == 1
    asyncio.run(run())


def test_resume_does_not_reset_exhausted_shared_tool_budget(tmp_path):
    text, plan = task_plan(waiting=True)
    runtime, _ = build(tmp_path, [plan])
    runtime.limits = BudgetLimits(tool=1)
    async def run():
        first = await runtime.process(text, ACTOR, CONVERSATION)
        response = await runtime.process('0.2g', ACTOR, CONVERSATION, workflow_event=dict(
            kind='answer', workflow_id=first['workflow']['id'], plan_version=1,
            task_id='drug', task_revision=1, values={'specification': '0.2g'}))
        assert response['workflow_status'] == 'partial_completed'
        assert response['budget']['used']['tool'] == 1
        assert response['budget']['closing'] is True
        assert any(failure['code'] == 'budget_exhausted' for failure in response['errors'])
        state = runtime.store.get(first['workflow']['id'], ACTOR, CONVERSATION)
        assert state.tasks['doctors'].attempts == 1
    asyncio.run(run())


def test_http_change_during_inflight_query_discards_late_old_result(tmp_path):
    async def run():
        started, release = asyncio.Event(), asyncio.Event()
        class SlowBackend(InMemoryHospitalBackend):
            async def search_doctors(self, *, department=None):
                if department == '消化内科':
                    started.set()
                    try:
                        await release.wait()
                    except asyncio.CancelledError:
                        # Simulate a remote request that does not stop when its
                        # caller cancels the coroutine.
                        await release.wait()
                return await super().search_doctors(department=department)
        text, plan = task_plan()
        changed = {**plan['tasks'][0], 'fragment': '改查心血管内科医生',
                   'arguments': {'department': '心血管内科'}}
        runtime, _ = build(tmp_path, [plan, dict(base_version=1, reason='用户改科室', update=[changed])],
                           backend=SlowBackend())
        app.dependency_overrides[get_chat_runtime] = lambda: (runtime, SessionMemory(), None)
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as http:
                first_request = asyncio.create_task(http.post('/chat', json={
                    'message': text, 'user_id': ACTOR, 'conv_id': CONVERSATION}))
                await asyncio.wait_for(started.wait(), 2)
                state = runtime.store.latest(ACTOR, CONVERSATION)
                # Do not release the old read until the HTTP change completed.
                changed_response = await asyncio.wait_for(http.post('/chat', json={
                    'message': '改查心血管内科医生', 'user_id': ACTOR, 'conv_id': CONVERSATION,
                    'workflow_event': dict(kind='change', workflow_id=state.id, plan_version=1)}), 3)
                assert changed_response.status_code == 200, changed_response.text
                assert changed_response.json()['workflow']['version'] == 2
                assert not release.is_set()
                release.set()
                old_response = await asyncio.wait_for(first_request, 3)
                assert old_response.status_code == 200, old_response.text
                # Let the callback archive a cancellation-resistant late result.
                for _ in range(20):
                    if not runtime.scheduler.pending:
                        break
                    await asyncio.sleep(.01)
                final = runtime.store.get(state.id, ACTOR, CONVERSATION)
                assert final.version == 2
                assert final.tasks['doctors'].result['items'][0]['department'] == '心血管内科'
                assert any(item.get('discarded') for item in final.archived_results)
                assert final.tasks['drug'].attempts == 1
        finally:
            release.set()
            app.dependency_overrides.clear()
    asyncio.run(run())


def test_local_change_keeps_unaffected_inflight_task_and_does_not_repeat_it(tmp_path):
    async def run():
        old_doctor_started, drug_started, new_doctor_started = (asyncio.Event() for _ in range(3))
        release_old = asyncio.Event()
        class Backend(InMemoryHospitalBackend):
            async def search_doctors(self, *, department=None):
                if department == '消化内科':
                    old_doctor_started.set()
                    try:
                        await release_old.wait()
                    except asyncio.CancelledError:
                        await release_old.wait()
                else:
                    new_doctor_started.set()
                return await super().search_doctors(department=department)
        text, plan = task_plan()
        changed = {**plan['tasks'][0], 'fragment': '改查心血管内科医生',
                   'arguments': {'department': '心血管内科'}}
        runtime, _ = build(tmp_path, [plan, dict(base_version=1, reason='用户修改科室', update=[changed])], backend=Backend())
        tool = runtime.tool_manager._tools['medication:lookup_drug_label']
        original = tool.handler
        calls = []
        async def slow_drug(params, context):
            calls.append(params)
            drug_started.set()
            await new_doctor_started.wait()
            return await original(params, context)
        tool.handler = slow_drug
        first = asyncio.create_task(runtime.process(text, ACTOR, CONVERSATION))
        try:
            await asyncio.wait_for(asyncio.gather(old_doctor_started.wait(), drug_started.wait()), 2)
            state = runtime.store.latest(ACTOR, CONVERSATION)
            response = await asyncio.wait_for(runtime.process('改查心血管内科医生', ACTOR, CONVERSATION,
                workflow_event=dict(kind='change', workflow_id=state.id, plan_version=1)), 3)
            assert response['workflow_status'] == 'completed'
            current = runtime.store.get(state.id, ACTOR, CONVERSATION)
            assert current.tasks['drug'].revision == current.tasks['drug'].attempts == 1
            assert len(calls) == 1
            assert current.tasks['drug'].status == TaskStatus.SUCCEEDED
        finally:
            release_old.set()
            await asyncio.wait_for(first, 3)
    asyncio.run(run())


def test_explicit_cancel_works_with_exhausted_model_budget_and_no_planner_call(tmp_path):
    text, plan = task_plan(waiting=True)
    runtime, provider = build(tmp_path, [plan])
    runtime.limits = BudgetLimits(model=1)
    async def run():
        first = await runtime.process(text, ACTOR, CONVERSATION)
        response = await runtime.process('取消当前药品任务', ACTOR, CONVERSATION,
            workflow_event=dict(kind='cancel', workflow_id=first['workflow']['id'],
                plan_version=1, task_id='drug', task_revision=1))
        assert response['workflow_status'] == 'completed'
        assert response['budget']['used']['model'] == 1
        assert len(provider.calls) == 1
        state = runtime.store.get(first['workflow']['id'], ACTOR, CONVERSATION)
        assert state.tasks['drug'].status == TaskStatus.CANCELLED
        assert state.tasks['doctors'].status == TaskStatus.SUCCEEDED
    asyncio.run(run())


def test_raw_text_gate_catches_two_named_drugs_without_relying_on_conjunction_or_single_label():
    corpus = Corpus(DEFAULT_CORPUS)
    assert needs_plan('精氨酸布洛芬片和富马酸酮替芬滴鼻液有什么禁忌', corpus.records)
    assert needs_plan('查消化内科医生并查药品说明书', corpus.records)
    assert not needs_plan('精氨酸布洛芬片0.2g有什么禁忌和注意事项', corpus.records)


def test_uncertain_request_uses_real_planner_even_with_confident_single_label(tmp_path):
    text = '请帮我安排消化内科就医事宜'
    _, plan = task_plan()
    plan.update(goal=text, tasks=[{**plan['tasks'][0], 'fragment': text}])
    runtime, provider = build(tmp_path, [plan])
    response = asyncio.run(runtime.process(text, ACTOR, CONVERSATION))
    assert response['workflow_status'] == 'completed'
    assert len(provider.calls) == 1
    assert response['budget']['used']['model'] == 1
    assert response['workflow']['tasks'][0]['result']['items']


def test_closing_budget_rejects_added_branch_and_keeps_previous_result(tmp_path):
    text, plan = task_plan(waiting=True)
    addition = dict(id='extra', fragment='另外查心内科医生', action='search_doctors',
        agent='appointment', arguments={'department': '心内科'}, completion='items')
    runtime, provider = build(tmp_path, [plan, dict(base_version=1, reason='用户加查', add=[addition])])
    runtime.limits = BudgetLimits(model=5)
    async def run():
        first = await runtime.process(text, ACTOR, CONVERSATION)
        state = runtime.store.get(first['workflow']['id'], ACTOR, CONVERSATION)
        ledger = runtime.ledger(state)
        for _ in range(3):
            ticket = ledger.reserve('model')
            ledger.finish(ticket, 0)
        assert ledger.summary()['closing']
        result = await runtime.process('另外查心内科医生', ACTOR, CONVERSATION,
            workflow_event=dict(kind='change', workflow_id=state.id, plan_version=1))
        current = runtime.store.get(state.id, ACTOR, CONVERSATION)
        assert 'extra' not in current.tasks
        assert current.version == 1
        assert current.tasks['doctors'].attempts == 1
        assert current.tasks['doctors'].result == state.tasks['doctors'].result
        assert result['budget']['used']['model'] == 5
        assert result['budget']['closing']
        assert 'budget_exhausted' in [failure['code'] for failure in result['errors']]
        assert len(provider.calls) == 2
    asyncio.run(run())


def test_short_date_change_keeps_confirmed_department_and_period_even_if_model_copies_old_fragment(tmp_path):
    today = datetime.now(timezone(timedelta(hours=8))).date()
    tomorrow, later = [(today+timedelta(days=delta)).isoformat() for delta in (1, 2)]
    text, plan = task_plan()
    fragment = '查消化内科明天下午号源'
    text = plan['goal'] = text.replace('查消化内科医生', fragment)
    plan['tasks'][0].update(id='slots', fragment=fragment, action='list_available_slots',
        arguments={'department': '消化内科', 'date': tomorrow, 'time_period': 'afternoon'})
    copied_fragment = {**plan['tasks'][0], 'arguments': {**plan['tasks'][0]['arguments'], 'date': later}}
    backend = InMemoryHospitalBackend(slots=[dict(id='SLOT-PM-LATER', practitioner_id='PRA-GASTRO-01',
        department='消化内科', status='free', start=later+'T15:00:00+08:00', end=later+'T15:15:00+08:00')])
    runtime, _ = build(tmp_path, [plan,
        dict(base_version=1, reason='重复无结果查询', update=[plan['tasks'][0]]),
        dict(base_version=1, reason='用户同意后天', update=[copied_fragment])], backend=backend)
    async def run():
        first = await runtime.process(text, ACTOR, CONVERSATION)
        assert first['workflow_status'] == 'partial_completed'
        final = await runtime.process('同意改查后天', ACTOR, CONVERSATION)
        assert final['workflow_status'] == 'completed'
        state = runtime.store.get(first['workflow']['id'], ACTOR, CONVERSATION)
        slot_task = state.tasks['slots']
        assert slot_task.spec.fragment == '同意改查后天'
        assert slot_task.arguments == {'department': '消化内科', 'date': later, 'time_period': 'afternoon'}
        assert slot_task.origins['department']['kind'] == 'retained_scope'
        assert slot_task.origins['date']['kind'] == 'user_change'
        assert slot_task.origins['time_period']['kind'] == 'retained_scope'
        assert slot_task.result['items'][0]['id'] == 'SLOT-PM-LATER'
        assert state.tasks['drug'].attempts == 1
    asyncio.run(run())


def test_missing_drug_spec_does_not_block_ready_slots_or_repeat_them_on_resume(tmp_path):
    text, plan = task_plan(waiting=True)
    fragment = '查消化内科2030年1月2日号源'
    text = plan['goal'] = text.replace('查消化内科医生', fragment)
    plan['tasks'][0].update(id='slots', fragment=fragment, action='list_available_slots',
        arguments={'department': '消化内科', 'date': '2030-01-02'})
    runtime, provider = build(tmp_path, [plan])
    async def run():
        first = await runtime.process(text, ACTOR, CONVERSATION)
        assert first['workflow_status'] == 'waiting_user'
        identity = first['workflow']['id']
        before = runtime.store.get(identity, ACTOR, CONVERSATION)
        assert before.tasks['slots'].status == TaskStatus.SUCCEEDED
        assert before.tasks['slots'].result['items']
        final = await runtime.process('0.2g', ACTOR, CONVERSATION)
        assert final['workflow_status'] == 'completed'
        after = runtime.store.get(identity, ACTOR, CONVERSATION)
        assert after.tasks['slots'].attempts == 1
        assert after.tasks['slots'].result == before.tasks['slots'].result
        assert after.tasks['drug'].attempts == 1
        assert len(provider.calls) == 1
    asyncio.run(run())


def test_two_drug_contraindication_tasks_execute_real_planner_and_keep_sections_isolated(tmp_path):
    products = [('精氨酸布洛芬片', '片剂', '0.2g'), ('富马酸酮替芬鼻喷雾剂', '鼻喷雾剂', '15ml:16.7mg')]
    fragments = [name+spec+'禁忌' for name, _, spec in products]
    text = '和'.join(fragments)
    plan = dict(goal=text, tasks=[dict(id='drug'+str(index), fragment=fragment, agent='medication',
        action='lookup_drug_label', arguments=dict(drug_name=name, dosage_form=form, specification=spec,
            section='dosage'), completion='source_evidence')
        for index, ((name, form, spec), fragment) in enumerate(zip(products, fragments))])
    runtime, provider = build(tmp_path, [plan])
    response = asyncio.run(runtime.process(text, ACTOR, CONVERSATION))
    assert response['workflow_status'] == 'completed'
    for task, (name, _, _) in zip(response['workflow']['tasks'], products):
        assert task['result']['evidence']
        assert {item['scope']['drug_name'] for item in task['result']['evidence']} == {name}
        assert all(item['scope']['section'] == 'contraindications' for item in task['result']['evidence'])
    assert len(provider.calls) == 1
    assert response['budget']['used']['tool'] == 2
