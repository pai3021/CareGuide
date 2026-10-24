import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest

from careguide.orchestration.budget import BudgetExhausted, BudgetLedger, BudgetLimits


def ledger(tmp_path, **limits):
    return BudgetLedger(tmp_path/'budget.sqlite3', 'workflow', 'alice', 'conversation',
                        BudgetLimits(**limits))


def test_concurrent_reservations_cannot_oversubscribe_shared_quota(tmp_path):
    budget = ledger(tmp_path, model=2)
    def reserve(_):
        try:
            return budget.reserve('model')
        except BudgetExhausted:
            return None
    with ThreadPoolExecutor(max_workers=8) as executor:
        granted = [ticket for ticket in executor.map(reserve, range(20)) if ticket]
    assert len(granted) == 2
    assert budget.summary()['used']['model'] == 2


def test_reopened_ledger_keeps_usage_and_original_limits(tmp_path):
    first = ledger(tmp_path, model=1)
    first.reserve('model')
    reopened = ledger(tmp_path, model=100)
    assert reopened.summary()['limits']['model'] == 1
    with pytest.raises(BudgetExhausted):
        reopened.reserve('model')


@pytest.mark.parametrize('actor,conversation', [('bob', 'conversation'), ('alice', 'other')])
def test_budget_is_bound_to_user_and_conversation(tmp_path, actor, conversation):
    ledger(tmp_path)
    with pytest.raises(PermissionError):
        BudgetLedger(tmp_path/'budget.sqlite3', 'workflow', actor, conversation)


def test_parallel_execution_time_is_reserved_before_dispatch(tmp_path):
    budget = ledger(tmp_path, execution_seconds=1)
    first = budget.reserve('tool', 0.5)
    second = budget.reserve('tool', 0.5)
    with pytest.raises(BudgetExhausted):
        budget.reserve('tool', 0.5)
    assert budget.summary()['used']['tool'] == 2
    budget.finish(first, 0.1)
    third = budget.reserve('tool', 0.5)
    assert third.timeout_s == pytest.approx(0.4)
    budget.finish(second, 0.1)
    budget.finish(third, 0.1)


def test_settlement_is_idempotent_and_failures_do_not_refund_calls(tmp_path):
    budget = ledger(tmp_path, tool=1)
    ticket = budget.reserve('tool', 1)
    budget.finish(ticket, 0.2)
    budget.finish(ticket, 0.2)
    assert budget.summary()['active_execution_seconds'] == pytest.approx(0.2)
    with pytest.raises(BudgetExhausted):
        budget.reserve('tool', 1)


def test_closing_blocks_new_exploration_but_keeps_bounded_completion_room(tmp_path):
    budget = ledger(tmp_path, model=5)
    for _ in range(4):
        budget.reserve('model')
    assert budget.summary()['closing']
    with pytest.raises(BudgetExhausted, match='closing'):
        budget.reserve('model', exploration=True)
    budget.reserve('model')
    with pytest.raises(BudgetExhausted):
        budget.reserve('model')


def test_timeout_marks_unknown_and_charges_reserved_execution_time(tmp_path):
    budget = ledger(tmp_path)
    async def run():
        with pytest.raises(TimeoutError):
            async with budget.call('tool', timeout_s=0.01):
                await asyncio.sleep(1)
    asyncio.run(run())
    summary = budget.summary()
    assert summary['reservations']['unknown'] == 1
    assert summary['active_execution_seconds'] == pytest.approx(0.01)
    assert summary['used']['tool'] == 1


def test_cancellation_does_not_return_budget_or_claim_remote_work_stopped(tmp_path):
    budget = ledger(tmp_path)
    async def run():
        started = asyncio.Event()
        async def work():
            async with budget.call('tool', timeout_s=0.1):
                started.set()
                await asyncio.Event().wait()
        task = asyncio.create_task(work())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(run())
    assert budget.summary()['reservations']['unknown'] == 1
    assert budget.summary()['used']['tool'] == 1


def test_provider_usage_and_absent_usage_are_separate(tmp_path):
    budget = ledger(tmp_path)
    first = budget.reserve('model')
    first.record_response({'usage': {'prompt_tokens': 10, 'completion_tokens': 3, 'total_tokens': 13}})
    budget.finish(first, 0)
    second = budget.reserve('model')
    budget.finish(second, 0)
    assert budget.summary()['tokens'] == {'prompt_tokens': 10,
        'completion_tokens': 3, 'total_tokens': 13, 'calls_without_usage': 1}


def test_malformed_usage_does_not_break_completed_business_work(tmp_path):
    budget = ledger(tmp_path)
    ticket = budget.reserve('model')
    ticket.record_response({'usage': {'prompt_tokens': 'invalid'}})
    budget.finish(ticket, 0)
    assert budget.summary()['tokens']['calls_without_usage'] == 1


def test_confirmed_process_loss_keeps_reservations_as_unknown(tmp_path):
    budget = ledger(tmp_path)
    budget.reserve('model', timeout_s=0.5)
    reopened = ledger(tmp_path)
    assert reopened.recover_interrupted_reservations() == 1
    assert reopened.recover_interrupted_reservations() == 0
    assert reopened.summary()['used']['model'] == 1
    assert reopened.summary()['reservations']['unknown'] == 1
    assert reopened.summary()['active_execution_seconds'] == pytest.approx(0.5)


def test_missing_persisted_budget_cannot_be_recreated_during_confirmation(tmp_path):
    with pytest.raises(PermissionError):
        BudgetLedger(tmp_path/'budget.sqlite3', 'missing', 'alice', 'c', require_existing=True)
