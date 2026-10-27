import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from careguide.archived_memory import ArchivedMemory
from careguide.conversation_memory import Message, MsgRole
from careguide.profile_memory import Extraction, ProfileConflict, ProfileStore, active_profile
from careguide.session_store import SessionStore
from tests.test_profile_memory import manager, candidate, message
from tests.test_model_planner import reply


def test_review_every_ten_user_turns_and_short_session_shutdown(tmp_path):
    async def scenario():
        underlying = SimpleNamespace(manager=SimpleNamespace(update_profile=AsyncMock(
            return_value={'status': 'skipped'})), info=lambda: {}, close=AsyncMock())
        memory = ArchivedMemory(underlying, tmp_path/'archive.sqlite3', review_interval=10)
        for i in range(9):
            memory.store.append('alice', 'c', 'user', f'问题{i}')
            memory.store.append('alice', 'c', 'assistant', f'回答{i}')
        memory._schedule_review('alice', 'c')
        assert memory._reviews == {}
        memory.store.append('alice', 'c', 'user', '第十个问题')
        memory.store.append('alice', 'c', 'assistant', '第十个回答')
        memory._schedule_review('alice', 'c')
        await memory._reviews[('alice', 'c')]
        underlying.manager.update_profile.assert_awaited_once()
        snapshot, _ = memory.store.snapshot('alice', 'c', 'review')
        assert snapshot['turn_count'] == 10
        memory._schedule_review('alice', 'c')
        assert underlying.manager.update_profile.await_count == 1
        memory.store.append('bob', 'short', 'user', '普通问题')
        memory._seen.add(('bob', 'short'))
        await memory.close()
        assert underlying.manager.update_profile.await_count == 2
    asyncio.run(scenario())


def test_incremental_review_checks_all_eligible_users_and_skips_processed_sources(tmp_path):
    async def scenario():
        profile = manager(tmp_path, [], [])
        memory = ArchivedMemory(SimpleNamespace(manager=profile), tmp_path/'archive.sqlite3', review_interval=10)
        first = memory.store.append('alice', 'c', 'user', '以后我偏好上午')
        last = memory.store.append('alice', 'c', 'user', '普通问题，无需长期记忆')
        profile._client.replies.append(reply({'candidates': [candidate(identity=first['id'])]}))
        outcome = await memory.review('alice', 'c')
        assert outcome['status'] == 'updated'
        assert active_profile(profile._profile_store.get('alice')[1])['preferred_time']['value'] == 'morning'
        assert memory.store.snapshot('alice', 'c', 'review')[0]['covered_seq'] == last['seq']
        assert (await memory.review('alice', 'c'))['status'] == 'skipped'
        assert len(profile._client.calls) == 1
    asyncio.run(scenario())


def test_old_review_cannot_revive_a_deleted_preference(tmp_path):
    archive = SessionStore(tmp_path/'archive.sqlite3')
    first = archive.append('alice', 'c', 'user', '以后我偏好上午')
    second = archive.append('alice', 'other', 'user', '忘记我的时间偏好')
    store = ProfileStore(tmp_path/'profile.sqlite3')
    store.ensure('alice', {})
    store.merge('alice', Extraction(candidates=[candidate(identity=first['id'])]), expected_revision=0,
                projection=lambda _: None, source_order={first['id']: first['seq']})
    store.merge('alice', Extraction(candidates=[candidate(None, '忘记我的时间偏好', second['id'], 'delete')]),
                expected_revision=1, projection=lambda _: None, source_order={second['id']: second['seq']})
    try:
        store.merge('alice', Extraction(candidates=[candidate(identity=first['id'])]), expected_revision=2,
                    projection=lambda _: None, source_order={first['id']: first['seq']})
    except ProfileConflict:
        pass
    else:
        raise AssertionError('Old source must not revive deleted preference')
    assert active_profile(store.get('alice')[1]) == {}


def test_transient_extraction_failure_is_retryable(tmp_path):
    async def scenario():
        memory = manager(tmp_path, [message('以后我偏好上午')], [])
        memory._client.replies = [TimeoutError('fixture'), reply({'candidates': [candidate()]})]
        assert (await memory.update_profile('alice', 'c'))['status'] == 'unavailable'
        assert not memory._profile_store.processed('alice', 'source-1')
        assert (await memory.update_profile('alice', 'c'))['status'] == 'updated'
    asyncio.run(scenario())
