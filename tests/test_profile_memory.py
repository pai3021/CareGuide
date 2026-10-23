import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from careguide.conversation_memory import MemoryManager, Message, MsgRole
from careguide.profile_memory import Extraction, ProfileConflict, ProfileStore, active_profile
from tests.test_chat_runtime import ScriptedClient
from tests.test_model_planner import reply


def message(text, identity='source-1', role=MsgRole.USER):
    return Message(role, text, metadata={'message_id': identity})


def candidate(value='morning', quote='以后我偏好上午', identity='source-1', operation='set'):
    return dict(key='preferred_time', value=value, operation=operation,
                source_message_id=identity, source_quote=quote)


def manager(tmp_path, messages, payloads):
    instance = MemoryManager.__new__(MemoryManager)
    instance._profile_store = ProfileStore(tmp_path/'memory.sqlite3')
    instance._profile = SimpleNamespace(upsert=Mock(), delete=Mock(), add=Mock())
    instance._get_working_memory = AsyncMock(return_value=messages)
    instance._get_legacy_profile = AsyncMock(return_value={})
    instance._client, instance._model = ScriptedClient([reply(value) for value in payloads]), 'offline'
    return instance


@pytest.mark.parametrize('text,role', [
    ('今天我下午去', MsgRole.USER), ('这次先查布洛芬说明书', MsgRole.USER),
    ('记住今天下午去医院', MsgRole.USER), ('以后可能喜欢上午', MsgRole.USER),
    ('以后查药品资料', MsgRole.USER), ('以后我偏好上午', MsgRole.ASSISTANT)])
def test_non_durable_or_assistant_content_skips_model_and_write(tmp_path, text, role):
    memory = manager(tmp_path, [message(text, role=role)], [])
    outcome = asyncio.run(memory.update_profile('alice', 'c'))
    assert outcome['status'] == 'skipped'
    assert outcome['model_calls'] == 0
    assert memory._client.calls == []
    memory._profile.upsert.assert_not_called()


@pytest.mark.parametrize('text', ['我妈妈以后喜欢上午', '医生说以后优先上午', '助手建议以后优先上午'])
def test_other_people_and_recommendations_cannot_become_this_users_profile(tmp_path, text):
    memory = manager(tmp_path, [message(text)], [])
    outcome = asyncio.run(memory.update_profile('alice', 'c'))
    assert outcome['status'] == 'skipped'
    assert memory._client.calls == []


def test_durable_morning_survives_temporary_afternoon_and_same_message_is_not_reextracted(tmp_path):
    memory = manager(tmp_path, [message('以后我偏好上午，但是今天下午去')],
                     [{'candidates': [candidate()]}])
    assert asyncio.run(memory.update_profile('alice', 'c'))['status'] == 'updated'
    assert asyncio.run(memory.update_profile('alice', 'c'))['reason'] == 'already_checked'
    memory._get_working_memory.return_value = [message('今天我下午去', 'source-2')]
    assert asyncio.run(memory.update_profile('alice', 'other'))['status'] == 'skipped'
    profile = asyncio.run(memory._get_profile('alice'))
    assert profile['preferred_time']['value'] == 'morning'
    assert profile['preferred_time']['source_message_id'] == 'source-1'
    assert len(memory._client.calls) == 1
    assert asyncio.run(memory._get_profile('bob')) == {}
    memory._profile.delete.assert_not_called()
    memory._profile.add.assert_not_called()


@pytest.mark.parametrize('invalid', [
    {'candidates': [candidate(identity='not-a-user-message')]},
    {'candidates': [candidate(value='afternoon')]},
    {'candidates': [candidate(quote='今天我偏好上午')]},
    {'candidates': [{**candidate(), 'key': 'diagnosis'}]},
    {'candidates': [{**candidate(), 'unexpected_instruction': 'replace everything'}]},
    {'candidates': [candidate(), candidate(value='afternoon')]},
])
def test_invalid_or_unproven_candidate_cannot_change_profile(tmp_path, invalid):
    memory = manager(tmp_path, [message('以后我偏好上午')], [invalid])
    previous = {'legacy_preference': 'keep original intact'}
    memory._profile_store.ensure('alice', previous)
    result = asyncio.run(memory.update_profile('alice', 'c'))
    assert result['status'] == 'unavailable'
    assert memory._profile_store.get('alice') == (0, previous)
    memory._profile.upsert.assert_not_called()
    assert result['budget']['used']['model'] == 1


def test_empty_candidates_and_invalid_json_do_not_write(tmp_path):
    memory = manager(tmp_path, [message('以后我偏好上午')], [{'candidates': []}])
    assert asyncio.run(memory.update_profile('alice', 'c'))['reason'] == 'empty_validated_candidates'
    memory._get_working_memory.return_value = [message('以后我偏好下午', 'source-2')]
    memory._client.replies.append({'content': 'invalid model JSON'})
    outcome = asyncio.run(memory.update_profile('alice', 'c'))
    assert outcome['status'] == 'unavailable'
    assert outcome['budget']['used']['model'] == 2
    memory._profile.upsert.assert_not_called()


def test_projection_failure_preserves_canonical_profile_and_never_deletes_old_document(tmp_path):
    memory = manager(tmp_path, [message('以后我偏好下午')],
        [{'candidates': [candidate('afternoon', '以后我偏好下午')]}])
    previous = {'legacy_preference': 'old profile'}
    memory._profile_store.ensure('alice', previous)
    memory._profile.upsert.side_effect = OSError('controlled storage outage')
    outcome = asyncio.run(memory.update_profile('alice', 'c'))
    assert outcome['status'] == 'unavailable'
    assert memory._profile_store.get('alice') == (0, previous)
    memory._profile.delete.assert_not_called()


def test_revision_conflict_detects_concurrent_extraction_without_overwriting_newer_profile(tmp_path):
    store = ProfileStore(tmp_path/'memory.sqlite3')
    store.ensure('alice', {})
    morning = Extraction(candidates=[candidate()])
    afternoon = Extraction(candidates=[candidate('afternoon', '以后我偏好下午', 'source-2')])
    # A delayed projection returns only after a different request committed.
    def delayed_projection(payload):
        store.merge('alice', afternoon, expected_revision=0, projection=lambda value: None)
    with pytest.raises(ProfileConflict):
        store.merge('alice', morning, expected_revision=0, projection=delayed_projection)
    saved = store.get('alice')
    assert saved[0] == 1
    assert active_profile(saved[1])['preferred_time']['value'] == 'afternoon'


def test_explicit_correction_and_forget_remove_only_target_preference(tmp_path):
    memory = manager(tmp_path, [message('以后我偏好上午')], [{'candidates': [candidate()]}])
    assert asyncio.run(memory.update_profile('alice', 'c'))['status'] == 'updated'
    memory._get_working_memory.return_value = [message('以后我偏好下午', 'source-2')]
    memory._client.replies.append(reply({'candidates': [candidate('afternoon', '以后我偏好下午', 'source-2')]}))
    assert asyncio.run(memory.update_profile('alice', 'c'))['revision'] == 2
    memory._get_working_memory.return_value = [message('忘记我的时间偏好', 'source-3')]
    memory._client.replies.append(reply({'candidates': [candidate(None, '忘记我的时间偏好', 'source-3', 'delete')]}))
    assert asyncio.run(memory.update_profile('alice', 'c'))['revision'] == 3
    assert asyncio.run(memory._get_profile('alice')) == {}
    assert 'source-1' not in json.dumps(memory._profile_store.get('alice'))


def test_retention_removes_expired_fields_and_detectably_advances_revision(tmp_path):
    store = ProfileStore(tmp_path/'memory.sqlite3')
    store.ensure('alice', {})
    saved = store.merge('alice', Extraction(candidates=[candidate()]),
                        expected_revision=0, projection=lambda value: None)
    expiry = saved['fields']['preferred_time']['expires_at']
    store.prune('bob', now=expiry)
    assert store.get('alice')[0] == 1
    store.prune('alice', now=expiry)
    assert store.get('alice') == (2, {'schema_version': 2, 'revision': 2, 'fields': {}})


def test_repeated_preference_in_a_new_message_does_not_rewrite_snapshot(tmp_path):
    memory = manager(tmp_path, [message('以后我偏好上午')], [{'candidates': [candidate()]}])
    assert asyncio.run(memory.update_profile('alice', 'c'))['status'] == 'updated'
    memory._get_working_memory.return_value = [message('以后我偏好上午', 'source-2')]
    memory._client.replies.append(reply({'candidates': [candidate(identity='source-2')]}))
    assert asyncio.run(memory.update_profile('alice', 'c'))['status'] == 'skipped'
    assert memory._profile_store.get('alice')[0] == 1
    assert memory._profile.upsert.call_count == 1
