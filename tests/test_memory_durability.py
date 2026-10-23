import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from careguide.conversation_memory import MemoryManager, Message, MsgRole


def test_redis_identity_components_cannot_collide_through_delimiters():
    for key in (MemoryManager._wm_key, MemoryManager._summary_key):
        assert key('alice:family', 'conversation') != key('alice', 'family:conversation')
        assert key('alice%3Afamily', 'conversation') != key('alice:family', 'conversation')
    assert MemoryManager._wm_key('alice', 'conversation') == 'careguide:wm:alice:conversation'


def test_episode_failure_preserves_working_messages(monkeypatch):
    manager = MemoryManager.__new__(MemoryManager)
    manager._redis = SimpleNamespace(get=AsyncMock(return_value='previous summary'),
        setex=AsyncMock(), delete=AsyncMock())
    manager._client, manager._model = object(), 'test'
    manager._get_working_memory = AsyncMock(return_value=[
        Message(role=MsgRole.USER, content=str(index)) for index in range(15)])
    manager._merge_summary = AsyncMock(return_value='merged summary')
    manager._store_episodic = AsyncMock(return_value=False)
    monkeypatch.setattr('careguide.conversation_memory.create_chat_completion',
        AsyncMock(return_value={'choices': [{'message': {'content': 'summary'}}]}))
    asyncio.run(manager._compress('test-user', 'test-conversation'))
    manager._redis.delete.assert_not_called()
    manager._redis.setex.assert_not_called()


def test_summary_failure_preserves_working_messages(monkeypatch):
    manager = MemoryManager.__new__(MemoryManager)
    manager._redis = SimpleNamespace(delete=AsyncMock())
    manager._client, manager._model = object(), 'test'
    manager._get_working_memory = AsyncMock(return_value=[
        Message(role=MsgRole.USER, content=str(index)) for index in range(15)])
    manager._store_episodic = AsyncMock()
    monkeypatch.setattr('careguide.conversation_memory.create_chat_completion',
        AsyncMock(side_effect=TimeoutError))
    asyncio.run(manager._compress('test-user', 'test-conversation'))
    manager._store_episodic.assert_not_called()
    manager._redis.delete.assert_not_called()
