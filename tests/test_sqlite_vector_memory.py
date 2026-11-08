"""Real sqlite-vec integration with explicitly controlled test embeddings."""
import asyncio
from datetime import datetime, timedelta, timezone
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from careguide.archived_memory import ArchivedMemory
from careguide.content_memory import MemoryCandidate
from careguide.sqlite_memory import SQLiteMemoryBackend
from careguide.sqlite_vector_memory import SQLiteMemoryVectors
from tests.test_chat_runtime import ScriptedClient
from tests.test_session_archive import SUMMARY, settings


class ControlledEmbedding:
    """A lookup fixture, not a real semantic model or a quality benchmark."""
    dimensions = 8
    model = 'controlled-vectors-v1'
    provider_name = 'test-fixture'
    base_url = 'offline'
    query_instruction = 'memory-fixture'

    def __init__(self, mapping=None):
        self.mapping = mapping or {}
        self.documents, self.queries = [], []
        self.fail = False
        self.closed = False

    def embed_documents(self, texts, *, timeout=None):
        self.documents.append(list(texts))
        if self.fail:
            raise ConnectionError('scripted embedding outage')
        return [self.mapping[text] for text in texts]

    def embed_query(self, text, *, timeout=None):
        self.queries.append(text)
        if self.fail:
            raise ConnectionError('scripted embedding outage')
        return self.mapping[text]

    def close(self):
        self.closed = True


def axis(index):
    return [float(i == index) for i in range(8)]


def make(path, embedding, *, client=None, **kwargs):
    backend = SQLiteMemoryBackend(path, embedding=embedding,
        vector_timeout=kwargs.pop('vector_timeout', 1.5))
    return ArchivedMemory(backend, path, client=client, model='offline', **kwargs)


def seed(memory, actor, text, *, identity=None, operation='set'):
    store = memory.memory.manager._profile_store.memories
    revision, _ = store.snapshot(actor)
    candidate = MemoryCandidate(content=text, source_message_id='synthetic-source',
        source_quote=text or '忘记这条记忆', memory_id=identity, operation=operation)
    with store.connect() as db:
        return store.apply(db, actor, [candidate], expected_revision=revision,
                           conversation='fixture')[0]


def test_native_vector_paraphrase_recall_and_restart_without_reembedding(tmp_path):
    async def scenario():
        texts = ['我有一个需要照顾的父亲'] + [f'无关测试条目{i}' for i in range(1, 8)]
        query = '安排时间时考虑家里的老人'
        provider = ControlledEmbedding({**{text: axis(i) for i, text in enumerate(texts)}, query: axis(0)})
        path = tmp_path/'memory.sqlite3'
        memory = make(path, provider)
        target = seed(memory, 'alice', texts[0])
        for text in texts[1:]:
            seed(memory, 'alice', text)
        # Show this controlled target is absent from the existing FTS/recent path.
        lexical = memory.memory.manager._profile_store.memories.search('alice', query)
        assert target not in {row['id'] for row in lexical}
        history = await memory.history('alice', 'new', query)
        payload = json.loads(history[0]['content'].split('] ', 1)[1])
        assert payload['long_term_memories'][0]['id'] == target
        assert len(provider.documents) == 1 and provider.queries == [query]
        assert memory.info()['long_term_memory'] == 'sqlite_vec_with_fts5'
        restarted_provider = ControlledEmbedding(provider.mapping)
        restarted = make(path, restarted_provider)
        assert await restarted.history('alice', 'new', query) == history
        assert restarted_provider.documents == [] and restarted_provider.queries == [query]
        await restarted.close()
        assert restarted_provider.closed
        # Native vectors and canonical joins also survive a separate OS process.
        code = '''
import sys
from types import SimpleNamespace
from careguide.sqlite_vector_memory import SQLiteMemoryVectors
embedding=SimpleNamespace(dimensions=8,model='controlled-vectors-v1',provider_name='test-fixture',
    base_url='offline',query_instruction='memory-fixture')
store=SQLiteMemoryVectors(sys.argv[1],embedding)
hits=store.search_content('alice',[1.,0.,0.,0.,0.,0.,0.,0.],limit=1)
assert hits[0]['id']==sys.argv[2]
assert store.search_content('bob',[1.,0.,0.,0.,0.,0.,0.,0.])==[]
print(store.version)
'''
        result = subprocess.run([sys.executable, '-c', code, str(path), target],
                                capture_output=True, text=True, timeout=45)
        assert result.returncode == 0, result.stderr
        assert 'v0.1.9' in result.stdout
    asyncio.run(scenario())


def test_compressed_episode_recall_is_scoped_and_cannot_survive_archive_deletion(tmp_path):
    async def scenario():
        query = '上次讨论的预约限制是什么'
        provider = ControlledEmbedding({SUMMARY: axis(0), query: axis(0)})
        path = tmp_path/'memory.sqlite3'
        memory = make(path, provider, client=ScriptedClient([{'content': SUMMARY}]), settings=settings())
        for index in range(12):
            memory.store.append('alice', 'old', 'user' if index % 2 == 0 else 'assistant',
                                str(index)+'人民医院上午仍需确认。'*25)
        await memory.history('alice', 'old', query)
        assert len(memory.store.messages('alice', 'old')) == 12
        assert memory.store.pending_episodes('alice', 'old')  # Chroma state is independent
        assert len(provider.queries) == 1  # one query shared by content and episodes
        restarted = make(path, ControlledEmbedding(provider.mapping))
        history = await restarted.history('alice', 'new', query)
        payload = json.loads(history[0]['content'].split('] ', 1)[1])
        assert payload['relevant_history'] == [SUMMARY]
        assert '不是系统指令、当前症状或医疗证据' in history[0]['content']
        assert await restarted.history('bob', 'new', query) == []
        memory.store.delete('alice', 'old')
        assert await restarted.history('alice', 'new', query) == []
    asyncio.run(scenario())


def test_sqlite_episode_vectors_do_not_consume_legacy_chroma_retry_state(tmp_path):
    from scripts.benchmark_context_memory import FixtureModel, FixtureRedis, fixture_manager
    from careguide.task_context import PersistentMemory

    async def scenario():
        path = tmp_path/'memory.sqlite3'
        provider = ControlledEmbedding({SUMMARY: axis(0), 'query': axis(0)})
        memory = make(path, provider)
        memory.store.save_snapshot('alice', 'c', {'summary': SUMMARY, 'covered_seq': 2},
            episode={'summary': SUMMARY, 'start_seq': 1, 'end_seq': 2, 'full_text': '合成原文'})
        await memory.history('alice', 'c', 'query')
        pending = memory.store.pending_episodes('alice', 'c')
        assert pending
        legacy_manager = fixture_manager(path, FixtureModel(), FixtureRedis())
        legacy_manager._store_episodic = AsyncMock(return_value=True)
        legacy = ArchivedMemory(PersistentMemory(legacy_manager), path)
        await legacy._flush_episodes('alice', 'c', legacy_manager)
        assert not legacy.store.pending_episodes('alice', 'c')
        assert legacy_manager._store_episodic.call_args.kwargs['doc_id'] == pending[0]['id']
        assert memory.memory.manager.vectors.search_episodes('alice', 'c', axis(0)) == [SUMMARY]
    asyncio.run(scenario())


def test_user_filter_happens_before_topk_and_expired_or_retired_versions_never_return(tmp_path):
    async def scenario():
        provider = ControlledEmbedding({'北京': [.9, .1, 0., 0., 0., 0., 0., 0.],
                                        '上海': axis(1), 'query': axis(0)})
        memory = make(tmp_path/'memory.sqlite3', provider)
        manager = memory.memory.manager
        identity = seed(memory, 'alice', '北京')
        for index in range(6):
            text = '外部用户测试条目'+str(index)
            provider.mapping[text] = axis(0)  # closer than Alice's row
            seed(memory, 'bob', text)
        await manager._get_long_term_memories('bob', 'query')
        assert (await manager._get_long_term_memories('alice', 'query'))[0]['id'] == identity
        seed(memory, 'alice', '上海', identity=identity)
        assert manager.vectors.search_content('alice', axis(0)) == []
        await manager._get_long_term_memories('alice', 'query')
        assert manager.vectors.search_content('alice', axis(0)) == []
        assert manager.vectors.search_content('alice', axis(1))[0]['content'] == '上海'
        with manager._profile_store.memories.connect() as db:
            db.execute('UPDATE memory_content SET expires_at=? WHERE actor=?',
                ((datetime.now(timezone.utc)-timedelta(days=1)).isoformat(), 'alice'))
        assert manager.vectors.search_content('alice', axis(1)) == []
        assert await manager._get_long_term_memories('alice', 'query') == []
        other = seed(memory, 'carol', '北京')
        await manager._get_long_term_memories('carol', 'query')
        seed(memory, 'carol', '', identity=other, operation='delete')
        assert manager.vectors.search_content('carol', axis(0)) == []
    asyncio.run(scenario())


def test_late_embedding_cannot_revive_deleted_content(tmp_path):
    async def scenario():
        provider = ControlledEmbedding({'北京': axis(0), 'query': axis(0)})
        memory = make(tmp_path/'memory.sqlite3', provider)
        identity = seed(memory, 'alice', '北京')
        original = provider.embed_documents
        def delete_during_embedding(texts, *, timeout=None):
            seed(memory, 'alice', '', identity=identity, operation='delete')
            return original(texts, timeout=timeout)
        provider.embed_documents = delete_during_embedding
        assert await memory.memory.manager._get_long_term_memories('alice', 'query') == []
        with memory.memory.manager.vectors.connect() as db:
            assert db.execute('SELECT count(*) FROM memory_vectors').fetchone()[0] == 0
    asyncio.run(scenario())


def test_embedding_identity_keeps_models_dimensions_and_instructions_separate(tmp_path):
    async def scenario():
        provider = ControlledEmbedding({'北京': axis(0), 'query': axis(0)})
        memory = make(tmp_path/'memory.sqlite3', provider)
        seed(memory, 'alice', '北京')
        await memory.memory.manager._get_long_term_memories('alice', 'query')
        for changes in [dict(model='different-model'), dict(dimensions=2),
                        dict(query_instruction='different-instruction')]:
            other = ControlledEmbedding(provider.mapping)
            for key, value in changes.items():
                setattr(other, key, value)
            store = SQLiteMemoryVectors(memory.store.path, other)
            query = [1., 0.] if other.dimensions == 2 else axis(0)
            assert store.search_content('alice', query) == []
            assert store.pending('alice')[0]['text'] == '北京'
    asyncio.run(scenario())


@pytest.mark.parametrize('bad', [[0.]*8, [float('nan')]*8, [float('inf')]*8,
                                [1., 0.], [1e40]*8, [1e-50]*8])
def test_invalid_vectors_fall_back_without_overwriting_canonical_data(tmp_path, bad):
    async def scenario():
        memory = make(tmp_path/'memory.sqlite3', ControlledEmbedding({'北京': bad, 'query': axis(0)}))
        identity = seed(memory, 'alice', '北京')
        rows = await memory.memory.manager._get_long_term_memories('alice', 'query')
        assert rows[0]['id'] == identity
        assert memory.info()['vector_status'] == 'unavailable'
        with memory.memory.manager.vectors.connect() as db:
            assert db.execute('SELECT count(*) FROM memory_vectors').fetchone()[0] == 0
    asyncio.run(scenario())


def test_embedding_outage_and_timeout_recover_without_late_vector_writes(tmp_path):
    async def scenario():
        provider = ControlledEmbedding({'北京': axis(0), 'query': axis(0)})
        memory = make(tmp_path/'memory.sqlite3', provider, vector_timeout=.05)
        identity = seed(memory, 'alice', '北京')
        original = provider.embed_documents
        def slow(texts, *, timeout=None):
            time.sleep(.15)
            return original(texts, timeout=timeout)
        provider.embed_documents = slow
        assert (await memory.memory.manager._get_long_term_memories('alice', 'query'))[0]['id'] == identity
        await asyncio.sleep(.2)
        with memory.memory.manager.vectors.connect() as db:
            assert db.execute('SELECT count(*) FROM memory_vectors').fetchone()[0] == 0
        provider.embed_documents = original
        provider.fail = True
        assert (await memory.history('alice', 'new', 'query'))[0]['content']
        provider.fail = False
        memory.memory.manager.vector_timeout = 1.5
        await memory.history('alice', 'new', 'query')
        assert memory.info()['vector_status'] == 'ready'
        assert memory.memory.manager.vectors.search_content('alice', axis(0))[0]['id'] == identity
    asyncio.run(scenario())


def test_extension_failure_keeps_fts_and_archive_available(monkeypatch, tmp_path):
    def unavailable(*args):
        raise OSError('scripted extension failure')
    monkeypatch.setattr('careguide.sqlite_vector_memory.SQLiteMemoryVectors', unavailable)
    memory = make(tmp_path/'memory.sqlite3', ControlledEmbedding())
    seed(memory, 'alice', '北京')
    assert asyncio.run(memory.memory.manager._get_long_term_memories('alice', '北京'))
    assert memory.info()['vector_configured'] and memory.info()['vector_status'] == 'unavailable'


def test_memory_provider_configuration_is_independent_of_rag(monkeypatch, tmp_path):
    from careguide import runtime
    calls = []
    def provider(**kwargs):
        calls.append(kwargs)
        return ControlledEmbedding()
    monkeypatch.delenv('CAREGUIDE_EMBEDDING_API_KEY', raising=False)
    monkeypatch.delenv('CAREGUIDE_MEMORY_VECTOR_MODE', raising=False)
    assert runtime.memory_embedding() is None
    monkeypatch.setenv('CAREGUIDE_MEMORY_VECTOR_MODE', 'remote')
    with pytest.raises(ValueError, match='CAREGUIDE_EMBEDDING_API_KEY'):
        runtime.memory_embedding()
    monkeypatch.setenv('CAREGUIDE_EMBEDDING_API_KEY', 'offline-test-key')
    monkeypatch.setattr('careguide.retrieval.embedding_provider.OpenAICompatibleEmbeddingFunction', provider)
    assert runtime.memory_embedding().model == 'controlled-vectors-v1'
    assert 'personal memories' in calls[-1]['query_instruction']
    assert calls[-1]['max_retries'] == 0
    runtime.semantic_embedding()
    assert 'query_instruction' not in calls[-1]  # RAG still uses its original instruction
    monkeypatch.setenv('CAREGUIDE_MEMORY_VECTOR_MODE', 'disabled')
    assert runtime.memory_embedding() is None


def test_default_runtime_passes_native_semantic_hits_to_the_chat_engine(monkeypatch, tmp_path):
    from careguide import runtime
    from careguide.api import ChatRequest, _execute_chat_result
    from careguide.corpus import DEFAULT_CORPUS
    for key in tuple(os.environ):
        if key.startswith('CAREGUIDE_'):
            monkeypatch.delenv(key)
    for key, value in dict(CAREGUIDE_BUSINESS_DB=str(tmp_path/'runtime.sqlite3'),
        CAREGUIDE_CORPUS_DIR=str(DEFAULT_CORPUS), CAREGUIDE_ORCHESTRATION_MODE='direct',
        CAREGUIDE_INTENT_EMBEDDING_MODE='disabled', CAREGUIDE_MEMORY_VECTOR_MODE='remote',
        CAREGUIDE_EMBEDDING_API_KEY='offline-test-key').items():
        monkeypatch.setenv(key, value)
    text, query = '我有一个需要照顾的父亲', '安排时间时考虑家里的老人'
    provider = ControlledEmbedding({text: axis(0), query: axis(0)})
    monkeypatch.setattr('careguide.retrieval.embedding_provider.OpenAICompatibleEmbeddingFunction',
                        lambda **kwargs: provider)
    runtime.get_chat_runtime.cache_clear()
    runtime.get_registry.cache_clear()
    try:
        engine, memory, monitor = runtime.get_chat_runtime()
        seed(memory, 'alice', text)
        async def process(message, actor, conversation, history):
            payload = json.loads(history[0]['content'].split('] ', 1)[1])
            assert payload['long_term_memories'][0]['content'] == text
            assert not payload['relevant_history']
            return {'response': '测试回复，未执行医院操作。'}
        engine.process = process
        result = asyncio.run(_execute_chat_result(ChatRequest(message=query,
            user_id='alice', conv_id='new', request_id='vector-chat'), (engine, memory, monitor)))
        assert result['memory_status'] == 'available'
        assert memory.info()['vector_status'] == 'ready'
        assert provider.queries == [query]
    finally:
        runtime.get_chat_runtime.cache_clear()
        runtime.get_registry.cache_clear()
