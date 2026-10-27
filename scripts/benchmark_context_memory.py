"""Reproducible synthetic control-flow comparison, NOT a semantic-quality benchmark.

Run: python -m scripts.benchmark_context_memory --output <new-report.json>
Optional --real-services uses isolated Redis/Chroma namespaces and test embeddings.
No provider keys or hospital APIs are used. Existing user records are not modified.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
import uuid

from careguide.archived_memory import ArchivedMemory
from careguide.context_memory import ContextSettings, estimate_tokens
from careguide.conversation_memory import MemoryManager
from careguide.profile_memory import ProfileStore
from careguide.task_context import PersistentMemory, SessionMemory


class FixtureModel:
    def __init__(self):
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **kwargs):
        self.calls += 1
        return {'choices': [{'finish_reason': 'stop', 'message': {'content':
            '目标：查询预约。约束：人民医院、上午。关键实体：本人。已确认决策：尚未挂号。'
            '未解决问题：仍需确认。来源：合成对话测试输入。'}}]}


class FixtureRedis:
    def __init__(self):
        self.data = {}

    async def lpush(self, key, value):
        self.data.setdefault(key, []).insert(0, value)

    async def lrange(self, key, start, end):
        return self.data.get(key, [])[start:None if end == -1 else end+1]

    async def llen(self, key):
        return len(self.data.get(key, []))

    async def expire(self, key, ttl):
        return True

    async def get(self, key):
        return self.data.get(key)

    async def setex(self, key, ttl, value):
        self.data[key] = value

    async def delete(self, *keys):
        for key in keys:
            self.data.pop(key, None)

    async def aclose(self):
        pass

    def pipeline(self, transaction=True):
        operations = []
        connection = self
        class Pipeline:
            def __getattr__(self, name):
                def enqueue(*args):
                    operations.append((name, args))
                    return self
                return enqueue

            async def execute(self):
                for name, args in operations:
                    await getattr(connection, name)(*args)
        return Pipeline()


class FixtureEmbedding:
    def __call__(self, input):
        import numpy as np
        return [np.array([1.0, float(len(text) % 7)/10, .1], dtype=np.float32) for text in input]


def fixture_manager(path, model, redis_client):
    manager = MemoryManager.__new__(MemoryManager)
    manager._client, manager._model, manager._redis = model, 'synthetic-fixture', redis_client
    manager._profile_store = ProfileStore(path)
    manager._profile = None
    manager.legacy_profile_projection = False
    manager.archive_store = None
    manager.token_context_enabled = False
    async def store_episode(*args, **kwargs):
        return True
    async def search_episode(*args, **kwargs):
        return []
    manager._store_episodic = store_episode
    manager._search_episodic = search_episode
    return manager


async def measure(mode, directory):
    model, connection = FixtureModel(), FixtureRedis()
    manager = fixture_manager(directory/f'{mode}.sqlite3', model, connection)
    memory = SessionMemory() if mode == 'session_baseline' else PersistentMemory(manager)
    if mode == 'enhanced':
        memory = ArchivedMemory(memory, directory/f'{mode}.sqlite3', client=model, model=manager._model,
            settings=ContextSettings(window=8192, input_cap=2400, output_reserve=256, margin=128,
                                     summary_tokens=256))
    counts, started = [], time.perf_counter()
    for index in range(30):
        text = f'第{index}轮，人民医院上午预约仍需确认。'+('补充历史信息。'*(3 if index % 3 else 55))
        history = await memory.history('fixture-user', 'fixture-conversation', query=text)
        payload = [{'role': 'system', 'content': '只有用户确认后才可执行预约。'}]+history+[
            {'role': 'user', 'content': text}]
        if mode == 'enhanced':
            with memory.request_context('fixture-user', 'fixture-conversation', str(index)) as policy:
                payload = await policy.prepare(payload, output=256)
        counts.append(estimate_tokens({'messages': payload, 'tools': []}))
        await memory.add('fixture-user', 'fixture-conversation', 'user', text)
        await memory.add('fixture-user', 'fixture-conversation', 'assistant', '仅查询，尚未执行预约。')
    if mode == 'enhanced':
        raw_count = len(memory.store.messages('fixture-user', 'fixture-conversation'))
        original_store = memory.store
        await memory.close()
        memory = ArchivedMemory(PersistentMemory(fixture_manager(directory/f'{mode}.sqlite3', model, FixtureRedis())),
                                directory/f'{mode}.sqlite3', client=model, model='synthetic-fixture',
                                settings=ContextSettings(window=8192, input_cap=2400, output_reserve=256,
                                                         margin=128, summary_tokens=256))
        restored = await memory.history('fixture-user', 'fixture-conversation')
        assert raw_count == 60 and len(original_store.messages('fixture-user', 'fixture-conversation')) == 60
        assert restored and '尚未执行预约' in restored[-1]['content']
    else:
        raw_count = None
        connection.data.clear()
        memory = SessionMemory() if mode == 'session_baseline' else PersistentMemory(manager)
        restored = await memory.history('fixture-user', 'fixture-conversation')
    await memory.close()
    return {'mode': mode, 'turns': 30, 'estimated_input_tokens_sum': sum(counts),
            'estimated_input_tokens_peak': max(counts), 'summary_model_calls': model.calls,
            'archived_chat_messages': raw_count, 'context_present_after_cache_loss': bool(restored),
            'fixture_elapsed_ms': round((time.perf_counter()-started)*1000, 3)}


async def real_services(directory):
    import chromadb
    import redis.asyncio as redis
    prefix = 'careguide_context_validation_'+uuid.uuid4().hex[:12]
    user, conversation = prefix, 'synthetic'
    connection = redis.from_url('redis://127.0.0.1:6380/0', decode_responses=True,
                                socket_connect_timeout=2, socket_timeout=2)
    manager = None
    client = None
    try:
        assert await connection.ping()
        client = chromadb.HttpClient(host='127.0.0.1', port=8002,
                                    settings=chromadb.Settings(anonymized_telemetry=False))
        model = FixtureModel()
        manager = MemoryManager(client=model, model='synthetic-fixture', redis_client=connection,
            chroma_client=client, embedding_function=FixtureEmbedding(), collection_prefix=prefix,
            profile_path=str(directory/'profiles.sqlite3'))
        memory = ArchivedMemory(PersistentMemory(manager), directory/'sessions.sqlite3', client=model,
            model='synthetic-fixture', settings=ContextSettings(window=8192, input_cap=2400,
                                            output_reserve=256, margin=128, summary_tokens=256))
        for i in range(12):
            await memory.add(user, conversation, 'user' if i % 2 == 0 else 'assistant',
                             f'{i}：人民医院上午，尚未确认。'+'背景信息。'*55)
        expected = await memory.history(user, conversation, '医院')
        assert memory.store.snapshot(user, conversation)[0]['summary']
        assert manager._episodic.count() > 0
        await connection.delete(manager._wm_key(user, conversation), manager._summary_key(user, conversation))
        recovered = await memory.history(user, conversation, '医院')
        assert len(memory.store.messages(user, conversation)) == 12
        assert recovered and expected and await connection.ttl(manager._wm_key(user, conversation)) > 0
        assert await memory.history(user+'-other', conversation) == []
        return {'status': 'passed', 'checks': ['redis_key_loss_recovery', 'redis_ttl',
            'durable_summary', 'chroma_episode_projection', 'actor_isolation'],
            'embedding': 'synthetic test vectors', 'model': 'scripted fixture'}
    except Exception as exc:
        return {'status': 'unavailable_or_failed', 'error_type': type(exc).__name__}
    finally:
        if manager:
            await connection.delete(manager._wm_key(user, conversation), manager._summary_key(user, conversation))
        if client:
            names = [item.name if hasattr(item, 'name') else str(item) for item in client.list_collections()]
            if prefix+'_episodic' in names:
                client.delete_collection(prefix+'_episodic')
        await connection.aclose()


async def run(args):
    with TemporaryDirectory(prefix='context-benchmark-', dir='tmp') as temporary:
        directory = Path(temporary)
        rows = [await measure(mode, directory) for mode in ('session_baseline', 'redis_chroma_baseline', 'enhanced')]
        report = {'protocol': 'careguide-context-control-v1', 'seed': 'fixed-30-turn-transcript',
            'model': 'scripted fixture', 'token_count_method': 'UTF-8 heuristic; not provider usage',
            'episodic_retrieval': 'disabled in synthetic comparisons to isolate context behavior; checked separately with real services',
            'semantic_quality_evaluated': False, 'clinical_quality_evaluated': False,
            'hospital_actions_executed': False, 'rows': rows}
        report['source_sha256'] = {str(path).replace('\\', '/'): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path('careguide')/name for name in (
                'session_store.py', 'archived_memory.py', 'context_memory.py', 'conversation_memory.py',
                'profile_memory.py', 'task_context.py', 'llm_utils.py', 'api.py', 'runtime.py')]
            + [Path('scripts/benchmark_context_memory.py')]}
        if args.real_services:
            report['real_services'] = await real_services(directory)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x', encoding='utf-8') as file:
            json.dump(report, file, ensure_ascii=False, indent=2)
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--real-services', action='store_true')
    args = parser.parse_args()
    Path('tmp').mkdir(exist_ok=True)
    asyncio.run(run(args))
