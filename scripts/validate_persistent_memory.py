"""Real isolated Redis/Chroma integration; TEST vectors are not semantic evidence."""
import asyncio
import json
import os
import uuid
from pathlib import Path

import chromadb
import redis.asyncio as redis

from careguide.conversation_memory import MemoryManager, MsgRole


class TestEmbedding:
    def __call__(self, input):
        import numpy as np
        return [np.array([1.0, float(len(text)%7)/10, 0.1], dtype=np.float32) for text in input]


class ScriptedSummary:
    def __init__(self):
        from types import SimpleNamespace
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **kwargs):
        prompt = kwargs['messages'][-1]['content']
        content = json.dumps({'preferences': ['查询本院挂号流程'], 'entities': {'医院': ['测试医院']}}) if 'preferences' in prompt else 'SCRIPTED：测试用户讨论挂号流程。'
        return {'choices': [{'message': {'content': content}}]}


async def run():
    client = chromadb.HttpClient(host='127.0.0.1', port=8002,
                                settings=chromadb.Settings(anonymized_telemetry=False))
    connection = redis.from_url('redis://127.0.0.1:6380/0', decode_responses=True)
    assert await connection.ping()
    prefix = 'careguide_validation_'+uuid.uuid4().hex[:12]
    user, conv = prefix, 'synthetic-conversation'
    manager = MemoryManager(client=ScriptedSummary(), redis_client=connection, chroma_client=client,
                            embedding_function=TestEmbedding(), collection_prefix=prefix)
    checks = []
    try:
        for index in range(16):
            await manager.add_message(user, conv, MsgRole.USER, f'测试挂号流程问题{index}')
        context = await manager.get_context(user, conv, '挂号流程')
        assert len(context.recent_messages) == 6
        assert context.summary and context.relevant_history
        checks.append('compression_working_summary_and_episodic_retrieval')
        assert await connection.ttl(manager._wm_key(user, conv)) > 0
        checks.append('redis_ttl')
        await manager.update_profile(user, conv)
        context = await manager.get_context(user, conv, '挂号流程')
        assert context.user_profile.get('preferences')
        checks.append('profile_extraction_and_context_merge')
        other = await manager.get_context(user+'-other', conv, '挂号流程')
        assert not other.recent_messages and not other.relevant_history and not other.user_profile
        checks.append('user_isolation')
        assert manager._wm_key(user, conv).startswith('careguide:')
        checks.append('independent_redis_namespace')
        report = {'passed': len(checks), 'checks': checks, 'redis': 'real localhost Redis:6380',
                  'chroma': 'real localhost Chroma:8002', 'embedding': 'explicit TEST vectors',
                  'summary_and_profile_model': 'SCRIPTED fixture', 'semantic_quality_evaluated': False}
        output = Path(__file__).resolve().parents[1]/'data/eval/results/persistent_memory_validation.json'
        output.write_bytes((json.dumps(report, ensure_ascii=False, indent=2)+'\n').encode())
        print(json.dumps(report, ensure_ascii=False))
    finally:
        await connection.delete(manager._wm_key(user, conv), manager._summary_key(user, conv))
        for suffix in ('_episodic', '_user_profile'):
            client.delete_collection(prefix+suffix)
        await manager.close()


if __name__ == '__main__':
    os.environ['NO_PROXY'] = '127.0.0.1,localhost'
    asyncio.run(run())
