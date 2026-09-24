"""Isolated real-provider Redis/Chroma memory gate; never a clinical quality score."""
import argparse
import asyncio
import json
import os
import uuid
from pathlib import Path


async def run(args):
    import chromadb
    import redis.asyncio as redis
    from careguide.conversation_memory import MemoryManager, MsgRole
    from careguide.llm_utils import create_llm_client
    from careguide.runtime import semantic_embedding
    from scripts.validate_real_provider import source_hashes
    before = source_hashes()
    connection = redis.from_url('redis://127.0.0.1:6380/0', decode_responses=True)
    assert await connection.ping()
    chroma = chromadb.HttpClient(host='127.0.0.1', port=8002,
        settings=chromadb.Settings(anonymized_telemetry=False))
    llm = create_llm_client(os.environ['CAREGUIDE_LLM_API_KEY'], os.environ['CAREGUIDE_LLM_BASE_URL'])
    embedding = semantic_embedding()
    prefix = 'careguide_validation_' + uuid.uuid4().hex[:12]
    user, conv = prefix, 'synthetic-appointment'
    options = dict(client=llm, model=os.environ['CAREGUIDE_LLM_MODEL'], redis_client=connection,
        chroma_client=chroma, embedding_function=embedding, collection_prefix=prefix)
    manager = MemoryManager(**options)
    report = {'checks': [], 'redis': 'real', 'chroma': 'real', 'summary_model': 'real_provider',
        'embedding': 'real_provider', 'embedding_dimensions': embedding.dimensions,
        'clinical_validity_evaluated': False, 'semantic_quality_evaluated': False,
        'heldout_evaluated': False, 'redis_disk_persistence_evaluated': False}
    def passed(name):
        report['checks'].append(name)
        print('PASS ' + name, flush=True)
    try:
        for index in range(16):
            await manager.add_message(user, conv, MsgRole.USER,
                f'我希望查询北京大学国际医院本院的挂号流程，偏好线上查询。第{index + 1}次补充：只需挂号材料和流程信息。')
        context = await manager.get_context(user, conv, '线上挂号需要哪些材料？')
        assert len(context.recent_messages) == 6 and context.summary and context.relevant_history
        assert '摘要生成失败' not in context.summary
        assert all('摘要生成失败' not in item for item in context.relevant_history)
        assert manager._episodic.count() > 0
        vectors = manager._episodic.get(include=['embeddings'])['embeddings']
        assert len(vectors[0]) == embedding.dimensions
        passed('real_summary_compression_and_semantic_episode_retrieval')
        assert await connection.ttl(manager._wm_key(user, conv)) > 0
        passed('working_memory_ttl')
        await manager.update_profile(user, conv)
        context = await manager.get_context(user, conv, '挂号材料')
        assert context.user_profile.get('preferences')
        passed('real_profile_extraction')
        cross = await manager.get_context(user, 'new-conversation', '线上挂号材料')
        assert cross.relevant_history and cross.user_profile and not cross.recent_messages
        passed('same_user_cross_conversation_retrieval')
        other = await manager.get_context(user + '-other', conv, '线上挂号材料')
        assert not other.recent_messages and not other.relevant_history and not other.user_profile
        passed('user_isolation')
        reopened = MemoryManager(**options)
        restored = await reopened.get_context(user, conv, '挂号材料')
        assert restored.summary == context.summary and restored.relevant_history and restored.user_profile
        passed('manager_recreation_retains_backend_memory')
        assert source_hashes() == before
        passed('frozen_corpus_and_dataset_unchanged')
        report['passed'] = len(report['checks'])
    except Exception as error:
        report['error_type'] = type(error).__name__
        raise
    finally:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x', encoding='utf-8') as file:
            json.dump(report, file, ensure_ascii=False, indent=2)
        await connection.delete(manager._wm_key(user, conv), manager._summary_key(user, conv))
        for suffix in ('_episodic', '_user_profile'):
            chroma.delete_collection(prefix + suffix)
        await manager.close()
        await llm.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-file')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error('Refusing to overwrite an existing report')
    from scripts.private_config import load_config
    load_config(args.config_file)
    os.environ['NO_PROXY'] = '127.0.0.1,localhost'
    asyncio.run(run(args))


if __name__ == '__main__':
    main()
