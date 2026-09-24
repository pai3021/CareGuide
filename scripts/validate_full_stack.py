"""Real provider + native Chroma integration in disposable medical stores."""
import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.config_file.is_file():
        parser.error('Select an existing authorized config and a new report path')
    from dotenv import dotenv_values
    for name, value in dotenv_values(args.config_file, interpolate=False).items():
        if name.startswith('CAREGUIDE_') and value is not None:
            os.environ[name] = value
    from scripts.validate_real_provider import preflight, source_hashes
    if not preflight(os.environ)['ready']:
        raise SystemExit('Provider configuration unavailable')
    from fastapi.testclient import TestClient
    from careguide.api import app
    from careguide.corpus import Corpus
    from careguide.retrieval.knowledge_base import index_collection
    import careguide.runtime as runtime
    import chromadb
    checks = []
    before = source_hashes()
    with TemporaryDirectory(prefix='careguide-full-stack-', ignore_cleanup_errors=True) as directory:
        os.environ.update(CAREGUIDE_BUSINESS_DB=directory+'/medical.sqlite3',
            CAREGUIDE_EVAL_DIR=directory+'/evaluations', CAREGUIDE_PERSIST_TRACES='false',
            CAREGUIDE_MEMORY_MODE='session', CAREGUIDE_RETRIEVAL_MODE='hybrid',
            CAREGUIDE_INTENT_EMBEDDING_MODE='remote')
        client = chromadb.PersistentClient(path=directory+'/chroma', settings=chromadb.Settings(anonymized_telemetry=False))
        collection = index_collection(client, Corpus(), runtime.semantic_embedding())
        assert collection.count() == len(Corpus().records)
        checks.append('real_embedding_native_chroma_index')
        print(json.dumps({'completed': checks[-1]}), flush=True)
        runtime.chroma_client = lambda: client
        runtime.get_registry.cache_clear(); runtime.get_chat_runtime.cache_clear()
        with TestClient(app) as api:
            assert api.get('/capabilities').json()['retrieval_mode'] == 'hybrid'
            reply = api.post('/chat', json={'message': '查询精氨酸布洛芬片0.2g说明书，查询后转人工协助。',
                'user_id': 'synthetic-validation', 'conv_id': 'integration'}).json()
            assert reply['workflow_status'] == 'escalated' and reply['evidence']
            assert not reply['intent_source_scores']['llm']['failed']
            assert not reply['intent_source_scores']['embedding']['failed']
            trace = api.get('/trace/tool/'+reply['request_id']).json()['trace']
            assert len(trace['execution']['execution_waves']) == 2 and trace['side_effects']
            identity = reply['handoff']['id']
            checks.append('real_provider_two_wave_handoff')
            print(json.dumps({'completed': checks[-1]}), flush=True)
            repeated = api.post('/chat', json={'message': '请转人工协助',
                'user_id': 'synthetic-validation', 'conv_id': 'integration'}).json()
            assert repeated['handoff']['id'] == identity
            assert not api.get('/trace/tool/'+repeated['request_id']).json()['trace']['side_effects']
            checks.append('handoff_reuse_without_duplicate_write')
            assert api.get('/handoffs/'+identity, params={'user_id': 'other-user'}).status_code == 404
            for status in ('in_progress', 'resolved'):
                assert api.post('/handoffs/'+identity+'/status', json={'user_id': 'synthetic-validation', 'status': status}).status_code == 200
            checks.append('owned_handoff_status_lifecycle')
            search = api.post('/search', params={'query': '头痛就诊前记录哪些信息', 'rewrite': True, 'rerank': True}).json()
            assert search['success'] and search['results'] and search['reranked']
            checks.append('real_hybrid_rewrite_rerank')
            print(json.dumps({'completed': checks[-1]}), flush=True)
            corpus = Corpus(); record = corpus.records[0]; source = corpus._sources[record.source_id]
            submission = api.post('/knowledge/upload', json={'sources': [source.model_dump(mode='json')],
                'records': [record.model_copy(update={'id': 'integration-copy-existing-fact'}).model_dump(mode='json')]}).json()
            assert api.get('/knowledge/stats').json()['total_chunks'] == len(corpus.records)
            assert api.post('/knowledge/'+submission['id']+'/publish').status_code == 409
            assert api.post('/knowledge/'+submission['id']+'/review', json={'reviewer': 'synthetic-reviewer',
                'approve': True, 'note': 'Integration copy of an existing verified source; no new clinical fact'}).status_code == 200
            assert api.post('/knowledge/'+submission['id']+'/publish').status_code == 200
            assert api.get('/knowledge/stats').json()['total_chunks'] == len(corpus.records)+1
            checks.append('reviewed_publication_with_versioned_real_index')
            print(json.dumps({'completed': checks[-1]}), flush=True)
            evaluated = api.post('/eval/run', json={'limit': 1, 'judge': True,
                'dialogs': [['查询精氨酸布洛芬片0.2g说明书注意事项']]}).json()
            assert evaluated['heldout_used'] is False and not evaluated['baseline_promoted']
            assert evaluated['report']['total'] == 2
            assert api.post('/eval/baseline/'+evaluated['id']).status_code == 200
            checks.append('real_dev_evaluation_judge_and_explicit_baseline')
            print(json.dumps({'completed': checks[-1]}), flush=True)
        # Chroma clients keep SQLite handles alive on Windows; close before the
        # disposable workspace is removed, with no global runtime shutdown.
        client._system.stop()
    unchanged = source_hashes() == before
    assert unchanged, 'Frozen source/dataset bytes changed'
    report = {'passed': True, 'checks': checks, 'external_calls_made': True,
        'runtime': 'native Chroma + FastAPI test transport + real providers',
        'source_and_dataset_bytes_unchanged': unchanged,
        'heldout_used': False, 'clinical_validity_evaluated': False,
        'persistent_redis_memory_validated': False,
        'knowledge_test_data': 'duplicate of an existing verified fact; disposable only'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        import traceback
        print(json.dumps({'passed': False, 'failure_type': type(error).__name__,
            'failure_locations': [{'file': Path(frame.filename).name, 'line': frame.lineno}
                                  for frame in traceback.extract_tb(error.__traceback__)]}), flush=True)
        raise SystemExit(1)
