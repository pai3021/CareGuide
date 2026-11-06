"""Index preservation and embedding checkpoint contracts; no remote inference."""
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from careguide.corpus import Corpus
from careguide.retrieval.knowledge_base import versioned_collection_name
from scripts.compare_rag_embeddings import cache_vectors, experiment_dir, text_hash, require_endpoint
from scripts.snapshot_rag_index import inspect_index, tree_hashes


@pytest.mark.parametrize('path', ['data/chroma','tmp/rag-recall-20261105/chroma-current',
                                'data/source_cache/qwen-embedding-baselines/06b-1024-chroma-20261106'])
def test_new_index_workdir_cannot_be_an_existing_baseline(path):
    with pytest.raises(ValueError,match='separate'):
        experiment_dir(Path(path))


def test_vector_checkpoint_rejects_changed_text_foreign_ids_and_invalid_dimensions():
    db=sqlite3.connect(':memory:')
    db.execute('create table vectors(kind text,id text,text_sha256 text,vector blob)')
    db.execute('insert into vectors values (?,?,?,?)',('doc','a',text_hash('original'),np.ones(1024,dtype='<f4').tobytes()))
    assert set(cache_vectors(db,[('a','original')],'doc'))=={'a'}
    with pytest.raises(ValueError,match='text invalid'):
        cache_vectors(db,[('a','changed')],'doc')
    with pytest.raises(ValueError,match='Foreign'):
        cache_vectors(db,[],'doc')
    db.execute('update vectors set vector=?',(np.ones(10,dtype='<f4').tobytes(),))
    with pytest.raises(ValueError,match='invalid'):
        cache_vectors(db,[('a','original')],'doc')
    db.close()


def test_old_index_inspection_is_readonly_and_checks_actual_record_ids(tmp_path):
    corpus=Corpus()
    path=tmp_path/'chroma.sqlite3'
    with sqlite3.connect(path) as db:
        db.executescript('create table collections(id text,name text,dimension integer);'
            'create table collection_metadata(collection_id text,key text,str_value text,int_value integer,float_value real,bool_value integer);'
            'create table segments(id text,collection text);create table embeddings(segment_id text,embedding_id text);')
        db.execute('insert into collections values (?,?,?)',('c',versioned_collection_name(corpus),1024))
        db.executemany('insert into collection_metadata values (?,?,?,?,?,?)',[
            ('c','embedding_model','Qwen/Qwen3-Embedding-0.6B',None,None,None),
            ('c','embedding_dimensions',None,1024,None,None),
            ('c','corpus_fingerprint',corpus.fingerprint,None,None,None),
            ('c','reuse_complete',None,None,None,1)])
        db.execute('insert into segments values (?,?)',('s','c'))
        db.executemany('insert into embeddings values (?,?)',[('s',r.id) for r in corpus.records])
    before=tree_hashes(tmp_path)
    assert inspect_index(tmp_path,corpus)['record_count']==len(corpus.records)
    assert tree_hashes(tmp_path)==before


@pytest.mark.parametrize('endpoint',['http://api.siliconflow.cn/v1','https://other.invalid/v1',
                                    'https://api.siliconflow.cn/v1?x=1'])
def test_embedding_experiment_is_limited_to_authorized_provider(endpoint):
    with pytest.raises(ValueError,match='scoped'):
        require_endpoint(endpoint)


@pytest.mark.parametrize('configured', ['', 'data/source_cache/independent-test/chroma'])
def test_runtime_can_select_independent_index_without_overwriting_old_default(monkeypatch, configured):
    import chromadb
    from careguide.runtime import PROJECT_ROOT, chroma_client
    captured={}
    def create(**kwargs):
        captured.update(kwargs)
        return 'test-client'
    monkeypatch.setattr(chromadb,'PersistentClient',create)
    monkeypatch.setenv('CAREGUIDE_CHROMA_PATH',configured)
    assert chroma_client()=='test-client'
    expected=PROJECT_ROOT/configured if configured else PROJECT_ROOT/'data/chroma'
    assert Path(captured['path'])==expected.resolve()
