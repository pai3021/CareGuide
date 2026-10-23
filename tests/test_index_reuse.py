import hashlib
from types import SimpleNamespace

import pytest

from careguide.corpus import Corpus
from careguide.retrieval.index_reuse import reuse_verified_index
from careguide.retrieval.knowledge_base import MedicalKnowledgeBase


def fixture():
    corpus = Corpus()
    records = {record.id: record for record in corpus.records}
    binding = {'fingerprint': 'original-manifest', 'count': len(records), 'retrieval_hashes': {
        key: hashlib.sha256(MedicalKnowledgeBase.retrieval_text(record).encode()).hexdigest()
        for key, record in records.items()}}
    class Collection:
        def __init__(self, name, metadata):
            self.name, self.metadata, self.saved = name, metadata, {}
        def get(self, *, include, ids=None):
            keys = list(records) if ids is None else ids
            if self.name != 'original':
                return {'ids': list(self.saved)}
            return {'ids': keys, 'documents': [records[key].content for key in keys],
                    'metadatas': [{'record_id': key, 'source_id': records[key].source_id,
                                  'kind': records[key].kind.value} for key in keys],
                    'embeddings': [[.1, .2] for _ in keys]}
        def add(self, **kwargs):
            assert self.name != 'original'
            self.saved.update(zip(kwargs['ids'], kwargs['embeddings']))
        def modify(self, *, metadata=None, name=None):
            assert self.name != 'original'
            if metadata is not None:
                assert 'hnsw:space' not in metadata, 'Chroma rejects changing distance metadata'
                self.metadata = metadata
            if name:
                assert name not in collections
                collections.pop(self.name)
                self.name = name
                collections[name] = self
    source = Collection('original', {'corpus_fingerprint': binding['fingerprint'],
        'embedding_model': 'existing-real-model', 'embedding_dimensions': 2, 'hnsw:space': 'cosine'})
    collections = {'original': source}
    def create(name, metadata, **kwargs):
        assert name not in collections
        collections[name] = Collection(name, metadata)
        return collections[name]
    client = SimpleNamespace(list_collections=lambda: list(collections.values()),
        get_collection=lambda name, **kw: collections[name], create_collection=create)
    return client, corpus, source, binding, collections


def test_reuses_verified_vectors_without_embedding_or_modifying_source():
    client, corpus, source, binding, collections = fixture()
    result = reuse_verified_index(client, corpus, source.name, binding, model='existing-real-model', dimensions=2, batch_size=3)
    assert result['records'] == len(corpus.records)
    assert result['paid_calls'] == 0
    assert not source.saved
    assert collections[result['target']].metadata['reuse_complete']
    with pytest.raises(ValueError, match='overwrite'):
        reuse_verified_index(client, corpus, source.name, binding, model='existing-real-model', dimensions=2)


def test_query_runtime_refuses_incomplete_reuse_even_with_target_name():
    client, corpus, source, binding, collections = fixture()
    result = reuse_verified_index(client, corpus, source.name, binding, model='existing-real-model', dimensions=2)
    target = collections[result['target']]
    target.metadata['reuse_complete'] = False
    with pytest.raises(ValueError, match='has not completed'):
        MedicalKnowledgeBase(corpus, collection=target, mode='hybrid',
            embedding=SimpleNamespace(model='existing-real-model', dimensions=2))


@pytest.mark.parametrize('fault', ['changed_input', 'wrong_fingerprint', 'wrong_model', 'wrong_dimensions', 'invalid_vector'])
def test_rejects_unproven_reuse_before_creating_target(fault):
    client, corpus, source, binding, collections = fixture()
    if fault == 'changed_input':
        binding['retrieval_hashes'][corpus.records[0].id] = 'changed'
    elif fault == 'wrong_fingerprint':
        binding['fingerprint'] = 'unrelated'
    elif fault == 'wrong_model':
        source.metadata['embedding_model'] = 'another-model'
    elif fault == 'wrong_dimensions':
        source.metadata['embedding_dimensions'] = 3
    else:
        original = source.get
        def invalid(**kwargs):
            rows = original(**kwargs)
            rows['embeddings'][0] = [float('nan'), .2]
            return rows
        source.get = invalid
    with pytest.raises(ValueError):
        reuse_verified_index(client, corpus, source.name, binding, model='existing-real-model', dimensions=2)
    assert list(collections) == ['original']
