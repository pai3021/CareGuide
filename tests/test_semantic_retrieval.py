"""Native Chroma integration with TEST ONLY vectors, not a semantic quality claim."""
from datetime import date
from types import SimpleNamespace

import pytest

from careguide.corpus import Corpus, DocumentKind
from careguide.retrieval.embedding_provider import OpenAICompatibleEmbeddingFunction
from careguide.retrieval.knowledge_base import MedicalKnowledgeBase, index_collection


class ScriptedTestEmbedding:
    model = 'scripted-test-only'
    dimensions = 4

    def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, query):
        return [1.0, 0.0, 0.0, 0.0]


def test_embedding_adapter_uses_medical_instruction_and_validates_vectors():
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[1.0,0.0,0.0,0.0])])

    provider = OpenAICompatibleEmbeddingFunction(api_key='test-only', base_url='https://example.invalid/v1',
        model='test-only', dimensions=4, client=SimpleNamespace(embeddings=SimpleNamespace(create=create)))
    provider.embed_query('挂号材料')
    assert 'sourced medical information' in calls[0]['input'][0]
    assert 'operations' not in calls[0]['input'][0]
    assert provider(['挂号材料'])[0].tolist() == [1.0, 0.0, 0.0, 0.0]
    with pytest.raises(RuntimeError, match='non-finite'):
        provider._validate_vector([float('nan'),0.0,0.0,0.0])


@pytest.mark.parametrize('mode', ['dense', 'hybrid'])
def test_native_chroma_query_filters_scope_and_is_read_only(tmp_path, mode):
    chromadb = pytest.importorskip('chromadb')
    client = chromadb.PersistentClient(path=str(tmp_path), settings=chromadb.Settings(anonymized_telemetry=False))
    corpus, embedding = Corpus(), ScriptedTestEmbedding()
    collection = index_collection(client, corpus, embedding)
    with pytest.raises(ValueError, match='already exists'):
        index_collection(client, corpus, embedding)
    knowledge = MedicalKnowledgeBase(corpus, collection=collection, embedding=embedding, mode=mode)
    before = collection.get(include=['documents','metadatas','embeddings'])
    candidates = [r for r in corpus.records if r.kind == DocumentKind.HOSPITAL_GUIDE and r.hospital_id == 'pkufh']
    hits = knowledge.search('挂号流程缴费', candidates, today=date(2026,9,17))
    assert hits
    assert all(h['scope']['hospital_id'] == 'pkufh' for h in hits)
    after = collection.get(include=['documents','metadatas','embeddings'])
    assert before['ids'] == after['ids']
    assert before['documents'] == after['documents']
    assert before['metadatas'] == after['metadatas']
    assert (before['embeddings'] == after['embeddings']).all()
    assert knowledge.search('挂号', candidates, today=date(2027,1,1)) == []
    wrong_embedding = ScriptedTestEmbedding()
    wrong_embedding.model = 'different-model'
    with pytest.raises(ValueError, match='metadata mismatch'):
        MedicalKnowledgeBase(corpus, collection=collection, embedding=wrong_embedding, mode=mode)


def test_dense_mode_never_silently_falls_back_to_hash():
    with pytest.raises(ValueError, match='real embedding provider'):
        MedicalKnowledgeBase(Corpus(), mode='hybrid')


@pytest.mark.parametrize('constant', [10, 60])
@pytest.mark.parametrize('explicit_weights', [None, {'dense': 4., 'bm25': 1.}])
def test_hybrid_rrf_uses_configured_weights(tmp_path, constant, explicit_weights):
    chromadb = pytest.importorskip('chromadb')
    client = chromadb.PersistentClient(
        path=str(tmp_path), settings=chromadb.Settings(anonymized_telemetry=False))
    corpus, embedding = Corpus(), ScriptedTestEmbedding()
    collection = index_collection(client, corpus, embedding)
    knowledge = MedicalKnowledgeBase(
        corpus, collection=collection, embedding=embedding, mode='hybrid',
        min_dense_score=-1, dense_weight=4, lexical_weight=1,
        rrf_constant=constant, hybrid_weights=explicit_weights)
    hits = knowledge.search('挂号流程缴费', list(corpus.records),
                            today=date(2026, 9, 17), top_k=2)
    assert hits
    weights_key = 'fusion_weights' if explicit_weights is not None else 'rrf_weights'
    assert hits[0]['retrieval'][weights_key] == {'dense': 4.0, 'bm25': 1.0}
    assert hits[0]['retrieval']['rrf_constant'] == constant
    for hit in hits:
        info = hit['retrieval']
        expected = sum(weight/(constant+info[key]) for key, weight in
                       [('dense_rank', 4.), ('bm25_rank', 1.)] if key in info)
        assert info['rank_score'] == pytest.approx(expected)
    with pytest.raises(ValueError, match='weights must be positive'):
        MedicalKnowledgeBase(
            corpus, collection=collection, embedding=embedding, mode='hybrid',
            dense_weight=0)


@pytest.mark.parametrize('pool_size', [2, 4])
def test_expanded_pool_reaches_reranker_but_final_top_k_stays_fixed(tmp_path, pool_size):
    chromadb = pytest.importorskip('chromadb')
    client = chromadb.PersistentClient(path=str(tmp_path), settings=chromadb.Settings(anonymized_telemetry=False))
    corpus, embedding = Corpus(), ScriptedTestEmbedding()
    collection = index_collection(client, corpus, embedding)
    received = []

    class Reranker:
        model = 'test-only'

        def rerank(self, query, candidates, *, top_n):
            received.extend(key for key, _ in candidates)
            assert top_n == 1
            return [candidates[-1][0]], [.8], {}

    knowledge = MedicalKnowledgeBase(corpus, collection=collection, embedding=embedding,
        mode='hybrid', reranker=Reranker(), rerank_candidates=pool_size)
    candidates = [record for record in corpus.records if corpus.fresh(record, date(2026, 9, 17))]
    hits = knowledge.search('头痛', candidates, today=date(2026, 9, 17), top_k=1)
    assert len(received) == pool_size
    assert len(hits) == 1 and hits[0]['record_id'] == received[-1]
    assert set(received) <= {record.id for record in candidates}


@pytest.mark.parametrize('constant', [0, -1, True, 1.5])
def test_invalid_rrf_constant_is_rejected(constant):
    with pytest.raises(ValueError, match='positive integer'):
        MedicalKnowledgeBase(Corpus(), rrf_constant=constant)


def test_dense_runtime_can_apply_dedicated_reranker(tmp_path):
    chromadb = pytest.importorskip('chromadb')
    client = chromadb.PersistentClient(
        path=str(tmp_path), settings=chromadb.Settings(anonymized_telemetry=False))
    corpus, embedding = Corpus(), ScriptedTestEmbedding()
    collection = index_collection(client, corpus, embedding)

    class Reranker:
        model = 'test-reranker'

        def rerank(self, query, candidates, *, top_n):
            identities = list(reversed([identity for identity, _ in candidates]))[:top_n]
            return identities, [0.9 - index * 0.1 for index in range(len(identities))], {}

    knowledge = MedicalKnowledgeBase(
        corpus, collection=collection, embedding=embedding, mode='dense',
        reranker=Reranker(), rerank_candidates=2)
    candidates = list(corpus.records[:3])
    hits = knowledge.search('头痛', candidates, today=date(2026, 9, 17), top_k=1)
    assert len(hits) == 1
    assert hits[0]['retrieval']['mode'] == 'dense_rerank'
    assert hits[0]['retrieval']['rerank_model'] == 'test-reranker'


@pytest.mark.parametrize('fusion', ['rrf', 'minmax'])
def test_three_leg_runtime_keeps_scope_and_reranker_boundary(tmp_path, fusion):
    chromadb = pytest.importorskip('chromadb')
    client = chromadb.PersistentClient(path=str(tmp_path), settings=chromadb.Settings(anonymized_telemetry=False))
    corpus, embedding = Corpus(), ScriptedTestEmbedding()
    collection = index_collection(client, corpus, embedding)
    candidates = list(corpus.records[:2])
    seen = {}

    class Sparse:
        def search(self, query, *, eligible_ids, top_k):
            seen['eligible'] = eligible_ids
            return [('foreign-record', 999.), (candidates[1].id, .9), (candidates[0].id, .2)]

    class Reranker:
        model = 'test-only'

        def rerank(self, query, values, *, top_n):
            seen['rerank'] = [key for key, _ in values]
            return [candidates[1].id], [.8], {}

    knowledge = MedicalKnowledgeBase(corpus, collection=collection, embedding=embedding,
        mode='hybrid', sparse_retriever=Sparse(), hybrid_fusion=fusion,
        hybrid_weights={'dense': .6, 'bm25': .1, 'bge_sparse': .3},
        reranker=Reranker(), rerank_candidates=2)
    hits = knowledge.search('挂号流程', candidates, today=date(2026, 9, 17), top_k=1)
    assert seen['eligible'] == {r.id for r in candidates}
    assert set(seen['rerank']) <= seen['eligible']
    assert hits[0]['record_id'] == candidates[1].id
    assert hits[0]['retrieval']['bge_sparse_score'] == .9
    assert hits[0]['retrieval']['mode'] == 'hybrid_'+fusion+'_rerank'
    hits[0]['retrieval']['fusion_weights']['dense'] = 99
    assert knowledge.hybrid_weights['dense'] == .6


def test_zero_dense_weight_does_not_spend_embedding_calls(tmp_path):
    chromadb = pytest.importorskip('chromadb')
    client = chromadb.PersistentClient(path=str(tmp_path), settings=chromadb.Settings(anonymized_telemetry=False))
    corpus, embedding = Corpus(), ScriptedTestEmbedding()
    collection = index_collection(client, corpus, embedding)
    target = corpus.records[0]

    def forbidden(*args):
        pytest.fail('Disabled dense branch must not call embedding provider')

    embedding.embed_query = forbidden

    class Sparse:
        def search(self, query, *, eligible_ids, top_k):
            return [(target.id, .8)]

    knowledge = MedicalKnowledgeBase(corpus, collection=collection, embedding=embedding,
        mode='hybrid', sparse_retriever=Sparse(),
        hybrid_weights={'dense': 0, 'bm25': 0, 'bge_sparse': 1})
    assert knowledge.search('test', [target], today=date(2026, 9, 17))[0]['record_id'] == target.id
