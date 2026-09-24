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


def test_hybrid_rrf_uses_configured_weights(tmp_path):
    chromadb = pytest.importorskip('chromadb')
    client = chromadb.PersistentClient(
        path=str(tmp_path), settings=chromadb.Settings(anonymized_telemetry=False))
    corpus, embedding = Corpus(), ScriptedTestEmbedding()
    collection = index_collection(client, corpus, embedding)
    knowledge = MedicalKnowledgeBase(
        corpus, collection=collection, embedding=embedding, mode='hybrid',
        min_dense_score=-1, dense_weight=4, lexical_weight=1)
    hits = knowledge.search('挂号流程缴费', list(corpus.records),
                            today=date(2026, 9, 17), top_k=2)
    assert hits
    assert hits[0]['retrieval']['rrf_weights'] == {'dense': 4.0, 'bm25': 1.0}
    with pytest.raises(ValueError, match='weights must be positive'):
        MedicalKnowledgeBase(
            corpus, collection=collection, embedding=embedding, mode='hybrid',
            dense_weight=0)


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
