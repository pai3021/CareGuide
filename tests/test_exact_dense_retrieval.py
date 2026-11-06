from datetime import date

import numpy as np
import pytest

from careguide.corpus import Corpus, DocumentKind
from careguide.retrieval.exact_index import ExactCosineIndex
from careguide.retrieval.knowledge_base import MedicalKnowledgeBase, index_collection
from tests.test_semantic_retrieval import ScriptedTestEmbedding


def test_exact_scope_filter_runs_before_topk_and_rejects_unknown_ids():
    index = ExactCosineIndex(['outside', 'inside'], [[1, 0], [.1, .9]], dimensions=2)
    hits = index.search([1, 0], eligible_ids={'inside'}, top_k=1)
    assert hits[0][0] == 'inside' and hits[0][1] < .2
    with pytest.raises(ValueError, match='Unknown'):
        index.search([1, 0], eligible_ids={'unknown'}, top_k=1)
    with pytest.raises(ValueError, match='dimensions'):
        index.search([1, 0, 0], eligible_ids={'inside'}, top_k=1)


@pytest.mark.parametrize('mode', ['dense', 'hybrid'])
def test_native_exact_search_preserves_scope_freshness_and_readonly_index(tmp_path, monkeypatch, mode):
    chromadb = pytest.importorskip('chromadb')
    client = chromadb.PersistentClient(path=str(tmp_path), settings=chromadb.Settings(anonymized_telemetry=False))
    corpus, embedding = Corpus(), ScriptedTestEmbedding()
    collection = index_collection(client, corpus, embedding)
    before = collection.get(include=['documents', 'metadatas', 'embeddings'])
    calls = []
    monkeypatch.setattr(embedding, 'embed_query', lambda query: calls.append(query) or [1, 0, 0, 0])
    knowledge = MedicalKnowledgeBase(corpus, collection=collection, embedding=embedding, mode=mode, dense_search='exact')
    candidates = [r for r in corpus.records if r.kind == DocumentKind.HOSPITAL_GUIDE and r.hospital_id == 'pkufh']
    hits = knowledge.search('挂号流程', candidates, today=date(2026, 9, 17), top_k=1)
    assert hits and len(calls) == 1
    assert all(h['scope']['hospital_id'] == 'pkufh' and h['retrieval']['dense_search'] == 'exact' for h in hits)
    assert knowledge.search('挂号', candidates, today=date(2027, 1, 1)) == []
    assert len(calls) == 1
    after = collection.get(include=['documents', 'metadatas', 'embeddings'])
    assert before['ids'] == after['ids'] and before['metadatas'] == after['metadatas']
    np.testing.assert_array_equal(before['embeddings'], after['embeddings'])


def test_exact_collection_rejects_invalid_vector_matrix_and_search_mode():
    with pytest.raises(ValueError):
        ExactCosineIndex(['a'], [[0, 0]], dimensions=2)
    with pytest.raises(ValueError):
        ExactCosineIndex(['a', 'a'], [[1, 0], [1, 0]], dimensions=2)
    with pytest.raises(ValueError, match='Dense search'):
        MedicalKnowledgeBase(Corpus(), dense_search='unknown')
