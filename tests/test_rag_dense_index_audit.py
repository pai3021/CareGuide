import numpy as np
import pytest

from scripts.audit_rag_dense_index import exact_top_k, unit_vectors


def test_exact_cosine_normalizes_magnitude_and_breaks_ties_by_id():
    ids = np.asarray(['z', 'a', 'b'])
    docs = unit_vectors([[100, 0], [1, 0], [0, 2]])
    found = exact_top_k(ids, docs, unit_vectors([[3, 0], [0, 3]]), k=2)
    assert [r[0] for r in found[0]] == ['a', 'z']
    assert [r[0] for r in found[1]] == ['b', 'a']
    assert found[0][0][1] == pytest.approx(1)


@pytest.mark.parametrize('values', [[[0, 0]], [[float('nan'), 1]], [1, 2]])
def test_cosine_audit_rejects_missing_direction_or_invalid_matrix(values):
    with pytest.raises(ValueError):
        unit_vectors(values)


def test_exact_search_rejects_document_alignment_error():
    with pytest.raises(ValueError):
        exact_top_k(np.asarray(['same', 'same']), np.ones((2, 3)), np.ones((1, 3)), k=1)
