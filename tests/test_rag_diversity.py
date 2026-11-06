import numpy as np
import pytest

from scripts.evaluate_rag_diversity import mmr_order


def test_mmr_can_replace_redundant_passage_without_changing_first_choice():
    scores = [.99, .98, .9]
    similarities = [[1, .99, .1], [.99, 1, .1], [.1, .1, 1]]
    assert mmr_order(scores, similarities, 1.) == [0, 1, 2]
    assert mmr_order(scores, similarities, .8) == [0, 2, 1]
    assert scores == [.99, .98, .9]


@pytest.mark.parametrize('scores,matrix,weight', [
    ([1, 2], np.eye(3), .9), ([float('nan')], [[1]], .9),
    ([1], [[1]], 1.1), ([1, 2], [[1, .2], [.3, 1]], .9),
])
def test_mmr_rejects_unaligned_or_nonfinite_evidence(scores, matrix, weight):
    with pytest.raises(ValueError):
        mmr_order(scores, matrix, weight)
