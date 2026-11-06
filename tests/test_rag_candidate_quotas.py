import pytest
from scripts.evaluate_rag_candidate_quotas import quota_pool


def test_quota_preserves_lexical_candidate_with_fixed_budget_and_unique_ids():
    row = {'streams': {'dense': [('a', .9), ('b', .8), ('c', .7)], 'bm25': [('x', 4), ('a', 3)]}}
    assert set(quota_pool(row, 3, 1, 'dense')) == {'a', 'b', 'x'}
    assert len(quota_pool(row, 3, 2, 'dense')) == 3
    assert set(quota_pool(row, 3, 0, 'dense')) == {'a', 'b', 'c'}


def test_quota_rejects_over_budget_reservation():
    with pytest.raises(ValueError):
        quota_pool({'streams': {}}, 3, 4, 'dense')
