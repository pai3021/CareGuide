"""Algorithm/budget regressions only; synthetic scores are not retrieval evidence."""
import json

import pytest

from careguide.evaluation.kuake_ir_dense import reciprocal_rank_fusion
from careguide.retrieval.score_fusion import fuse_scores, reserve_lexical_candidates
from careguide.retrieval.sparse_index import SparseLexicalIndex
from scripts.cache_rag_recall_scores import RequestBudget
from scripts.evaluate_rag_score_fusion import paired_recall, weight_grid


def test_rrf_keeps_existing_rank_semantics_and_weights():
    streams = {'dense': [('a', .9), ('b', .89), ('c', .1)],
               'bm25': [('b', 100.), ('c', 2.), ('a', 1.)]}
    result = fuse_scores(streams, weights={'dense': 4, 'bm25': 1})
    assert [key for key, _ in result] == reciprocal_rank_fusion(
        ['a', 'b', 'c'], ['b', 'c', 'a'], weights=[4, 1])
    assert result[0][0] == 'a'


def test_minmax_uses_score_gaps_and_is_affine_invariant():
    streams = {'dense': [('a', .9), ('b', .89), ('c', .1)],
               'bm25': [('b', 100.), ('c', 2.), ('a', 1.)]}
    result = fuse_scores(streams, weights={'dense': 1, 'bm25': 1}, method='minmax')
    assert result[0][0] == 'b'
    changed_units = {name: [(key, score*100-300) for key, score in stream]
                     for name, stream in streams.items()}
    changed = fuse_scores(changed_units, weights={'dense': 1, 'bm25': 1}, method='minmax')
    assert [key for key, _ in changed] == [key for key, _ in result]
    assert [score for _, score in changed] == pytest.approx([score for _, score in result])


def test_empty_constant_missing_and_disabled_legs_have_explicit_semantics():
    streams = {'dense': [('a', -.1), ('b', -.2)], 'bm25': [('c', 4), ('a', 4)],
               'sparse': []}
    result = fuse_scores(streams, weights={'dense': 1, 'bm25': 1, 'sparse': 1}, method='minmax')
    assert result == [('a', 1.0), ('b', 0.0), ('c', 0.0)]
    result = fuse_scores(streams, weights={'dense': 1, 'bm25': 0, 'sparse': 0}, method='minmax')
    assert [key for key, _ in result] == ['a', 'b']  # disabled leg cannot add candidates


@pytest.mark.parametrize('streams,weights', [
    ({'d': [('a', float('nan'))]}, {'d': 1}),
    ({'d': [('a', 1), ('a', 2)]}, {'d': 1}),
    ({'d': [('a', 1)]}, {'d': float('inf')}),
    ({'d': [('a', 1)]}, {'d': 0}),
    ({'d': [('a', 1)]}, {'other': 1}),
])
def test_invalid_fusion_inputs_fail_closed(streams, weights):
    with pytest.raises(ValueError):
        fuse_scores(streams, weights=weights)


def test_sparse_dot_product_and_scope_filter():
    index = SparseLexicalIndex({'a': {'123': 2, '8': 1}, 'b': {'123': 1}, 'c': {'9': 5}})
    assert index.search({'123': 3, '8': 2}) == [('a', 8.), ('b', 3.)]
    assert index.search({'123': 3}, eligible_ids={'b', 'c'}) == [('b', 3.)]
    assert index.search({'999': 1}) == []
    with pytest.raises(ValueError):
        index.search({'123': -1})


def test_request_budget_counts_attempt_before_response_and_survives_restart(tmp_path):
    path = tmp_path/'requests.json'
    budget = RequestBudget(path, 2)
    budget.reserve()  # even if the request subsequently fails, it stays consumed
    assert json.loads(path.read_text())['attempted_requests'] == 1
    resumed = RequestBudget(path, 2)
    resumed.reserve()
    with pytest.raises(RuntimeError, match='exhausted'):
        resumed.reserve()
    assert json.loads(path.read_text())['attempted_requests'] == 2


def test_simplex_contains_three_way_and_individual_controls():
    grid = list(weight_grid(['dense', 'bm25', 'bge_sparse']))
    assert len(grid) == 66
    assert all(sum(weights.values()) == pytest.approx(1) for weights in grid)
    assert {'dense': .6, 'bm25': .2, 'bge_sparse': .2} in grid


def test_paired_recall_reports_both_gains_and_regressions():
    rows = [{'gold_record_id': 'a'}, {'gold_record_id': 'b'}, {'gold_record_id': 'c'}]
    result = paired_recall(rows, [['a'], ['x'], ['c']], [['x'], ['b'], ['c']], 1)
    assert result == {'gained_queries': 1, 'lost_queries': 1, 'net_queries': 0, 'absolute_delta': 0.0}


def test_lexical_reservation_protects_original_pool_without_duplicate_or_unbounded_growth():
    ranking = [f'd{i}' for i in range(100)]
    pool = reserve_lexical_candidates(ranking, ['unique', 'd0', 'unique', 'outside_budget'])
    assert pool[:20] == ranking[:20]
    assert pool[20:] == ['unique']
    assert 'outside_budget' not in pool
    assert len(reserve_lexical_candidates(ranking, ['x', 'y', 'z'])) == 23
