from careguide.evaluation.kuake_ir_dense import ranking_metrics, reciprocal_rank_fusion
from scripts.run_kuake_ir_dense_evaluation import select_items


def test_reciprocal_rank_fusion_rewards_cross_retriever_agreement():
    result = reciprocal_rank_fusion(["lexical", "shared", "a"],
                                    ["dense", "shared", "b"])
    assert result[0] == "shared"
    assert set(result) == {"lexical", "dense", "shared", "a", "b"}


def test_weighted_rrf_can_prefer_the_stronger_retriever():
    result = reciprocal_rank_fusion(
        ["lexical"], ["dense"], constant=60, weights=(1.0, 4.0))
    assert result[0] == "dense"


def test_weighted_rrf_rejects_invalid_weights():
    import pytest
    with pytest.raises(ValueError, match="weights"):
        reciprocal_rank_fusion(["a"], ["b"], weights=(1.0,))


def test_ranking_metrics_measure_single_relevant_document():
    result = ranking_metrics([["g1", "x"], ["x", "g2"], ["x"]],
                             ["g1", "g2", "g3"])
    assert result["recall_at_1"] == 1 / 3
    assert result["recall_at_5"] == 2 / 3
    assert result["mrr_at_10"] == 0.5


def test_fast_query_sample_is_deterministic_and_not_a_prefix():
    queries = {f"q{i:03d}": f"query {i}" for i in range(100)}
    first, mode = select_items(queries, sample_size=20, sample_seed="fixed")
    second, second_mode = select_items(queries, sample_size=20, sample_seed="fixed")
    assert first == second
    assert mode == second_mode == "deterministic_hash_sample"
    assert first != list(queries.items())[:20]
    assert len({identity for identity, _ in first}) == 20
