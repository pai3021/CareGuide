from scripts.run_careguide_kuake_rag_rewrite_evaluation import deduplicate


def test_multi_query_deduplication_preserves_query_and_rank_order():
    result = deduplicate([
        ["original-1", "shared", "original-3"],
        ["shared", "rewrite-1"],
        ["rewrite-2", "original-1"],
    ], limit=5)
    assert result == ["original-1", "shared", "original-3", "rewrite-1", "rewrite-2"]


def test_multi_query_deduplication_respects_pool_limit():
    assert deduplicate([["a", "b"], ["c", "d"]], limit=3) == ["a", "b", "c"]
