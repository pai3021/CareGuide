import pytest
from scripts.evaluate_rag_cached_rewrites import expand_lexical, guarded_rewrites


def test_cached_rewrite_guard_preserves_numeric_facts_without_claiming_semantic_equivalence():
    assert guarded_rewrites('服药１０天', ['服药10天后', '服药20天后']) == ['服药10天后']
    assert guarded_rewrites('宝宝3岁发烧', ['宝宝发烧', '三岁宝宝发烧']) == []
    with pytest.raises(ValueError):
        guarded_rewrites('query', ['only one'])


def test_zero_weight_reproduces_original_and_no_invented_candidates():
    original = [('a', 5.), ('b', 2.)]
    rewritten = [[('c', 8.), ('a', 2.)]]
    assert expand_lexical(original, rewritten, 0) == original
    assert set(k for k, _ in expand_lexical(original, rewritten, .5)) == {'a', 'b', 'c'}
    assert expand_lexical(original, [], 1) == original
