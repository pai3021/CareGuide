"""Protect query/document token boundaries and original BM25 equivalence."""
import pytest

from careguide.retrieval.bm25_index import BM25Index, tokenize_knowledge_text
from scripts.evaluate_rag_lexical_variants import PostingBM25, check_baseline, tokens


def test_extended_ngrams_never_cross_punctuation_or_duplicate_isolated_chars():
    assert tokens('甲，乙；丙丁戊', '123') == ['甲', '乙', '丙丁', '丁戊', '丙', '丁', '戊', '丙丁戊']
    assert tokens('支原体感染', '2') == tokenize_knowledge_text('支原体感染')
    assert '支原体' in tokens('支原体感染', '23')
    assert '乙丙' not in tokens('乙，丙', '123')


@pytest.mark.parametrize('query', ['支原体感染', 'N-123 胸痛', '甲', '不存在的英文xyz', '', 'AB-1 与 CD-2'])
def test_posting_scorer_matches_native_bm25_including_match_gates(query):
    docs = {'a': '支原体感染 支原体检查', 'b': 'N-123 胸痛 和疼痛', 'c': '甲，乙丙',
            'd': 'AB-1 检测 CD-2', 'e': '支原体感染 低烧 咳嗽 '*5}
    native = BM25Index()
    native.upsert([{'chunk_id': k, 'content': v, 'retrieval_text': v, 'metadata': {}} for k, v in docs.items()])
    expected = [(v.chunk_id, v.score) for v in native.search(query, top_k=100)]
    check_baseline(expected, PostingBM25(docs, '2').search(query))


def test_baseline_check_rejects_changed_ids_or_scores():
    with pytest.raises(ValueError):
        check_baseline([('a', 1)], [('b', 1)])
    with pytest.raises(ValueError):
        check_baseline([('a', 1)], [('a', 2)])
