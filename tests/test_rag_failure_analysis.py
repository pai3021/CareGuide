"""Loss-stage evidence and reranker experiment contracts, not semantic-quality tests."""
import pytest

from scripts.analyze_rag_failures import trace
from scripts.compare_rag_rerank_models import ARMS, ENGLISH_INSTRUCTION


@pytest.mark.parametrize('streams,pool,final,stage', [
    ({'dense': [('a', .9)]}, ['a'], ['a'], 'absent_from_retrieval_union'),
    ({'dense': [('g', .9)]}, ['a'], ['a'], 'removed_at_pool_cutoff'),
    ({'dense': [('g', .9)]}, ['a','g'], ['a'], 'inside_pool_outside_final10'),
    ({'dense': [('g', .9)]}, ['a','g'], ['g'], 'final10_hit'),
])
def test_stage_trace_separates_unretrieved_truncated_and_demoted(streams,pool,final,stage):
    value = trace({'gold_record_id': 'g', 'streams': streams},pool,final)
    assert value['stage'] == stage


def test_instruction_model_and_pool_arms_change_one_factor_at_a_time():
    cn, small, large, expanded = ARMS.values()
    assert cn['model'] == small['model'] and cn['pool_size'] == small['pool_size']
    assert small['instruction'] == large['instruction'] == expanded['instruction'] == ENGLISH_INSTRUCTION
    assert small['pool_size'] == large['pool_size'] == 40
    assert large['model'] == expanded['model'] == 'Qwen/Qwen3-Reranker-4B'
    assert expanded['pool_size'] == 100
    assert not any('gold' in value['instruction'] for value in ARMS.values())


@pytest.mark.parametrize('instruction', ['', ENGLISH_INSTRUCTION])
def test_runtime_explicit_instruction_preserves_blank_default(monkeypatch, instruction):
    from careguide.runtime import semantic_reranker
    from careguide.retrieval.reranker import DEFAULT_RERANK_INSTRUCTION
    monkeypatch.setenv('CAREGUIDE_RERANK_API_KEY', 'test-only')
    monkeypatch.setenv('CAREGUIDE_RERANK_MODEL', 'Qwen/Qwen3-Reranker-4B')
    monkeypatch.setenv('CAREGUIDE_RERANK_INSTRUCTION', instruction)
    adapter = semantic_reranker()
    assert adapter.model == 'Qwen/Qwen3-Reranker-4B'
    assert adapter.instruction == (instruction or DEFAULT_RERANK_INSTRUCTION)
