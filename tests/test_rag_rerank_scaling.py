import pytest

from scripts.compare_rag_rerank_scaling import ARMS, partition, select_arm, validate_response


def test_tuning_selection_does_not_consume_better_validation_outcomes():
    rows = {'t': {'gold_record_id': 'g'}, 'v': {'gold_record_id': 'h'}}
    completed = [{'case_id': 't', 'arm': arm, 'ids': ['g'] if arm == '4b80_retrieval' else ['x']}
                 for arm in ARMS]
    completed += [{'case_id': 'v', 'arm': arm, 'ids': ['h'] if arm == '8b80_retrieval' else ['x']}
                  for arm in ARMS]
    selected, _ = select_arm(['t'], completed, rows)
    assert selected == '4b80_retrieval'


def test_selection_requires_all_tuning_arms_and_prefers_cheaper_tie():
    rows = {'t': {'gold_record_id': 'g'}}
    completed = [{'case_id': 't', 'arm': arm, 'ids': ['g']} for arm in ARMS]
    assert select_arm(['t'], completed, rows)[0] == '4b80_current'
    with pytest.raises(ValueError):
        select_arm(['t'], completed[:-1], rows)


def test_top10_only_or_duplicate_response_cannot_be_reported_as_complete_pool():
    pool = [str(i) for i in range(80)]
    row = {'candidate_ids': pool, 'ids': pool[:10], 'scores': [1.]*10}
    with pytest.raises(ValueError):
        validate_response(row, pool)
    row = {'candidate_ids': pool, 'ids': pool, 'scores': [1.]*80}
    validate_response(row, pool)
    with pytest.raises(ValueError):
        validate_response({**row, 'ids': ['0']*80}, pool)


def test_partition_is_outcome_independent_disjoint_and_order_invariant():
    keys = [f'case-{i}' for i in range(100)]
    tune, valid = partition(keys)
    assert len(tune) == len(valid) == 50
    assert not set(tune) & set(valid)
    assert (tune, valid) == partition(list(reversed(keys)))


def test_model_and_instruction_comparisons_change_one_factor():
    current, retrieval, large = ARMS.values()
    assert current['model'] == retrieval['model']
    assert current['instruction'] != retrieval['instruction']
    assert retrieval['instruction'] == large['instruction']
    assert large['model'] == 'Qwen/Qwen3-Reranker-8B'
