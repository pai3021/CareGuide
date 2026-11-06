"""Protocol and durable-budget tests; fixtures are not retrieval-quality evidence."""
import json

import pytest

from scripts.evaluate_rag_candidate_pools import BASELINE, choose_trial, partition_rows
from scripts.rerank_rag_candidate_plan import AttemptLedger, paired_interval, summarize


def test_fusion_selection_uses_tuning_objective_and_preserves_exact_baseline_on_ties():
    other = {'method': 'minmax', 'constant': 60, 'weights': {'dense': .8, 'bm25': .2}}
    assert choose_trial([other, BASELINE], [
        {'candidate_coverage': .8, 'mrr_at_10': .4},
        {'candidate_coverage': .8, 'mrr_at_10': .4}]) == 1
    assert choose_trial([BASELINE, other], [
        {'candidate_coverage': .8, 'mrr_at_10': .6},
        {'candidate_coverage': .9, 'mrr_at_10': .3}]) == 1


def test_partition_is_independent_of_retrieval_scores_and_input_order():
    rows = [{'id': str(i), 'gold_record_id': str(i), 'streams': {}} for i in range(10)]
    tuning, validation = partition_rows(rows)
    other = partition_rows([{**row, 'streams': {'dense': [('x', 999)]}} for row in reversed(rows)])
    assert [[row['id'] for row in values] for values in (tuning, validation)] == [
        [row['id'] for row in values] for values in other]
    assert not {row['id'] for row in tuning} & {row['id'] for row in validation}


def test_rerank_attempt_budget_counts_failures_survives_resume_and_locks_identity(tmp_path):
    path = tmp_path/'attempts.json'
    identity = {'plan_sha256': 'fixed', 'model': 'test-only'}
    ledger = AttemptLedger(path, identity, 2)
    ledger.reserve({'case_id': 'first', 'arm': 'baseline'})
    resumed = AttemptLedger(path, identity, 2)
    resumed.reserve({'case_id': 'first', 'arm': 'alternative'})
    with pytest.raises(RuntimeError, match='exhausted'):
        resumed.reserve({'case_id': 'second', 'arm': 'baseline'})
    assert len(json.loads(path.read_text())['attempts']) == 2
    with pytest.raises(ValueError, match='different'):
        AttemptLedger(path, {'plan_sha256': 'different'}, 2)
    with pytest.raises(ValueError, match='different'):
        AttemptLedger(path, identity, 3)


def test_query_paired_interval_uses_signed_absolute_deltas():
    assert paired_interval([0]*10)['bootstrap_95_interval'] == [0, 0]
    result = paired_interval([1, -1, 0, 0])
    assert result['absolute_delta'] == 0
    assert result['bootstrap_95_interval'][0] <= 0 <= result['bootstrap_95_interval'][1]


def test_rerank_replay_rejects_duplicated_or_foreign_responses():
    plan = {'jobs': [{'case_id': 'q', 'arm': 'baseline_rrf20', 'candidate_ids': ['a']}],
            'final_top_n': 10, 'arms': {'baseline_rrf20': {}}, 'case_ids': ['q']}
    rows = {'q': {'gold_record_id': 'a'}}
    record = {'case_id': 'q', 'arm': 'baseline_rrf20', 'candidate_ids': ['a'],
              'ids': ['a'], 'scores': [.8], 'seconds': .1, 'input_tokens': None}
    value = summarize(plan, rows, [record])
    assert value['results']['baseline_rrf20']['metrics']['recall_at_10'] == 1
    assert value['results']['baseline_rrf20']['total_input_tokens'] is None
    with pytest.raises(ValueError, match='incomplete'):
        summarize(plan, rows, [record, record])
    with pytest.raises(ValueError, match='Invalid cached'):
        summarize(plan, rows, [{**record, 'ids': ['foreign']}])
