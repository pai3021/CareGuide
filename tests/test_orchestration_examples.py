import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def example(folder, scenario, phase='full'):
    environment = {key: value for key, value in os.environ.items() if not key.startswith('CAREGUIDE_')}
    environment.update(PYTHONUTF8='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', LANGSMITH_TRACING='false')
    outcome = subprocess.run([sys.executable, str(ROOT/'scripts/reproduce_orchestration.py'),
        '--scenario', scenario, '--phase', phase, '--state-dir', str(folder)],
        cwd=ROOT, env=environment, capture_output=True, encoding='utf-8', timeout=30)
    assert outcome.returncode == 0, outcome.stdout+'\n'+outcome.stderr
    return json.loads(outcome.stdout)


def test_actual_process_restart_resumes_checkpoint_and_budget_without_replaying_completed_query(tmp_path):
    first = example(tmp_path, 'wait', 'start')
    assert first['rounds'][0]['status'] == 'waiting_user'
    assert first['rounds'][0]['used']['tool'] == 1
    resumed = example(tmp_path, 'wait', 'resume')
    assert first['pid'] != resumed['pid']
    assert resumed['model_calls_this_process'] == 0
    final = resumed['rounds'][0]
    assert final['workflow_id'] == first['rounds'][0]['workflow_id']
    assert final['status'] == 'completed'
    assert final['used']['tool'] == 2
    assert final['used']['model'] == 1
    assert final['tasks']['doctors']['attempts'] == 1
    assert final['tasks']['drug']['attempts'] == 1
    assert final['trace_complete']


@pytest.mark.parametrize('scenario', ['dependency', 'replan', 'partial', 'memory'])
def test_documented_examples_are_executable_with_observable_results(tmp_path, scenario):
    result = example(tmp_path, scenario)
    assert result['paid_calls'] == 0
    final = result['rounds'][-1]
    if scenario == 'memory':
        assert result['model_calls'] == 2
        assert result['rounds'][1]['profile']['preferred_time']['value'] == 'morning'
        assert final['profile']['preferred_time']['value'] == 'afternoon'
    else:
        assert final['trace_complete']
        if scenario == 'partial':
            assert final['status'] == 'partial_completed'
            assert final['tasks']['slots']['attempts'] == 2
            assert final['tasks']['drug']['status'] == 'succeeded'
            assert final['used']['replan'] == 1
        elif scenario == 'replan':
            assert final['status'] == 'completed'
            assert final['plan_version'] == 2
            assert final['tasks']['drug']['revision'] == 1
            assert final['tasks']['doctors']['revision'] == 2
        elif scenario == 'dependency':
            assert final['status'] == 'completed'
            assert all(task['attempts'] == 1 for task in final['tasks'].values())
            assert final['used']['tool'] == 5
