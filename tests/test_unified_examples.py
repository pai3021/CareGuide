"""Public commands exercise real local workers and independent-process recovery."""
import json
from pathlib import Path
import subprocess
import sys

import pytest


def invoke(*args):
    result = subprocess.run([sys.executable, 'scripts/reproduce_unified.py', *args],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30,
        encoding='utf-8')
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize('scenario,source,count', [('compose', 'compose', 2), ('template', 'template', 3)])
def test_documented_unified_command(scenario, source, count, tmp_path):
    result = invoke('--scenario', scenario, '--state-dir', str(tmp_path))
    assert result['status'] == 'completed'
    assert result['plan_source'] == source
    assert len(result['tasks']) == count
    assert all(t['attempts'] == 1 and t['worker_result'] for t in result['tasks'].values())
    assert result['budget']['used']['model'] == result['paid_calls'] == 0


def test_wait_command_restores_in_another_process_without_repeating_success(tmp_path):
    first = invoke('--scenario', 'wait', '--phase', 'start', '--state-dir', str(tmp_path))
    assert first['status'] == 'waiting_user'
    assert first['budget']['used']['tool'] == 1
    final = invoke('--scenario', 'wait', '--phase', 'resume', '--state-dir', str(tmp_path))
    assert final['status'] == 'completed'
    assert final['budget']['used']['tool'] == 2
    assert final['budget']['used']['model'] == 0
    assert final['tasks']['goal1']['attempts'] == final['tasks']['goal2']['attempts'] == 1
