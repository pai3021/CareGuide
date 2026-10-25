"""Offline unified composition/template/restart examples; no provider or model loads."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from careguide.orchestration.unified import UnifiedRuntime
from scripts.reproduce_orchestration import build


async def run(args):
    folder = Path(args.state_dir) if args.state_dir else Path(tempfile.mkdtemp(prefix='careguide-unified-'))
    folder.mkdir(parents=True, exist_ok=True)
    old, _ = build(folder, [])
    runtime = UnifiedRuntime(old.engine, old.store.path)
    actor, conversation = 'unified-example-user', 'unified-example-conversation'
    metadata = folder/'unified-example.json'
    if args.phase == 'resume':
        saved = json.loads(metadata.read_text(encoding='utf-8'))
        if saved['scenario'] != 'wait' or args.scenario != 'wait':
            raise ValueError('Only wait supports cross-process resume')
        result = await runtime.process('0.2g', actor, conversation, workflow_event=dict(
            kind='answer', workflow_id=saved['workflow_id'], plan_version=saved['version'],
            task_id='goal2', task_revision=1, values={'specification': '0.2g'}))
    else:
        if args.scenario == 'template':
            text = '查消化内科医生的号源并预约'
        else:
            text = '查消化内科医生，同时查询精氨酸布洛芬片'+('' if args.scenario == 'wait' else '0.2g')+'说明书'
        result = await runtime.process(text, actor, conversation)
        metadata.write_text(json.dumps({'scenario': args.scenario,
            'workflow_id': result['workflow']['id'], 'version': result['workflow']['version']}), encoding='utf-8')
        if args.phase == 'start' and args.scenario != 'wait':
            raise ValueError('Use start only with wait')
        if args.scenario == 'template':
            for task, field, value in [('slots', 'practitioner_id', 'PRA-GASTRO-01'),
                                       ('booking', 'slot_id', 'SLOT-GASTRO-001')]:
                result = await runtime.process(value, actor, conversation, workflow_event=dict(
                    kind='answer', workflow_id=result['workflow']['id'], plan_version=1,
                    task_id=task, task_revision=1, values={field: value}))
            action = result['action_requests'][0]
            assert (await runtime.confirm_action(action['id'], actor, conversation))['status'] == 'completed'
            assert (await runtime.confirm_action(action['id'], actor, conversation))['status'] == 'completed'
            result = await runtime.process('继续', actor, conversation, workflow_event=dict(
                kind='run', workflow_id=result['workflow']['id'], plan_version=1))
        elif args.scenario == 'wait' and args.phase == 'both':
            result = await runtime.process('0.2g', actor, conversation)
    trace = runtime.traces.get(result['request_id'])
    state = runtime.store.get(result['workflow']['id'], actor, conversation)
    return {'scenario': args.scenario, 'phase': args.phase, 'state_dir': str(folder),
        'evidence': 'offline_local_composition_real_workers', 'paid_calls': 0,
        'status': result['workflow_status'], 'budget': result['budget'],
        'plan_source': next((e.get('plan_source') for e in state.events if e['event'] == 'request_understood'), None),
        'tasks': {k: {'status': t.status.value, 'attempts': t.attempts,
                      'worker_result': bool(t.worker_result)} for k, t in state.tasks.items()},
        'trace_present': trace is not None}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', choices=['compose', 'template', 'wait'], default='compose')
    parser.add_argument('--phase', choices=['both', 'start', 'resume'], default='both')
    parser.add_argument('--state-dir')
    print(json.dumps(asyncio.run(run(parser.parse_args())), ensure_ascii=False, indent=2))
