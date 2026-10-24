"""Bounded offline routing comparison; not a real-model accuracy benchmark."""
from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path
import statistics
import sys
import tempfile
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--mode', choices=['baseline', 'current'], required=True)
parser.add_argument('--baseline-root')
parser.add_argument('--samples', type=int, default=10)


class Provider:
    def __init__(self, plan):
        self.plan, self.calls = plan, 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
    async def create(self, **kwargs):
        self.calls += 1
        return {'choices': [{'message': {'content': json.dumps(self.plan, ensure_ascii=False)}}]}


async def run(args):
    sys.path.insert(0, str(Path(args.baseline_root) if args.mode == 'baseline' else ROOT))
    from careguide.action_store import ActionStore
    from careguide.agent_orchestrator import AgentOrchestrator
    from careguide.appointment_service import AppointmentService
    from careguide.corpus import Corpus
    from careguide.domain import IntentCategory
    from careguide.hospital_workflow import InMemoryHospitalBackend
    from careguide.intent_recognizer import IntentResult
    from careguide.retrieval.knowledge_base import MedicalKnowledgeBase
    from careguide.tools import MedicalToolRegistry
    folder = Path(tempfile.mkdtemp(prefix='careguide-routing-benchmark-'))
    corpus = Corpus(ROOT/'data/medical_v1')
    doctor = dict(id='doctors', fragment='查消化内科医生', action='search_doctors',
        agent='appointment', arguments={'department': '消化内科'}, completion='items')
    drug = dict(id='drug', fragment='查询精氨酸布洛芬片0.2g说明书', action='lookup_drug_label',
        agent='medication', arguments={'drug_name': '精氨酸布洛芬片', 'dosage_form': '片剂', 'specification': '0.2g'},
        completion='source_evidence')
    second = dict(id='second', fragment='查询富马酸酮替芬鼻喷雾剂15ml:16.7mg说明书', action='lookup_drug_label',
        agent='medication', arguments={'drug_name': '富马酸酮替芬鼻喷雾剂', 'dosage_form': '鼻喷雾剂', 'specification': '15ml:16.7mg'},
        completion='source_evidence')
    dependency_fragment = '查消化内科所有医生的号源'
    parent = {**doctor, 'fragment': dependency_fragment}
    slots = dict(id='slots', fragment=dependency_fragment, action='list_available_slots', agent='appointment',
        depends_on=['doctors'], inputs={'practitioner_id': dict(task_id='doctors', path=['items'],
            mode='each', item_field='id')}, completion='items')
    cases = [dict(id='single_doctor', text=doctor['fragment'], tasks=[doctor], primary='doctor_search', entities=doctor['arguments']),
        dict(id='single_drug', text=drug['fragment'], tasks=[drug], primary='drug_label', entities=drug['arguments']),
        dict(id='doctor_and_drug', text=doctor['fragment']+'，同时'+drug['fragment'], tasks=[doctor, drug],
             primary='doctor_search', entities=doctor['arguments']),
        dict(id='two_drugs', text=drug['fragment']+'和'+second['fragment'], tasks=[drug, second],
             primary='drug_label', entities=drug['arguments']),
        dict(id='doctor_then_slots', text=dependency_fragment, tasks=[parent, slots], primary='doctor_search', entities=doctor['arguments'])]
    report = {'kind': 'offline_controlled_single_label_routing_comparison', 'mode': args.mode,
        'samples_each': args.samples, 'clock': 'time.perf_counter', 'startup_excluded': True,
        'model_responses': 'oracle fixture JSON through the actual Planner API',
        'classifier': 'fixed single label; not a classifier accuracy measurement',
        'role_workers': 'deterministic existing fallback in both versions',
        'paid_calls': 0, 'cases': []}
    for case in cases:
        class Recognizer:
            async def recognize(self, message, history):
                label = IntentCategory.APPOINTMENT if case['primary'] == 'doctor_search' else IntentCategory.MEDICATION_QUERY
                return IntentResult(label, .99, [label], 'route', entities=case['entities'],
                    fine_intent=case['primary'], business_action=case['primary'])
        path = folder/(case['id']+'.sqlite3')
        provider = Provider({'goal': case['text'], 'tasks': case['tasks']})
        engine = AgentOrchestrator(Recognizer(), MedicalToolRegistry(MedicalKnowledgeBase(corpus)),
            appointment_service=AppointmentService(InMemoryHospitalBackend(), ActionStore(path)))
        if args.mode == 'current':
            from careguide.orchestration.planner import ModelPlanner
            from careguide.orchestration.runtime import OrchestrationRuntime
            engine = OrchestrationRuntime(engine, path, planner=ModelPlanner(provider))
        elapsed, observations = [], []
        for sample in range(args.samples+1):
            model_before = provider.calls
            started = time.perf_counter()
            response = await engine.process(case['text'], 'benchmark-user', f"{case['id']}-{sample}")
            duration = (time.perf_counter()-started)*1000
            trace = engine.traces.get(response['request_id'])
            calls = trace['execution']['tool_calls']
            outputs = [call['output'] for call in calls]
            completed = []
            evidence_correct = True
            for task in case['tasks']:
                if task['action'] == 'lookup_drug_label':
                    found = any(any(item.get('scope', {}).get('drug_name') == task['arguments']['drug_name']
                                    for item in output.get('evidence', [])) for output in outputs)
                    # Every emitted drug passage must match one requested product.
                    expected_names = {task['arguments']['drug_name'] for task in case['tasks'] if task['action'] == 'lookup_drug_label'}
                    evidence_correct &= all(item.get('scope', {}).get('drug_name') in expected_names
                        for output in outputs for item in output.get('evidence', []))
                else:
                    found = any(call['tool_name'] == task['action'] and call['output'].get('items') for call in calls)
                if found:
                    completed.append(task['id'])
            observation = {'returned_tasks': completed, 'expected_tasks': [task['id'] for task in case['tasks']],
                'complete': len(completed) == len(case['tasks']), 'evidence_scope_correct': evidence_correct,
                'tool_calls': len(calls), 'planner_calls': provider.calls-model_before,
                'workflow_status': response['workflow_status']}
            if sample:
                elapsed.append(duration); observations.append(observation)
        assert all(item == observations[0] for item in observations), 'Non-deterministic semantic fixture result'
        ordered = sorted(elapsed)
        report['cases'].append({'id': case['id'], 'text': case['text'], **observations[0],
            'latency_ms': {'samples': [round(value, 3) for value in elapsed],
                'median': round(statistics.median(elapsed), 3),
                'p95_nearest_rank': round(ordered[max(0, math.ceil(.95*len(ordered))-1)], 3)}})
    report['summary'] = {'returned_tasks': sum(len(case['returned_tasks']) for case in report['cases']),
        'expected_tasks': sum(len(case['expected_tasks']) for case in report['cases']),
        'complete_cases': sum(case['complete'] for case in report['cases']), 'cases': len(cases)}
    if args.mode == 'current':
        from careguide.orchestration.runtime import needs_plan
        gate_cases = [
            ('查消化内科医生', False), ('查询精氨酸布洛芬片0.2g禁忌和注意事项', False),
            ('查消化内科医生并查药品说明书', True), ('精氨酸布洛芬片和富马酸酮替芬滴鼻液有什么禁忌', True),
            (dependency_fragment, True), ('查消化内科医生 查询精氨酸布洛芬片0.2g说明书', True)]
        report['gate_cases'] = [{'text': text, 'expected_planning': expected,
            'actual_planning': needs_plan(text, corpus.records)} for text, expected in gate_cases]
        assert all(case['expected_planning'] == case['actual_planning'] for case in report['gate_cases'])
    return report


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    arguments = parser.parse_args()
    if arguments.mode == 'baseline' and not arguments.baseline_root:
        parser.error('--baseline-root must contain the archived main careguide package')
    if not 1 <= arguments.samples <= 30:
        parser.error('--samples must be between 1 and 30')
    print(json.dumps(asyncio.run(run(arguments)), ensure_ascii=False, indent=2))
