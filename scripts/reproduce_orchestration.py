"""Offline, provider-facing examples. No API keys, network calls or data resets.

Run with the existing project Python, from the repository root. `start` and
`resume` are intentionally separate OS processes over one SQLite checkpoint.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from careguide.action_store import ActionStore
from careguide.agent_orchestrator import AgentOrchestrator
from careguide.appointment_service import AppointmentService
from careguide.corpus import Corpus, DEFAULT_CORPUS
from careguide.domain import IntentCategory
from careguide.hospital_workflow import InMemoryHospitalBackend
from careguide.intent_recognizer import IntentResult
from careguide.orchestration.planner import ModelPlanner
from careguide.orchestration.runtime import OrchestrationRuntime
from careguide.retrieval.knowledge_base import MedicalKnowledgeBase
from careguide.tools import MedicalToolRegistry
from careguide.trace_store import validate_trace_completeness


class ScriptedProvider:
    """The same chat-completions interface used by the real ModelPlanner."""
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.replies:
            raise AssertionError('Unexpected extra model call in offline example')
        return {'choices': [{'message': {'content': json.dumps(self.replies.pop(0), ensure_ascii=False)}}]}


class SingleLabel:
    async def recognize(self, message, history):
        return IntentResult(IntentCategory.APPOINTMENT, .99, [IntentCategory.APPOINTMENT],
            'route', fine_intent='doctor_search', business_action='doctor_search')


def task_plan(*, waiting=False, dependency=False, partial=False):
    drug_text = '查询精氨酸布洛芬片'+('' if waiting else '0.2g')+'说明书'
    hospital_text = '查消化内科医生的号源并预约' if dependency else '查消化内科医生'
    text = hospital_text+'，同时'+drug_text
    args = {'drug_name': '精氨酸布洛芬片', 'dosage_form': '片剂'}
    if not waiting:
        args['specification'] = '0.2g'
    tasks = [dict(id='doctors', fragment=hospital_text, action='search_doctors',
        agent='appointment', arguments={'department': '消化内科'}, completion='items'),
        dict(id='drug', fragment=drug_text, action='lookup_drug_label', agent='medication',
             arguments=args, completion='source_evidence')]
    if dependency:
        tasks.extend([
            dict(id='slots', fragment=hospital_text, action='list_available_slots', agent='appointment',
                depends_on=['doctors'], inputs={'practitioner_id': dict(task_id='doctors',
                    path=['items'], mode='selected', item_field='id')}, completion='items'),
            dict(id='booking', fragment=hospital_text, action='prepare_appointment', agent='appointment',
                depends_on=['slots'], inputs={'slot_id': dict(task_id='slots', path=['items'],
                    mode='selected', item_field='id')}, completion='prepared')])
    if partial:
        text = text.replace('查消化内科医生', '查消化内科号源')
        tasks[0].update(id='slots', fragment='查消化内科号源', action='list_available_slots')
    return text, {'goal': text, 'tasks': tasks}


def build(folder, replies, *, backend=None):
    path = folder/'runtime.sqlite3'
    provider = ScriptedProvider(replies)
    engine = AgentOrchestrator(SingleLabel(), MedicalToolRegistry(
        MedicalKnowledgeBase(Corpus(DEFAULT_CORPUS))),
        appointment_service=AppointmentService(backend or InMemoryHospitalBackend(), ActionStore(path)))
    return OrchestrationRuntime(engine, path, planner=ModelPlanner(provider)), provider


def summarize(runtime, response):
    workflow = response['workflow']
    state = runtime.store.get(workflow['id'], 'example-user', 'example-conversation')
    return {'status': response['workflow_status'], 'workflow_id': state.id,
        'plan_version': state.version, 'used': response['budget']['used'],
        'provider_usage': response['budget']['tokens'],
        'tasks': {key: {'status': task.status.value, 'revision': task.revision, 'attempts': task.attempts,
            'arguments': task.arguments, 'missing_fields': task.missing_fields,
            'evidence_ids': [item.get('record_id') for item in (task.result or {}).get('evidence', [])]}
            for key, task in state.tasks.items()},
        'errors': [failure['code'] for failure in response['errors']],
        'trace_complete': validate_trace_completeness(runtime.traces.get(response['request_id']))['complete']}


async def run(args):
    folder = Path(args.state_dir) if args.state_dir else Path(tempfile.mkdtemp(prefix='careguide-example-'))
    folder.mkdir(parents=True, exist_ok=True)
    metadata = folder/'example.json'
    actor, conversation = 'example-user', 'example-conversation'
    reports = []
    if args.scenario == 'memory':
        from careguide.conversation_memory import MemoryManager, Message, MsgRole
        from careguide.profile_memory import ProfileStore
        instance = MemoryManager.__new__(MemoryManager)
        instance._profile_store = ProfileStore(folder/'runtime.sqlite3')
        documents = {}
        def upsert(**kwargs):
            documents[kwargs['ids'][0]] = kwargs['documents'][0]
        instance._profile = SimpleNamespace(upsert=upsert)
        source = [Message(MsgRole.USER, '以后优先上午', metadata={'message_id': uuid.uuid4().hex})]
        async def messages(*values):
            return source
        async def legacy(*values):
            return {}
        instance._get_working_memory, instance._get_legacy_profile = messages, legacy
        def extraction(value, quote):
            return {'candidates': [dict(key='preferred_time', operation='set', value=value,
                source_message_id=source[0].metadata['message_id'], source_quote=quote)]}
        provider = ScriptedProvider([extraction('morning', source[0].content)])
        instance._client, instance._model = provider, 'offline-script'
        for text, value in [('以后优先上午', 'morning'), ('这次下午', None), ('以后优先下午', 'afternoon')]:
            if reports:
                source[:] = [Message(MsgRole.USER, text, metadata={'message_id': uuid.uuid4().hex})]
                if value:
                    provider.replies.append(extraction(value, text))
            outcome = await instance.update_profile(actor, conversation)
            reports.append({'input': text, 'update': outcome, 'profile': await instance._get_profile(actor)})
        return dict(scenario=args.scenario, pid=os.getpid(), evidence='offline_scripted_model_and_projection',
                    paid_calls=0, model_calls=len(provider.calls), state_dir=str(folder), rounds=reports)
    if args.phase == 'resume':
        saved = json.loads(metadata.read_text(encoding='utf-8'))
        if saved['scenario'] != 'wait' or args.scenario != 'wait':
            raise ValueError('Only the wait scenario supports a separate resume phase')
        runtime, provider = build(folder, [])
        state = runtime.store.latest(actor, conversation)
        event = dict(kind='answer', workflow_id=saved['workflow_id'], plan_version=state.version,
                     task_id='drug', task_revision=state.tasks['drug'].revision,
                     values={'specification': '0.2g'}, event_id='offline-restart-answer')
        response = await runtime.process('0.2g', actor, conversation, workflow_event=event)
        reports.append(summarize(runtime, response))
    else:
        if metadata.exists():
            raise ValueError('State directory already contains an example; use resume or a fresh directory')
        text, plan = task_plan(waiting=args.scenario in {'wait', 'replan'},
                               dependency=args.scenario == 'dependency', partial=args.scenario == 'partial')
        replies = [plan]
        if args.scenario == 'replan':
            changed = {**plan['tasks'][0], 'fragment': '改查心血管内科医生',
                       'arguments': {'department': '心血管内科'}}
            replies.append(dict(base_version=1, reason='用户修改科室', update=[changed]))
        if args.scenario == 'partial':
            replies.append(dict(base_version=1, reason='重复计划应停止', update=[plan['tasks'][0]]))
        class FailingSlots(InMemoryHospitalBackend):
            async def list_available_slots(self, **kwargs):
                raise OSError('controlled example service outage')
        runtime, provider = build(folder, replies,
            backend=FailingSlots() if args.scenario == 'partial' else None)
        response = await runtime.process(text, actor, conversation)
        reports.append(summarize(runtime, response))
        identity = response['workflow']['id']
        metadata.write_text(json.dumps(dict(scenario=args.scenario, workflow_id=identity)), encoding='utf-8')
        if args.phase == 'start':
            if args.scenario != 'wait':
                raise ValueError('Use --phase start only with --scenario wait')
        elif args.scenario in {'wait', 'replan'}:
            if args.scenario == 'replan':
                response = await runtime.process('改查心血管内科医生', actor, conversation,
                    workflow_event=dict(kind='change', workflow_id=identity, plan_version=1))
                reports.append(summarize(runtime, response))
            state = runtime.store.get(identity, actor, conversation)
            response = await runtime.process('0.2g', actor, conversation, workflow_event=dict(
                kind='answer', workflow_id=identity, plan_version=state.version,
                task_id='drug', task_revision=1, values={'specification': '0.2g'}))
            reports.append(summarize(runtime, response))
        elif args.scenario == 'dependency':
            for task_id, field, value in [('slots', 'practitioner_id', 'PRA-GASTRO-01'),
                                          ('booking', 'slot_id', 'SLOT-GASTRO-001')]:
                response = await runtime.process(value, actor, conversation, workflow_event=dict(
                    kind='answer', workflow_id=identity, plan_version=1, task_id=task_id,
                    task_revision=1, values={field: value}))
                reports.append(summarize(runtime, response))
            action = response['action_requests'][0]
            confirmed = await runtime.confirm_action(action['id'], actor, conversation)
            again = await runtime.confirm_action(action['id'], actor, conversation)
            assert confirmed['status'] == again['status'] == 'completed'
            response = await runtime.process('继续', actor, conversation, workflow_event=dict(
                kind='run', workflow_id=identity, plan_version=1))
            reports.append(summarize(runtime, response))
            assert len(runtime.appointment_service.backend.appointments) == 1
    return dict(scenario=args.scenario, phase=args.phase, pid=os.getpid(), evidence='offline_scripted_planner_real_workers',
                paid_calls=0, model_calls_this_process=len(provider.calls), state_dir=str(folder), rounds=reports)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', choices=['dependency', 'wait', 'replan', 'partial', 'memory'], required=True)
    parser.add_argument('--phase', choices=['full', 'start', 'resume'], default='full')
    parser.add_argument('--state-dir')
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args)), ensure_ascii=False, indent=2))
