"""Deterministic user replies from authoritative task outcomes and evidence."""
from __future__ import annotations

import json
import uuid

from careguide.orchestration.contracts import TaskStatus
from careguide.trace_store import redact_sensitive


LABELS = {
    TaskStatus.SUCCEEDED: '已完成', TaskStatus.FAILED: '未完成',
    TaskStatus.WAITING_INPUT: '待补充或选择', TaskStatus.WAITING_CONFIRMATION: '待确认',
    TaskStatus.BLOCKED: '依赖受阻', TaskStatus.CANCELLED: '已取消',
    TaskStatus.INVALIDATED: '已失效', TaskStatus.EXPIRED: '已过期',
    TaskStatus.RECONCILING: '结果待核对', TaskStatus.WAITING_DEPENDENCIES: '等待前序任务',
    TaskStatus.RUNNING: '执行中', TaskStatus.READY: '待执行',
}
FIELD_LABELS = {'drug_name': '药品全名', 'dosage_form': '剂型', 'specification': '规格',
    'time_period': '时段',
    'hospital_id': '医院', 'campus': '院区', 'department': '科室',
    'practitioner_id': '医生', 'date': '日期', 'slot_id': '号源', 'appointment_id': '预约编号'}


def workflow_status(state, *, extra_failures=()):
    tasks = [task for task in state.tasks.values() if task.status != TaskStatus.CANCELLED]
    if not tasks and any(event.get('event') == 'direct_answer' for event in state.events):
        return 'failed' if extra_failures else 'completed'
    if state.status == 'expired':
        return 'expired'
    if not tasks and state.status == 'waiting_user':
        return 'waiting_user'
    if tasks and all(task.status == TaskStatus.EXPIRED for task in tasks):
        return 'expired'
    if any(task.status == TaskStatus.RECONCILING for task in tasks):
        return 'reconciling'
    if any(task.status == TaskStatus.WAITING_CONFIRMATION for task in tasks):
        return 'pending_confirmation'
    if any(task.status == TaskStatus.WAITING_INPUT for task in tasks):
        return 'waiting_user'
    if tasks and all(task.status == TaskStatus.SUCCEEDED for task in tasks) and not extra_failures:
        return 'completed'
    if any(task.status in {TaskStatus.RUNNING, TaskStatus.READY, TaskStatus.WAITING_DEPENDENCIES}
           for task in tasks):
        return 'executing'
    if any(task.status == TaskStatus.SUCCEEDED or task.result and (
           task.result.get('evidence') or task.result.get('items')) for task in tasks):
        return 'partial_completed'
    if not tasks and state.tasks:
        return 'cancelled'
    failures = [task.failure for task in tasks if task.failure]+list(extra_failures)
    if failures and any(failure.get('code') == 'not_covered' for failure in failures) and all(
            failure.get('code') in {'not_covered', 'dependency_blocked'} for failure in failures):
        return 'not_covered'
    return 'failed'


def compose_workflow(state, budget, *, request_id=None, extra_failures=(), intent_signals=None):
    status = workflow_status(state, extra_failures=extra_failures)
    sections, evidence, seen, tasks, actions, tools = [], [], set(), [], [], []
    failures = list(extra_failures)
    for task in state.tasks.values():
        data = task.result or {}
        current_scope = [{'label': FIELD_LABELS[key], 'value': (
            {'morning': '上午', 'afternoon': '下午', 'evening': '晚上'}.get(value, value)
            if key == 'time_period' else value)} for key, value in task.arguments.items()
            if key in FIELD_LABELS and value is not None]
        lines = [f'{task.spec.fragment}：{LABELS[task.status]}。']
        if current_scope:
            lines.append('当前条件：'+'，'.join(item['label']+'：'+str(item['value']) for item in current_scope))
        if task.status not in {TaskStatus.CANCELLED, TaskStatus.INVALIDATED, TaskStatus.EXPIRED}:
            if task.failure:
                failures.append(task.failure)
                lines.extend([task.failure['message'], task.failure['next_step']])
            if task.status == TaskStatus.WAITING_INPUT:
                lines.append('请补充或选择：'+ '、'.join(FIELD_LABELS.get(key, key) for key in task.missing_fields))
            if data.get('simulation'):
                lines.append('以下为 CareGuide 合成医院信息。')
            for item in data.get('items', []):
                label = '，'.join(str(item[key]) for key in ('id', 'name', 'department', 'start', 'status')
                                 if item.get(key) is not None)
                if label:
                    lines.append(label)
            for item in data.get('evidence', []):
                content = item.get('content') or item.get('text') or item.get('summary') or ''
                citation = item.get('citation', {})
                url = item.get('source_url') or item.get('url') or (
                    citation.get('url') if isinstance(citation, dict) else '')
                if content:
                    lines.append('资料摘录：'+content)
                    if url:
                        lines.append('来源：'+url)
                    if isinstance(citation, dict) and citation.get('review_status') == 'open_dataset_unreviewed':
                        lines.append('资料层级：开放研究数据，未经临床审核。')
                signature = json.dumps(item, sort_keys=True, ensure_ascii=False)
                if signature not in seen:
                    evidence.append(item)
                    seen.add(signature)
            if task.status == TaskStatus.WAITING_CONFIRMATION:
                lines.append('请使用确认入口确认模拟操作；尚未执行预约写入。')
                actions.append(data)
            elif task.status == TaskStatus.RECONCILING and task.action_id:
                lines.append('操作结果尚未核实，请使用核对入口；不会重复提交。')
                actions.append(data)
        sections.append('\n'.join(lines))
        tasks.append({'id': task.spec.id, 'revision': task.revision, 'label': task.spec.fragment,
            'status': task.status.value, 'action': task.spec.action, 'depends_on': task.spec.depends_on,
            'result': data, 'worker_result': task.worker_result,
            'failure': task.failure, 'missing_fields': task.missing_fields,
            'options': task.options, 'action_id': task.action_id, 'expires_at': task.expires_at,
            'arguments': task.arguments, 'current_scope': current_scope,
            'result_observed_at': task.result_observed_at})
    for event in state.events:
        for call in event.get('calls', []):
            if call.get('success'):
                tools.append(call['tool_name'])
    for failure in extra_failures:
        sections.append(failure['message']+'\n'+failure['next_step'])
    signals = intent_signals or {}
    agents = list(dict.fromkeys(task.spec.agent.value for task in state.tasks.values()))
    return redact_sensitive({'request_id': request_id or uuid.uuid4().hex,
        'conv_id': state.conversation, 'response': '\n\n'.join(sections) or '本次任务未完成。',
        'workflow_status': status, 'workflow': {'id': state.id, 'version': state.version,
            'revision': state.revision, 'status': status, 'expires_at': state.expires_at,
            'tasks': tasks}, 'budget': budget.summary(), 'errors': failures,
        'evidence': evidence, 'knowledge_used': bool(evidence),
        'medical_answering_enabled': False, 'simulation': any(
            task.spec.action.startswith('prepare_') or task.spec.action in {
                'search_doctors', 'list_available_slots', 'get_appointment'} for task in state.tasks.values()),
        'action_requests': actions, 'appointment': [], 'tools_used': list(dict.fromkeys(tools)),
        'agent_types': agents, 'agent_type': agents[0] if agents else None,
        'primary_agent': agents[0] if agents else None, 'supporting_agents': agents[1:],
        'intent': signals.get('intent'), 'intents': signals.get('intents', []),
        'execution_mode': 'validated_task_plan', 'memory_status': 'available'})
