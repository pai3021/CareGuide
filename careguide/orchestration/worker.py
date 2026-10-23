"""Run one validated subtask using existing scoped roles and hospital service."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import json
import uuid
import re

from careguide.errors import ErrorCode, Failure
from careguide.intent_recognizer import IntentRecognizer
from careguide.orchestration.bindings import explicit_value, resolve_inputs
from careguide.orchestration.budget import BudgetExhausted, budgeted_call, use_budget
from careguide.orchestration.contracts import HOSPITAL_ACTIONS, TaskStatus
from careguide.task_context import TaskContext


@dataclass
class TaskOutcome:
    status: TaskStatus
    data: dict = field(default_factory=dict)
    failure: dict | None = None
    missing_fields: list[str] = field(default_factory=list)
    options: list[dict] = field(default_factory=list)
    action_id: str | None = None
    traces: list[dict] = field(default_factory=list)
    observed_at: str | None = None
    valid_until: str | None = None


class TaskWorker:
    def __init__(self, engine, *, max_concurrency=4):
        self.engine = engine
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.entities = IntentRecognizer(embedding_mode='disabled')
        self.entities.corpus = engine.registry.corpus

    def prepare(self, record, workflow):
        verified = self.entities._explicit_entities(record.spec.fragment, [])
        if record.spec.action == 'lookup_drug_label':
            sections = [value for pattern, value in [(r'禁忌', 'contraindications'),
                (r'注意事项', 'precautions'), (r'用法|用量', 'dosage')]
                if re.search(pattern, record.spec.fragment)]
            if sections:
                verified['section'] = sections[0] if len(sections) == 1 else 'all'
            elif record.origins.get('section', {}).get('kind') not in {'retained_scope', 'user_change'}:
                verified['section'] = 'all'
        if 'department' in record.spec.arguments and '心内科' in record.spec.fragment:
            verified['department'] = '心血管内科'
        if record.spec.action == 'list_available_slots':
            periods = [('上午', 'morning'), ('下午', 'afternoon'), ('晚上', 'evening')]
            mentioned = [value for word, value in periods if word in record.spec.fragment or value in record.spec.fragment]
            explicit_periods = [value for value in mentioned
                                if explicit_value('time_period', value, record.spec.fragment)]
            if len(explicit_periods) == 1:
                verified['time_period'] = explicit_periods[0]
            elif not mentioned and record.origins.get('time_period', {}).get('kind') not in {
                    'user_reply', 'user_selection', 'retained_scope', 'user_change'}:
                preference = getattr(workflow, 'preference_snapshot', {}).get('preferred_time', {})
                if (preference.get('value') in {'morning', 'afternoon', 'evening'}
                        and preference.get('expires_at', '') > self._now()):
                    record.arguments['time_period'] = preference['value']
                    record.origins['time_period'] = {'kind': 'long_term_preference', 'confirmed': True,
                        'source_message_id': preference['source_message_id'], 'version': preference.get('version'),
                        'expires_at': preference['expires_at']}
            relative = [('今天', 0), ('明天', 1), ('后天', 2)]
            dates = [delta for word, delta in relative if word in record.spec.fragment]
            if len(dates) == 1:
                source_at = next((source['at'] for source in reversed(getattr(workflow, 'source_messages', []))
                                  if record.spec.fragment in source['content']), getattr(workflow, 'created_at', self._now()))
                source_date = datetime.fromisoformat(source_at).astimezone(timezone(timedelta(hours=8))).date()
                verified['date'] = (source_date+timedelta(days=dates[0])).isoformat()
            elif not dates:
                literal_dates = re.findall(r'(20\d{2})[-年/](\d{1,2})[-月/](\d{1,2})日?', record.spec.fragment)
                if len(literal_dates) == 1:
                    try:
                        verified['date'] = datetime(*(int(part) for part in literal_dates[0])).date().isoformat()
                    except ValueError:
                        pass
        for key, value in verified.items():
            origin = record.origins.get(key, {})
            if key == 'department' and record.arguments.get(key) == '心内科' and value == '心血管内科':
                record.arguments[key] = value
                origin['normalized_from'] = '心内科'
            if origin.get('retained_from') and record.arguments.get(key) != value:
                record.arguments.pop(key, None)
                record.origins.pop(key, None)
        for key, origin in list(record.origins.items()):
            if origin.get('expires_at') and origin['expires_at'] <= self._now():
                record.arguments.pop(key, None)
                record.origins.pop(key, None)
        bound = resolve_inputs(record, workflow, verified_entities=verified)
        if record.spec.action == 'list_available_slots' and mentioned and not explicit_periods and (
                record.origins.get('time_period', {}).get('kind') not in {'user_reply', 'user_selection'}):
            bound.arguments.pop('time_period', None)
            bound.missing_fields = sorted(set(bound.missing_fields+['time_period']))
        record.arguments, record.origins = bound.arguments, bound.origins
        return bound

    async def run(self, record, workflow, budget):
        bound = self.prepare(record, workflow)
        if bound.missing_fields:
            return TaskOutcome(TaskStatus.WAITING_INPUT, missing_fields=bound.missing_fields,
                               options=bound.options)
        arguments = bound.calls()
        executions = await asyncio.gather(*(
            self._one(record, workflow, values, budget) for values in arguments))
        calls = [call for execution in executions for call in execution['tool_calls']]
        effective = [call for call in calls if not call.get('resolved')]
        outputs = [call['output'] for call in effective if call['success']]
        evidence, items, seen = [], [], set()
        for output in outputs:
            for item in output.get('evidence', []):
                signature = json.dumps(item, sort_keys=True, ensure_ascii=False)
                if signature not in seen:
                    seen.add(signature)
                    evidence.append(item)
            items.extend(output.get('items', []))
        data = {**(outputs[-1] if outputs else {}), 'evidence': evidence,
                'items': items, 'input_results': [
                    {'input': call['input'], 'output': call['output']} for call in effective]}
        statuses = [output.get('status') for output in outputs]
        pending = next((output for output in outputs if output.get('status') == 'pending_confirmation'), None)
        if pending:
            return TaskOutcome(TaskStatus.WAITING_CONFIRMATION, data=data,
                action_id=pending['id'], traces=calls, observed_at=self._now())
        uncertain = next((output for output in outputs if output.get('status') in {'executing', 'reconciling'}), None)
        if uncertain:
            return TaskOutcome(TaskStatus.RECONCILING, data=data, action_id=uncertain['id'], traces=calls)
        missing = sorted({field for output in outputs for field in output.get('missing_fields', [])})
        if 'needs_clarification' in statuses:
            return TaskOutcome(TaskStatus.WAITING_INPUT, data=data, missing_fields=missing, traces=calls)
        fulfilled = ((record.spec.completion == 'source_evidence' and bool(evidence)) or
                     (record.spec.completion == 'items' and bool(items)) or 'completed' in statuses)
        failures = [call['failure'] for call in effective if call.get('failure')]
        failures.extend(execution['failure'] for execution in executions if execution.get('failure'))
        # An optional post-tool model response cannot erase an already fulfilled
        # task. A failed fan-out branch still makes the all-candidates task partial.
        unsuccessful = any(not execution['tool_calls'] or any(
            call.get('failure') for call in execution['tool_calls'] if not call.get('resolved'))
            for execution in executions)
        if fulfilled and not unsuccessful:
            if failures:
                data['warnings'] = failures
            ttl = 60 if record.spec.action == 'list_available_slots' else 300
            return TaskOutcome(TaskStatus.SUCCEEDED, data=data, traces=calls,
                action_id=next((output['id'] for output in outputs if output.get('status') == 'completed'
                                and output.get('id')), None),
                observed_at=self._now(), valid_until=(datetime.now(timezone.utc)+timedelta(seconds=ttl)).isoformat())
        failure = failures[-1] if failures else Failure(ErrorCode.NOT_COVERED,
            'completion_check', record.spec.id).payload()
        return TaskOutcome(TaskStatus.FAILED, data=data, failure=failure, traces=calls)

    @staticmethod
    def _now():
        return datetime.now(timezone.utc).isoformat()

    async def _one(self, record, workflow, arguments, budget):
        async with self.semaphore:
            context = TaskContext(task_type=record.spec.action, requested_action='sourced_information',
                entities=dict(arguments), request_id=uuid.uuid4().hex, task_id=record.spec.id,
                planned_action=record.spec.action, actor_user_id=workflow.actor,
                conversation_id=workflow.conversation)
            if record.spec.action == 'lookup_department':
                context.fine_intent = 'department_query'
            if record.spec.action.startswith('prepare_'):
                context.workflow_binding = {'workflow_id': workflow.id, 'plan_version': workflow.version,
                    'task_id': record.spec.id, 'task_revision': record.revision, 'run_id': workflow.active_run}
            with use_budget(budget):
                if record.spec.action in HOSPITAL_ACTIONS:
                    try:
                        async with budgeted_call('tool', 10):
                            return await self.engine._run_hospital_workflow(
                                HOSPITAL_ACTIONS[record.spec.action][1], context)
                    except (BudgetExhausted, TimeoutError) as exc:
                        code = ErrorCode.BUDGET_EXHAUSTED if isinstance(exc, BudgetExhausted) else ErrorCode.TOOL_TIMEOUT
                        failure = Failure(code, 'hospital_execution', record.spec.id).payload()
                        return {'tool_calls': [], 'failure': failure, 'error': failure['message']}
                return await self.engine._run_agent(record.spec.agent, record.spec.fragment, [], context)
