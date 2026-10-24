"""Bind only current results and explicit user information to task parameters."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import itertools
import re

from pydantic import ValidationError

from careguide.orchestration.contracts import ACTION_ARGUMENTS, TaskStatus
from careguide.tools import normalize_specification


CRITICAL_FIELDS = frozenset({'drug_name', 'other_drug', 'dosage_form', 'specification',
    'hospital_id', 'campus', 'department', 'practitioner_id', 'date', 'time_period', 'slot_id', 'appointment_id'})


class InvalidResultReference(ValueError):
    def __init__(self, source_task, reason):
        super().__init__(reason)
        self.source_task = source_task


@dataclass
class BoundInputs:
    arguments: dict = field(default_factory=dict)
    origins: dict = field(default_factory=dict)
    missing_fields: list[str] = field(default_factory=list)
    options: list[dict] = field(default_factory=list)
    fanout: dict[str, list] = field(default_factory=dict)

    def calls(self, *, max_calls=12):
        keys = list(self.fanout)
        count = 1
        for key in keys:
            count *= len(self.fanout[key])
        if count > max_calls:
            raise ValueError('Fan-out exceeds workflow dispatch limit')
        if not keys:
            return [dict(self.arguments)]
        return [{**self.arguments, **dict(zip(keys, values))}
                for values in itertools.product(*(self.fanout[key] for key in keys))]


def explicit_value(field, value, text, verified_entities=None, *, today=None):
    if value is None:
        return False
    verified_entities = verified_entities or {}
    if verified_entities.get(field) == value:
        return True
    if field == 'department' and value == '心血管内科' and '心内科' in text:
        return True
    if field == 'time_period':
        word = {'morning': '上午', 'afternoon': '下午', 'evening': '晚上'}.get(value)
        if word is None:
            return False
        negative = r'(?:不要|不用|不选|不喜欢|不在|避开|避免|除去).{0,4}'
        return (word in text or value in text) and not re.search(negative+f'(?:{word}|{value})', text)
    if isinstance(value, str) and value.casefold() in text.casefold():
        return True
    if field == 'specification' and isinstance(value, str):
        mentioned = re.findall(r'\d+(?:\.\d+)?\s*(?:mg|g|毫克|克)', text, re.I)
        return any(normalize_specification(item) == normalize_specification(value) for item in mentioned)
    if field == 'date':
        # Product dates use the documented Asia/Shanghai calendar, independent
        # of the host/container timezone. No unspecified date is invented.
        today = today or datetime.now(timezone(timedelta(hours=8))).date()
        for year, month, day in re.findall(r'(20\d{2})[-年/](\d{1,2})[-月/](\d{1,2})日?', text):
            try:
                if value == datetime(int(year), int(month), int(day)).date().isoformat():
                    return True
            except ValueError:
                continue
        return any(word in text and value == (today+timedelta(days=delta)).isoformat()
                   for word, delta in [('今天', 0), ('明天', 1), ('后天', 2)])
    return False


def resolve_inputs(record, workflow, *, verified_entities=None, now=None):
    now = now or datetime.now(timezone.utc)
    result = BoundInputs()
    schema = ACTION_ARGUMENTS[record.spec.action]
    fields = schema.model_fields
    verified_entities = {key: value for key, value in (verified_entities or {}).items() if key in fields}
    supplied = {**record.spec.arguments, **verified_entities}
    for key, value in record.arguments.items():
        if record.origins.get(key, {}).get('kind') in {'user_reply', 'user_selection', 'long_term_preference', 'retained_scope', 'user_change'}:
            supplied[key] = value
    for key, value in supplied.items():
        origin = record.origins.get(key, {})
        if key not in fields:
            raise ValueError('Unknown user answer field')
        if value is None:
            continue
        if (key in CRITICAL_FIELDS and origin.get('kind') not in {'user_reply', 'user_selection', 'long_term_preference', 'retained_scope', 'user_change'}
                and not explicit_value(key, value, record.spec.fragment, verified_entities)):
            result.missing_fields.append(key)
            result.origins[key] = {'kind': 'model_suggestion', 'confirmed': False}
        else:
            result.arguments[key] = value
            result.origins[key] = origin or {'kind': 'user_text', 'confirmed': True}
    for key, reference in record.spec.inputs.items():
        source = workflow.tasks[reference.task_id]
        if source.status != TaskStatus.SUCCEEDED or source.result is None:
            raise InvalidResultReference(source.spec.id, 'Upstream result is not complete')
        if reference.result_version is not None and reference.result_version != source.revision:
            raise InvalidResultReference(source.spec.id, 'Upstream result revision changed')
        if source.result_valid_until and source.result_valid_until <= now.isoformat():
            raise InvalidResultReference(source.spec.id, 'Upstream result expired')
        value = source.result
        for path in reference.path:
            if not isinstance(value, dict) or path not in value:
                raise InvalidResultReference(source.spec.id, 'Missing upstream result field')
            value = value[path]
        origin = {'kind': 'tool_result', 'task_id': source.spec.id,
                  'result_version': source.revision, 'confirmed': False}
        if reference.mode in {'selected', 'each'}:
            if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
                raise InvalidResultReference(source.spec.id, 'Expected candidate list')
            choices = [item[reference.item_field] for item in value if reference.item_field in item]
            if len(choices) != len(value) or not choices:
                raise InvalidResultReference(source.spec.id, 'Candidates have no usable identities')
            if reference.mode == 'each':
                result.fanout[key] = choices
                result.origins[key] = {**origin, 'mode': 'each', 'user_authorized': True}
                continue
            selected = result.arguments.get(key)
            if (result.origins.get(key, {}).get('kind') not in {'user_text', 'user_reply', 'user_selection'}
                    or selected not in choices):
                result.missing_fields.append(key)
                result.arguments.pop(key, None)
                result.options.append({'field': key, 'source_task': source.spec.id,
                    'source_version': source.revision, 'values': value})
                continue
            value = selected
            origin = {**origin, 'kind': 'user_selection', 'confirmed': True}
        elif isinstance(value, (list, dict)):
            raise InvalidResultReference(source.spec.id, 'A collection is not a confirmed scalar choice')
        result.arguments[key] = value
        result.origins[key] = origin
    required = {name for name, spec in fields.items() if spec.is_required()}
    required.update({'lookup_drug_label': {'drug_name', 'dosage_form', 'specification'},
                     'lookup_drug_interaction': {'drug_name', 'other_drug', 'dosage_form', 'specification'},
                     'prepare_appointment': {'slot_id'},
                     'prepare_reschedule': {'slot_id', 'appointment_id'},
                     'prepare_cancellation': {'appointment_id'}}.get(record.spec.action, set()))
    if record.spec.action == 'search_care_guidance':
        result.arguments['query'] = record.spec.fragment
    for key in required:
        if result.arguments.get(key) in (None, '') and key not in result.fanout:
            result.missing_fields.append(key)
    result.missing_fields = sorted(set(result.missing_fields))
    # Validate every concrete fan-out call, not a list disguised as a scalar.
    if not result.missing_fields:
        for arguments in result.calls():
            schema.model_validate(arguments)
    return result
