"""Rule-based understanding with optional candidates. Scores never grant authority."""
from __future__ import annotations

import asyncio
from collections import Counter
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
import re
import threading

from careguide.domain import AgentType
from careguide.external_data import CAREGUIDE_INTENT_LABELS
from careguide.intent_recognizer import IntentRecognizer
from careguide.orchestration.contracts import InputReference, PlanProposal, TaskSpec, validate_plan
from careguide.orchestration.decisions import Goal, IntentUnderstanding


class LocalClassifier:
    """Reuse the existing read-only RoBERTa adapter, with one lazy load per runtime."""
    def __init__(self, path, device='auto'):
        self.path, self.device = Path(path), device
        self._instance = None
        self._lock = threading.Lock()

    def predict(self, text):
        with self._lock:
            if self._instance is None:
                from careguide.transformer_intent_classifier import TransformerIntentClassifier
                self._instance = TransformerIntentClassifier(self.path, device=self.device)
            instance = self._instance
        return instance.predict(text)


EXAMPLES = (
    ('查消化内科医生', 'search_doctors'),
    ('查询医生可预约时间', 'list_available_slots'),
    ('查询药品说明书禁忌', 'lookup_drug_label'),
    ('查询科室服务范围', 'lookup_department'),
    ('就诊前需要记录哪些信息', 'search_care_guidance'),
    ('查询挂号流程和材料', 'lookup_hospital_guide'),
)


def local_candidates(text):
    def vector(value):
        return Counter(value[i:i+2] for i in range(max(0, len(value)-1)))
    a = vector(text)
    scored = []
    for example, action in EXAMPLES:
        b = vector(example)
        denominator = math.sqrt(sum(v*v for v in a.values())*sum(v*v for v in b.values()))
        score = sum(v*b.get(k, 0) for k, v in a.items())/denominator if denominator else 0
        scored.append({'action': action, 'example': example, 'similarity': score,
                       'source': 'local_char_ngram', 'calibrated': False})
    return sorted(scored, key=lambda row: row['similarity'], reverse=True)[:3]


class LocalUnderstanding:
    def __init__(self, corpus, classifier=None, *, classifier_timeout=5, semantic=None, semantic_mode='local_ngram'):
        self.entities = IntentRecognizer(embedding_mode='disabled')
        self.entities.corpus = corpus
        self.classifier = classifier
        self.classifier_timeout = classifier_timeout
        self.semantic, self.semantic_mode = semantic, semantic_mode
        self.drugs = sorted({r.drug_name for r in corpus.records if r.drug_name}, key=len, reverse=True)
        departments = {r.department for r in corpus.records if r.department}
        self.departments = departments | {'消化内科', '心血管内科', '心内科', '神经内科'}
        self._department = '(?:'+'|'.join(re.escape(v) for v in sorted(self.departments, key=len, reverse=True))+')'
        self._prefix = r'(?:(?:请|麻烦)?(?:帮我)?(?:查询|查一下|查查|查|看看))?'

    def _goal(self, fragment, identity, message_id, at):
        text = fragment.strip(' ，,。！？?！;；\n')
        match_text = re.sub(r'(?:，|,)?然后告诉我结果$', '', text)
        entities = self.entities._explicit_entities(text, [])
        department = entities.get('department')
        if department == '心内科':
            department = '心血管内科'
        action, args, relation = None, {}, 'single'
        constraints = []
        time_scope = r'(?:(?:今天|明天|后天|20\d{2}[-年/]\d{1,2}[-月/]\d{1,2}日?))?(?:上午|下午|晚上)?'
        doctor_flow = self._prefix+f'(?P<department>{self._department})医生(?:的|和)?'+time_scope+r'(?:的)?号源(?P<booking>并(?:帮我)?(?:准备)?预约)?(?P<negative>[，,]?先别预约)?'
        matched_flow = re.fullmatch(doctor_flow, match_text)
        if matched_flow:
            department = matched_flow['department'].replace('心内科', '心血管内科')
            action, args, relation = 'search_doctors', {'department': department}, 'doctor_slots'
            if matched_flow['booking']:
                constraints.append('prepare_appointment')
            if matched_flow['negative']:
                constraints.append('no_appointment')
        elif (matched := re.fullmatch(self._prefix+f'(?P<department>{self._department})?(?:医生|医师)', match_text)):
            department = matched['department']
            if department:
                department = department.replace('心内科', '心血管内科')
            action, args = 'search_doctors', ({'department': department} if department else {})
        elif (matched := re.fullmatch(self._prefix+f'(?P<department>{self._department})?'+time_scope+r'(?:的)?(?:号源|可预约时间)', match_text)):
            department = matched['department']
            if department:
                department = department.replace('心内科', '心血管内科')
            action, args = 'list_available_slots', ({'department': department} if department else {})
        elif re.fullmatch(r'(?:请)?帮我(?:准备)?预约(?:模拟)?号源SLOT-[A-Z0-9-]+', match_text, re.I):
            action, args = 'prepare_appointment', {'slot_id': re.search(r'SLOT-[A-Z0-9-]+', text, re.I)[0]}
        else:
            names = [name for name in self.drugs if name in text]
            if len(names) == 1:
                spec = r'(?:\d+(?:\.\d+)?\s*(?:ml|毫升)\s*[:：/]\s*\d+(?:\.\d+)?\s*(?:mg|g|毫克|克)|\d+(?:\.\d+)?\s*(?:mg|g|ml|μg|ug|毫克|克|毫升))'
                drug_pattern = self._prefix+re.escape(names[0])+f'(?:{spec})?'+r'(?:的)?(?:说明书|禁忌|注意事项|用法|用量)'
                if re.fullmatch(drug_pattern, match_text, re.I):
                    action = 'lookup_drug_label'
                    args = {k: v for k, v in entities.items() if k in {'drug_name', 'dosage_form', 'specification'}}
                    args['section'] = ('contraindications' if '禁忌' in text else
                                       'precautions' if '注意事项' in text else
                                       'dosage' if re.search(r'用法|用量', text) else 'all')
            if action is None and re.fullmatch(r'(?:查询|查)?(?:头痛就诊前需要记录哪些信息|头痛就诊前怎么记录|就诊前需要什么准备)', match_text):
                action, args = 'search_care_guidance', {'query': text, 'top_k': 1}
        if action is None:
            return None, 'unknown'
        if action == 'list_available_slots':
            # The slot worker anchors relative dates to this exact source message.
            for word, value in [('上午', 'morning'), ('下午', 'afternoon'), ('晚上', 'evening')]:
                if word in text:
                    args['time_period'] = value
            for word, delta in [('今天', 0), ('明天', 1), ('后天', 2)]:
                if word in text:
                    day = datetime.fromisoformat(at).astimezone(timezone(timedelta(hours=8))).date()
                    args['date'] = (day+timedelta(days=delta)).isoformat()
        required = {'lookup_drug_label': {'drug_name', 'dosage_form', 'specification'},
                    'prepare_appointment': {'slot_id'}}.get(action, set())
        return Goal(id=identity, fragment=text, action=action, arguments=args,
                    missing_fields=sorted(key for key in required if not args.get(key)),
                    constraints=constraints, sources={key: message_id for key in args}), relation

    async def understand(self, text, *, message_id, at, history=(), classify=True):
        from careguide.timing import phase
        with phase('understanding'):
            return await self._understand(text, message_id=message_id, at=at, history=history, classify=classify)

    async def _understand(self, text, *, message_id, at, history=(), classify=True):
        output = IntentUnderstanding(recognition_source=['rule'], request_relation='new')
        # Only each fully matched clause can enter a local combination. Unknown
        # clauses keep the whole message on the dynamic path before any dispatch.
        pieces = re.split(r'[；;\n]|[，,](?:同时|另外|以及)?|同时|另外|以及', text)
        if re.search(r'先别预约', text):
            pieces = [text]  # preserve negation scope for the fixed flow grammar
        relations = []
        for index, piece in enumerate(pieces):
            if not piece.strip():
                continue
            goal, relation = self._goal(piece, f'goal{index+1}', message_id, at)
            if goal:
                output.goals.append(goal)
                relations.append(relation)
            else:
                output.unresolved_spans.append(piece)
        if not output.unresolved_spans and output.goals:
            if len(output.goals) == 1:
                output.relation = relations[0]
            elif (len(output.goals) == 2 and set(g.action for g in output.goals) == {
                    'search_doctors', 'lookup_drug_label'} and all(r == 'single' for r in relations)):
                output.relation = 'independent'
        if output.relation == 'unknown':
            output.request_relation = 'unknown'
            output.candidates = local_candidates(text)
            output.recognition_source.append('semantic')
            if classify:
                await self._classify(text, output)
        return output

    async def _classify(self, text, output):
        if self.classifier is None:
            return
        try:
            async with asyncio.timeout(self.classifier_timeout):
                prediction = await asyncio.to_thread(self.classifier.predict, text)
            if (prediction.get('label') not in CAREGUIDE_INTENT_LABELS or not prediction.get('scores')
                    or any(key not in CAREGUIDE_INTENT_LABELS or not isinstance(value, (int, float))
                           or not math.isfinite(value) or not 0 <= value <= 1
                           for key, value in prediction['scores'].items())):
                raise ValueError('Invalid local classifier labels or scores')
            output.candidates.append({'source': 'classifier', **prediction, 'calibrated': False})
            output.recognition_source.append('classifier')
            if prediction.get('truncated'):
                output.diagnostics.append('classifier_input_truncated')
        except Exception as exc:
            output.diagnostics.append('classifier_unavailable:'+type(exc).__name__)

    async def enrich(self, text, output, *, budget, scope, allow_remote=True):
        from careguide.timing import phase
        with phase('intent_candidates'):
            return await self._enrich(text, output, budget=budget, scope=scope, allow_remote=allow_remote)

    async def _enrich(self, text, output, *, budget, scope, allow_remote=True):
        if output.relation != 'unknown':
            return output
        async def semantic():
            if self.semantic is None or not allow_remote:
                output.semantic_retrieval = {'status': 'no_planner' if not allow_remote else self.semantic_mode,
                    'remote_calls': 0, 'direct_acceptance': False}
                return
            rows, metadata = await self.semantic.retrieve(text, budget=budget, scope=scope)
            output.semantic_retrieval = metadata
            if rows:
                output.candidates = [row for row in output.candidates if row['source'] != 'local_char_ngram']+rows
                output.recognition_source.append('embedding')
            else:
                output.diagnostics.append('semantic_embedding:'+metadata['status'])
        # Both are candidate sources for the same request. Neither changes
        # goals, relation, authorization, or parameters, irrespective of score.
        await asyncio.gather(self._classify(text, output), semantic())
        return output

    async def close(self):
        if self.semantic is not None:
            await self.semantic.close()


TEMPLATE_VERSION = 'doctor-slots-v1'


def compose_plan(text, understood):
    if understood.unresolved_spans or understood.ambiguities or understood.relation == 'unknown':
        return None, {'plan_source': 'llm', 'reason': 'unresolved_meaning_or_relationship'}
    tasks = []
    for goal in understood.goals:
        agent = AgentType.MEDICATION if goal.action.startswith('lookup_drug') else (
            AgentType.TRIAGE if goal.action in {'search_care_guidance', 'lookup_department'} else AgentType.APPOINTMENT)
        tasks.append(TaskSpec(id=goal.id, fragment=goal.fragment, action=goal.action,
            agent=agent, arguments=goal.arguments,
            completion='prepared' if goal.action == 'prepare_appointment' else
                       'source_evidence' if agent != AgentType.APPOINTMENT else 'items'))
    if understood.relation == 'doctor_slots':
        doctor = tasks[0]
        goal = understood.goals[0]
        tasks.append(TaskSpec(id='slots', fragment=goal.fragment, action='list_available_slots',
            agent=AgentType.APPOINTMENT, depends_on=[doctor.id], inputs={
                'practitioner_id': InputReference(task_id=doctor.id, mode='selected', item_field='id')},
            completion='items'))
        if 'prepare_appointment' in goal.constraints and 'no_appointment' not in goal.constraints:
            tasks.append(TaskSpec(id='booking', fragment=goal.fragment, action='prepare_appointment',
                agent=AgentType.APPOINTMENT, depends_on=['slots'], inputs={
                    'slot_id': InputReference(task_id='slots', mode='selected', item_field='id')}, completion='prepared'))
    if not tasks:
        return None, {'plan_source': 'llm', 'reason': 'no_goals'}
    proposal = validate_plan(PlanProposal(goal=text, tasks=tasks), text)
    source = 'template' if understood.relation == 'doctor_slots' else 'compose' if len(tasks) > 1 else 'single'
    return proposal, {'plan_source': source, 'template_version': TEMPLATE_VERSION if source == 'template' else None,
                      'recognition_source': understood.recognition_source}
