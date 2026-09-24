"""Build the explicitly split 240/120 dataset once, without model calls."""
import hashlib
import json
from collections import Counter
from pathlib import Path

from careguide.domain import IntentCategory
from careguide.intent_dataset import DATASET_ROOT, IntentCase
from scripts.intent_seed_cases import DEV, HELDOUT

AGENT_NAMES = {'symptom_query':'TriageAgent', 'medication_query':'MedicationAgent',
               'appointment':'AppointmentAgent'}
REFERENCE_ROOT = 'https://www.nhs.uk/'
REFERENCES = {
    'breathing': REFERENCE_ROOT+'symptoms/shortness-of-breath/',
    'stroke': REFERENCE_ROOT+'conditions/stroke/symptoms/',
    'poisoning': REFERENCE_ROOT+'conditions/poisoning/',
    'headache': REFERENCE_ROOT+'symptoms/headaches/',
}
MULTI = {
    'triage-mixed': ['appointment','medication_query','appointment','medication_query','appointment'],
    'medication-mixed': ['appointment','appointment','symptom_query','appointment','symptom_query'],
    'appointment-mixed': ['symptom_query','medication_query','symptom_query','medication_query','symptom_query'],
    'oral-navigation': {4:'appointment'},
    'urinary-navigation': {4:'appointment'},
    'symptom-followup-context': {3:'appointment',4:'medication_query'},
    'unverified-combination': {3:'appointment',4:'symptom_query'},
    'medication-followup-context': {4:'appointment'},
    'arrival-and-checkin': {4:'symptom_query'},
    'identity-and-binding': {4:'medication_query'},
    'appointment-followup-context': {2:'symptom_query'},
    'headache-navigation': {4:'appointment'},
}


def parse_utterance(text):
    if not text.startswith('HISTORY:'):
        return text, []
    user, assistant, message = text.split('|')
    return message.removeprefix('USER:'), [
        {'role':'user','content':user.removeprefix('HISTORY:')},
        {'role':'assistant','content':assistant.removeprefix('ASSISTANT:')}]


def make_cases(groups, split):
    cases = []
    prefix = 'MI-D' if split == 'dev' else 'MI-H'
    for label, families in groups.items():
        for family, block in families.items():
            lines = block.splitlines()
            assert len(lines) == 5, family
            for index, raw_message in enumerate(lines):
                message, history = parse_utterance(raw_message)
                primary = label if label in {i.value for i in IntentCategory} else None
                intents = [primary] if primary else []
                extra = MULTI.get(family, {})
                secondary = extra[index] if isinstance(extra, list) else extra.get(index)
                if secondary:
                    intents.append(secondary)
                critical = label == 'emergency'
                disposition = 'urgent_handoff' if critical else 'route' if primary else label
                policy = 'urgent_handoff' if critical else (
                    'clarify_task' if label == 'needs_clarification' else
                    'out_of_scope' if label == 'out_of_scope' else
                    'redirect_to_professional' if family == 'individualized-request' else
                    'guidance_only_no_transaction' if family == 'live-transactions' else
                    'sourced_information_only')
                tags = [label, family]
                if history:
                    tags.append('multi_turn_context')
                if secondary:
                    tags.append('multi_intent')
                if family in {'negated-emergency', 'educational-risk', 'historical-risk-context',
                              'emergency-word-process', 'adverse-effect-reference'}:
                    tags.append('emergency_hard_negative')
                if critical:
                    tags.append('critical_priority')
                if family.startswith('emergency-overrides') or family == 'emergency-followup-context':
                    tags.append('emergency_preempts_other_task')
                if family in {'unsupported-products','unsupported-hospital-coverage','unverified-combination'}:
                    tags.append('intent_known_data_may_be_uncovered')
                references = []
                if critical:
                    combined = raw_message
                    if any(t in combined for t in ['吞','服','气体','清洁','清洗','毒']):
                        references.append(REFERENCES['poisoning'])
                    if any(t in combined for t in ['呼吸','喘','胸','发蓝','发青']):
                        references.append(REFERENCES['breathing'])
                    if any(t in combined for t in ['一侧','半边','嘴歪','脸下垂','说话','讲话','口齿']):
                        references.append(REFERENCES['stroke'])
                    if any(t in combined for t in ['头痛','抽搐','意识','叫不醒','无反应','不回应','没反应','失去']):
                        references.append(REFERENCES['headache'])
                    if not references:
                        references.append(REFERENCES['poisoning'])
                rationale = (
                    '描述当前或刚发生的明确风险情境，应用应优先引导紧急专业求助；不作诊断。'
                    if critical else
                    '缺少任务或指代内容，无法确定领域；先澄清，不强制分类。'
                    if label == 'needs_clarification' else
                    '没有V1支持的查询任务，或请求报告诊断等范围外服务；不指派领域Agent。'
                    if label == 'out_of_scope' else
                    f'当前主要请求属于{primary}，按任务目标分类；数据未覆盖或输出限制不改变请求所属领域。')
                if secondary:
                    rationale += f'同时明确请求{secondary}，保留辅助意图。'
                if history:
                    rationale += '需要结合给定对话历史解释当前消息。'
                case = {'id':f'{prefix}{len(cases)+1:03d}', 'split':split,
                        'scenario_family':family, 'message':message, 'history':history,
                        'primary_intent':primary, 'expected_intents':intents,
                        'disposition':disposition,
                        'expected_agents':[] if critical else [AGENT_NAMES[i] for i in intents],
                        'critical':critical, 'policy_action':policy, 'tags':tags,
                        'annotation_rationale':rationale,
                        'annotation_status':'developer_routing_annotation',
                        'clinical_reviewed':False,
                        'provenance':'hand_authored_synthetic_user_utterance',
                        'emergency_reference_urls':sorted(set(references))}
                IntentCase.model_validate(case)
                cases.append(case)
    return cases


def main():
    if DATASET_ROOT.exists():
        raise SystemExit('Dataset directory already exists; refusing overwrite or post-result relabeling.')
    splits = {'dev':make_cases(DEV,'dev'), 'heldout':make_cases(HELDOUT,'heldout')}
    assert len(splits['dev']) == 240 and len(splits['heldout']) == 120
    DATASET_ROOT.mkdir(parents=True)
    files = {}
    for split, cases in splits.items():
        raw = ''.join(json.dumps(case,ensure_ascii=False,separators=(',',':'))+'\n' for case in cases).encode('utf-8')
        name = split+'.jsonl'
        (DATASET_ROOT/name).write_bytes(raw)
        files[name] = {'sha256':hashlib.sha256(raw).hexdigest(), 'count':len(cases),
                       'primary_intent_counts':dict(Counter(c['primary_intent'] or '__boundary__' for c in cases)),
                       'disposition_counts':dict(Counter(c['disposition'] for c in cases)),
                       'multi_intent_count':sum(len(c['expected_intents'])>1 for c in cases),
                       'multi_turn_count':sum(bool(c['history']) for c in cases)}
    manifest = {'version':'careguide-medical-intent-v1', 'created_at':'2026-09-18',
                'labels':[intent.value for intent in IntentCategory], 'files':files,
                'total':360, 'synthetic':True, 'clinical_reviewed':False,
                'recognizer_implemented':False, 'evaluation_run':False,
                'split_policy':'Preassigned hand-authored scenario groups; no random paraphrase splitting',
                'heldout_policy':'Only explicit evaluator access; never prompts, embedding templates, rules or tuning',
                'limitations':['Constructed by the same developer; not independent clinical annotation or real patient distribution',
                               'Developer reviewed routing labels; emergency labels require professional review',
                               'Class balancing is an application challenge set, not estimated traffic prevalence']}
    (DATASET_ROOT/'manifest.json').write_bytes((json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    print('Built 360 synthetic routing cases: dev 240, heldout 120; no recognition evaluation run.')


if __name__ == '__main__':
    main()
