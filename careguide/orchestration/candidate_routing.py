"""Bounded read-only template admission, separate from candidate ranking.

Scores are engineering filters, not calibrated probabilities. A candidate must
also pass an action-specific slot/coverage contract. This deliberately supports
a finite query vocabulary; unknown words, relations and constraints abstain.
"""
from dataclasses import dataclass
import math
import re
import unicodedata

from careguide.external_data import CAREGUIDE_INTENT_LABELS
from careguide.orchestration.decisions import Goal
from careguide.orchestration.query_parser import DATE, SPEC, parse_time_scope
from careguide.orchestration.semantic_intent import load_bank


READ_ACTIONS = frozenset({'search_doctors', 'list_available_slots', 'lookup_drug_label'})
# Explicit legacy-label adapter; no claim that the old weights predict actions.
LABEL_ACTIONS = {
    'appointment_action': frozenset({'search_doctors', 'list_available_slots'}),
    '注意事项': frozenset({'lookup_drug_label'}),
}
COMMON = frozenset(('请 麻烦 劳驾 请你 想请你 我 想 想要 要 需要 能 能否 能不能 '
    '可以 可否 帮我 给我 推荐 推荐一下 一下 一份 一位 一个 一些 的 呢 吗 谢谢 '
    '有哪些 有什么 有 列 列出 展示 显示 提供 看 看看 查看 看一下 查询 查一下 查查 '
    '查 找 找找 找一下 找一找 了解 了解一下 告诉我').split())
WORDS = {
    'search_doctors': frozenset('医生 医师 大夫 坐诊 门诊 名单 列表 信息 哪些人 谁 在'.split()),
    'list_available_slots': frozenset(('号源 可约时间 可预约时间 预约时间 余号 空余 剩余 信息 列表 '
        '可预约 能约 还有 有没有 上午 下午 晚上').split()),
    'lookup_drug_label': frozenset(('说明书 禁忌 注意事项 不能用的情况 注意的地方 '
        '注意哪些事情 资料 信息 中 里 上 关于').split()),
}
BLOCKED = re.compile(r'不要|不想|不是|不查|不需要|不超过|别|只|仅|除了|排除|取消|改|'
    r'换|刚才|之前|上次|继续|这个|那个|这位|那位|这盒|它|上述|同时|另外|以及|'
    r'然后|并且|还要|也要|或者|还是|分别|和|与|及|再|最|优先|如果|否则')
SECTIONS = {'禁忌': 'contraindications', '不能用的情况': 'contraindications',
            '注意事项': 'precautions', '注意的地方': 'precautions', '注意哪些事情': 'precautions'}


@dataclass(frozen=True)
class RoutingPolicy:
    classifier_min: float = .90
    classifier_margin: float = .20
    semantic_min: float = .85
    semantic_margin: float = .10
    version: str = 'read-only-candidate-v1'


def classifier_family(candidates, policy):
    rows = [row for row in candidates if row.get('source') == 'classifier']
    if len(rows) != 1:
        return None
    row, labels = rows[0], set(CAREGUIDE_INTENT_LABELS)
    scores = row.get('scores', {})
    if (row.get('truncated') or set(scores) != labels or
            any(isinstance(v, bool) or not isinstance(v, (int, float)) or
                not math.isfinite(v) or not 0 <= v <= 1 for v in scores.values()) or
            not math.isclose(sum(scores.values()), 1., abs_tol=1e-4)):
        return None
    label = row.get('label')
    if label not in scores or label == '其他':
        return None
    others = [v for key, v in scores.items() if key != label]
    if scores[label] < policy.classifier_min or scores[label]-max(others) < policy.classifier_margin:
        return None
    return LABEL_ACTIONS.get(label, frozenset())


def semantic_action(candidates, bank, policy):
    """Require two agreeing references and separation from another action.

    Verify ids/text/action tags against the trusted local bank. Never consume
    arguments supplied by a candidate. Character neighbours cannot admit.
    """
    refs = {row['id']: row for row in bank['examples']}
    rows, seen = [], set()
    for row in candidates:
        if row.get('source') != 'semantic_embedding':
            continue
        ref, score = refs.get(row.get('example_id')), row.get('similarity')
        if (not ref or row['example_id'] in seen or row.get('bank_version') != bank['version'] or
                row.get('example') != ref['text'] or row.get('actions') != ref['actions'] or
                row.get('operation') != ref['operation'] or row.get('reference_only') is not True or
                isinstance(score, bool) or not isinstance(score, (int, float)) or
                not math.isfinite(score) or not -1 <= score <= 1):
            return None
        seen.add(row['example_id'])
        rows.append(row)
    rows.sort(key=lambda row: row['similarity'], reverse=True)
    if len(rows) < 3:
        return None
    first, second = rows[:2]
    if (first['operation'] != 'new' or second['operation'] != 'new' or
            len(first['actions']) != 1 or second['actions'] != first['actions'] or
            min(first['similarity'], second['similarity']) < policy.semantic_min):
        return None
    action = first['actions'][0]
    competitors = [row['similarity'] for row in rows[2:]
                   if row['actions'] != [action] or row['operation'] != 'new']
    if (action not in READ_ACTIONS or not competitors or
            second['similarity']-max(competitors) < policy.semantic_margin):
        return None
    return action


def covered(text, words):
    """Order-independent vocabulary coverage; not the strict query grammar."""
    reachable = {0}
    for index in range(len(text)):
        if index in reachable:
            reachable.update(index+len(word) for word in words if text.startswith(word, index))
    return len(text) in reachable


def arguments_for(text, action, local, at):
    """Extract only current-message slots, and abstain on unconsumed meaning."""
    if action not in READ_ACTIONS:
        return None, 'unsupported_action'
    body = unicodedata.normalize('NFKC', text)
    if not body.strip() or len(body) > 240:
        return None, 'unsupported_length'
    # Entity matches are removed before lexical checking, so medicine names
    # cannot accidentally act as operators or as general allowed vocabulary.
    args = {}
    if action in {'search_doctors', 'list_available_slots'}:
        matches = list(re.finditer(local._department, body))
        if len(matches) != 1:
            return None, 'explicit_single_department_required'
        args['department'] = matches[0].group().replace('心内科', '心血管内科')
        body = body[:matches[0].start()]+body[matches[0].end():]
        anchor = (r'医生|医师|大夫|坐诊' if action == 'search_doctors' else
                  r'号源|可约时间|可预约时间|预约时间|余号|可预约|能约')
        if not re.search(anchor, body):
            return None, 'action_anchor_missing'
        if action == 'list_available_slots':
            if len(re.findall(DATE, body)) > 1 or sum(word in body for word in ('上午', '下午', '晚上')) > 1:
                return None, 'multiple_time_scopes'
            scope = parse_time_scope(body, at)
            if scope is None:
                return None, 'invalid_date'
            args.update(scope)
            body = re.sub(DATE, '', body)
    else:
        names = [name for name in local.drugs if name in body]
        if len(names) != 1 or body.count(names[0]) != 1:
            return None, 'explicit_single_drug_required'
        entities = local.entities._explicit_entities(text, [])
        args = {key: value for key, value in entities.items()
                if key in {'drug_name', 'dosage_form', 'specification'}}
        body = body.replace(names[0], '', 1)
        specs = list(re.finditer(SPEC, body, re.I))
        if len(specs) > 1:
            return None, 'multiple_specifications'
        if specs:
            # Normalize for lexical checking only; tool arguments retain the
            # current source's original spelling via the shared extractor.
            if not args.get('specification'):
                return None, 'unresolved_specification'
            body = body[:specs[0].start()]+body[specs[0].end():]
        sections = {value for word, value in SECTIONS.items() if word in body}
        if len(sections) > 1 or not (sections or '说明书' in body):
            return None, 'unsupported_label_scope'
        args['section'] = next(iter(sections), 'all')
    if BLOCKED.search(body):
        return None, 'complex_or_restricted_expression'
    # Only punctuation between words is ignored; operators and unknown values
    # must be explicitly covered. The candidate does not supply missing slots.
    body = re.sub(r'[\s，,。！？?!、：:]', '', body)
    if not covered(body, COMMON | WORDS[action]):
        return None, 'unconsumed_text'
    return args, None


def admit(text, output, local, *, message_id, at, policy=None):
    policy = policy or RoutingPolicy()
    metadata = {'version': policy.version, 'status': 'rejected', 'calibrated': False,
                'thresholds': {'classifier_min': policy.classifier_min,
                    'classifier_margin': policy.classifier_margin,
                    'semantic_min': policy.semantic_min, 'semantic_margin': policy.semantic_margin}}
    if not message_id or not at:
        return None, {**metadata, 'reason': 'missing_current_source'}
    # Parsed partial goals cannot be discarded in favour of a single candidate.
    if output.goals or output.ambiguities:
        return None, {**metadata, 'reason': 'partial_or_ambiguous_request'}
    family = classifier_family(output.candidates, policy)
    semantic = (semantic_action(output.candidates, load_bank(), policy)
                if any(row.get('source') == 'semantic_embedding' for row in output.candidates) else None)
    if semantic is not None and family is not None and semantic not in family:
        return None, {**metadata, 'reason': 'candidate_conflict'}
    actions = [semantic] if semantic else sorted(family or ())
    accepted, rejections = [], {}
    for action in actions:
        arguments, reason = arguments_for(text, action, local, at)
        if arguments is not None:
            accepted.append((action, arguments))
        else:
            rejections[action] = reason
    if len(accepted) != 1:
        return None, {**metadata, 'reason': 'no_unique_supported_action', 'checks': rejections}
    action, arguments = accepted[0]
    missing = sorted(key for key in ('drug_name', 'dosage_form', 'specification') if not arguments.get(key)) if action == 'lookup_drug_label' else []
    goal = Goal(id='goal1', fragment=text, action=action, arguments=arguments,
                missing_fields=missing, sources={key: message_id for key in arguments})
    return goal, {**metadata, 'status': 'accepted', 'action': action,
                  'source': 'semantic_embedding' if semantic else 'classifier',
                  'label_adapter': 'careguide13-read-only-v1'}
