"""Offline development-case replay of parser and candidate acquisition policies.

Classifier outputs and retrieval references are fixtures, not real predictions.
No provider, Planner, hospital or tool is called. Counts verify control flow;
they do not estimate accuracy, semantic benefit, production latency or token cost.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from careguide.corpus import Corpus, DEFAULT_CORPUS
from careguide.external_data import CAREGUIDE_INTENT_LABELS
from careguide.orchestration.understanding import LocalUnderstanding, compose_plan


ROOT = Path(__file__).resolve().parents[1]
AT = '2026-11-04T17:30:00+00:00'
# text, expected local action (None means escalate), fixture label, mass, truncation
CASES = (
    ('查消化内科医生', 'search_doctors', 'appointment_action', .9, False),
    ('想找一位看消化内科的医生', 'search_doctors', 'appointment_action', .9, False),
    ('帮我找一位心内科的大夫', 'search_doctors', 'appointment_action', .9, False),
    ('我想看看消化内科有哪些医生', 'search_doctors', 'appointment_action', .9, False),
    ('消化内科有哪些医生可以选择', 'search_doctors', 'appointment_action', .9, False),
    ('能帮我查一下消化内科的医生名单吗', 'search_doctors', 'appointment_action', .9, False),
    ('我想看看明天下午还有哪些号源', 'list_available_slots', 'appointment_action', .9, False),
    ('帮我看看消化内科后天上午有没有号源', 'list_available_slots', 'appointment_action', .9, False),
    ('我想查一下2026年11月10日下午的可约时间', 'list_available_slots', 'appointment_action', .9, False),
    ('我想看看还有没有能约的时间', 'list_available_slots', 'appointment_action', .9, False),
    ('我想看看精氨酸布洛芬片0.2g的说明书', 'lookup_drug_label', '注意事项', .9, False),
    ('我想了解一下精氨酸布洛芬片的注意事项', 'lookup_drug_label', '注意事项', .9, False),
    ('想找看胃病的大夫', None, 'appointment_action', .9, False),
    ('把刚才的医生换成下午的', None, 'appointment_action', .9, False),
    ('我想看看这盒药的说明书', None, '注意事项', .9, False),
    ('查消化内科医生，预算不要超过200元', None, 'appointment_action', .45, False),
    ('我想看看2026年2月30日的可约时间', None, 'appointment_action', .45, False),
    ('我想看看精氨酸布洛芬片0.2g说明书，用英文回答', None, '注意事项', .9, True),
    ('我想看看消化内科明天上午和后天下午的号源', None, '其他', .9, False),
    ('那个改一下', None, None, 0, False),
)


def fixture_prediction(label, mass, truncated):
    scores = {key: (1-mass)/(len(CAREGUIDE_INTENT_LABELS)-1) for key in CAREGUIDE_INTENT_LABELS}
    scores[label] = mass
    return {'label': label, 'scores': scores, 'truncated': truncated}


class FixtureRetriever:
    def __init__(self):
        self.attempts = 0

    async def retrieve(self, text, **kwargs):
        self.attempts += 1
        return [{'source': 'semantic_embedding', 'actions': ['prepare_appointment'],
                 'operation': 'new', 'similarity': 1., 'reference_only': True}], {
                     'status': 'fixture_reference', 'remote_calls': 0, 'direct_acceptance': False}


async def evaluate():
    corpus, summaries, rows = Corpus(DEFAULT_CORPUS), {}, []
    for policy in ('parallel', 'on_demand'):
        counts = dict(cases=len(CASES), local_plans=0, planner_required=0,
                      fixture_classifier_calls=0, fixture_retrieval_attempts=0,
                      unexpected_local_accepts=0, mismatched_expected_actions=0)
        for index, (text, expected, label, mass, truncated) in enumerate(CASES):
            classifier_calls = []
            def predict(text):
                classifier_calls.append(1)
                return fixture_prediction(label, mass, truncated)
            semantic = FixtureRetriever()
            local = LocalUnderstanding(corpus, SimpleNamespace(predict=predict) if label else None,
                                       semantic=semantic, semantic_policy=policy)
            output = await local.understand(text, message_id='m', at=AT, classify=False)
            await local.enrich(text, output, budget=None, scope=('fixture', str(index)))
            plan, _ = compose_plan(text, output)
            actual = plan.tasks[0].action if plan and len(plan.tasks) == 1 else None
            counts['local_plans'] += int(plan is not None)
            counts['planner_required'] += int(plan is None)
            counts['fixture_classifier_calls'] += len(classifier_calls)
            counts['fixture_retrieval_attempts'] += semantic.attempts
            counts['unexpected_local_accepts'] += int(expected is None and plan is not None)
            counts['mismatched_expected_actions'] += int(actual != expected)
            rows.append({'case': index+1, 'policy': policy, 'text': text, 'expected_action': expected,
                         'actual_action': actual, 'retrieval': output.semantic_retrieval})
        summaries[policy] = counts
    sources = ['careguide/orchestration/query_parser.py', 'careguide/orchestration/understanding.py',
               'careguide/orchestration/unified.py', 'scripts/evaluate_understanding_layers.py']
    return {'protocol': 'layered-understanding-control-v1', 'date': '2026-11-05',
            'scope': '20 synthetic development cases; controlled candidates; control-flow checks only',
            'provider_requests': 0, 'planner_calls': 0, 'tools_executed': 0,
            'semantic_accuracy_evaluated': False, 'latency_or_token_savings_evaluated': False,
            'summary': summaries, 'rows': rows,
            'source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = asyncio.run(evaluate())
    if args.output:
        with args.output.open('x', encoding='utf-8') as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report['summary'], ensure_ascii=False, indent=2))
