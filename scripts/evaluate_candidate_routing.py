"""Local-only read-only routing smoke with the actual existing classifier.

Never creates an LLM or embedding client, runs tools, or downloads weights.
The selected development probes are not an accuracy/calibration benchmark.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from careguide.corpus import Corpus, DEFAULT_CORPUS
from careguide.orchestration.understanding import LocalClassifier, LocalUnderstanding, compose_plan


CASES = (
    ('canonical', '查消化内科医生', 'search_doctors'),
    ('doctor_list', '给我列一份消化内科坐诊大夫的名单', 'search_doctors'),
    ('doctor_order', '消化内科坐诊大夫的名单给我列一下', 'search_doctors'),
    ('doctor_alias', '推荐一下心内科有哪些坐诊大夫', 'search_doctors'),
    ('slots', '麻烦提供消化内科明天下午剩余号源的信息', 'list_available_slots'),
    ('label', '请展示精氨酸布洛芬片0.2g说明书里注意的地方', 'lookup_drug_label'),
    ('missing_spec', '关于精氨酸布洛芬片不能用的情况请给我展示一下', 'lookup_drug_label'),
    ('unknown_constraint', '给我列一份消化内科坐诊大夫的名单，预算200元', None),
    ('negation', '给我列一份消化内科坐诊大夫的名单，先不要预约', None),
    ('multiple', '给我列一份消化内科和心内科坐诊大夫名单', None),
    ('history', '给我列一份那个科室坐诊大夫名单', None),
    ('write', '给我列一份消化内科号源并准备预约', None),
)


async def evaluate(model_path, device='cpu'):
    classifier = LocalClassifier(model_path, device)
    local = LocalUnderstanding(Corpus(DEFAULT_CORPUS), classifier)
    # No remote retriever/client exists. Classifier uses local_files_only=True.
    rows = []
    at = datetime.now(timezone.utc).isoformat()
    for case_id, text, expected in CASES:
        output = await local.understand(text, message_id=case_id, at=at, classify=False)
        before, _ = compose_plan(text, output)
        await local.enrich(text, output, budget=None, scope=('local_smoke', case_id),
                           message_id=case_id, at=at, allow_remote=False)
        plan, metadata = compose_plan(text, output)
        prediction = next((r for r in output.candidates if r.get('source') == 'classifier'), {})
        actions = [task.action for task in plan.tasks] if plan else []
        rows.append({'id': case_id, 'text': text, 'expected_action_if_locally_accepted': expected,
            'rule_plan': before is not None, 'plan_source': metadata['plan_source'], 'actions': actions,
            'arguments': [task.arguments for task in plan.tasks] if plan else [],
            'classifier_label': prediction.get('label'), 'classifier_confidence': prediction.get('confidence'),
            'classifier_margin': prediction.get('margin'), 'truncated': prediction.get('truncated'),
            'candidate_admission': output.candidate_admission, 'diagnostics': output.diagnostics,
            'unexpected_local_action': bool(actions) and actions != [expected]})
    sources = ['careguide/orchestration/candidate_routing.py', 'careguide/orchestration/understanding.py',
        'careguide/orchestration/query_parser.py', 'careguide/orchestration/unified.py',
        'scripts/evaluate_candidate_routing.py']
    return {'protocol': 'native-classifier-candidate-routing-smoke-v1', 'at': at, 'device': device,
        'scope': 'Twelve selected synthetic development probes; actual existing local checkpoint; no remote candidates, Planner or tools.',
        'provider_requests': 0, 'tools_executed': 0, 'production_deployed': False,
        'calibration_or_representative_accuracy_evaluated': False,
        'summary': {'cases': len(rows), 'rule_plans': sum(row['rule_plan'] for row in rows),
            'candidate_plans': sum(row['plan_source'] == 'candidate_template' for row in rows),
            'unresolved': sum(not row['actions'] for row in rows),
            'unexpected_local_actions': sum(row['unexpected_local_action'] for row in rows)},
        'rows': rows, 'source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = asyncio.run(evaluate(args.model_path, args.device))
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps(report['summary'], ensure_ascii=False))
