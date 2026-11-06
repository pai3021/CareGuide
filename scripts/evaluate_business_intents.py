"""Evaluation-only business-action replay with actual local classifier scores.

No remote clients, semantic fixtures, Planner or tools. Native predictions are
collected once for diagnostics and reused across routing arms; this measures
coverage and errors, not latency. Ground truth is used only by scoring.
"""
import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from careguide.corpus import Corpus, DEFAULT_CORPUS
from careguide.evaluation.business_intents import DATASET, load_cases
from careguide.orchestration.candidate_routing import LABEL_ACTIONS, RoutingPolicy, classifier_family
from careguide.orchestration.understanding import LocalClassifier, LocalUnderstanding, compose_plan


def score_local(case, plan):
    tasks = plan.tasks if plan else []
    actions = {task.action for task in tasks}
    argument_match = all(any(task.action == action and all(task.arguments.get(k) == v for k, v in args.items())
                             for task in tasks) for action, args in case.arguments.items())
    forbidden = sorted(actions & set(case.forbidden_actions))
    valid = (plan is not None and case.operation == 'new' and not case.forbid_local_plan and
             actions == set(case.actions) and argument_match and not forbidden)
    return {'local_plan': plan is not None, 'raw_annotation_match': valid,
            'raw_annotation_mismatch': plan is not None and not valid,
            'action_set_match': plan is not None and actions == set(case.actions),
            'unsafe_local_plan': plan is not None and (case.operation != 'new' or case.forbid_local_plan or bool(forbidden)),
            'arguments_match': argument_match, 'forbidden_actions': forbidden}


def checkpoint_hashes(path):
    names = ('config.json', 'model.safetensors', 'pytorch_model.bin', 'tokenizer.json',
             'tokenizer_config.json', 'special_tokens_map.json', 'vocab.txt')
    result = {}
    for name in names:
        file = Path(path)/name
        if file.is_file():
            with file.open('rb') as handle:
                result[name] = hashlib.file_digest(handle, 'sha256').hexdigest()
    return result


async def evaluate(classifier=None, *, dataset_path=DATASET):
    dataset, cases, digest = load_cases(dataset_path)
    # Only texts go into native inference. Labels are never passed to a source.
    predictions, errors = {}, {}
    if classifier:
        for case in cases:
            try:
                predictions[case.text] = await asyncio.to_thread(classifier.predict, case.text)
            except Exception as exc:
                errors[case.id] = type(exc).__name__
    corpus, rows, summaries = Corpus(DEFAULT_CORPUS), [], {}
    class Replay:
        def __init__(self):
            self.calls = 0

        def predict(self, text):
            self.calls += 1
            if text not in predictions:
                raise RuntimeError('Native prediction unavailable')
            return predictions[text]

    for variant in ('rules_only', 'rules_classifier') if classifier else ('rules_only',):
        replay = Replay()
        local = LocalUnderstanding(corpus, replay if variant == 'rules_classifier' else None)
        samples = []
        for case in cases:
            output = await local.understand(case.text, message_id=case.id, at=dataset['source_at'], classify=False)
            before, _ = compose_plan(case.text, output)
            await local.enrich(case.text, output, budget=None, scope=('business_eval', case.id),
                message_id=case.id, at=dataset['source_at'], allow_remote=False,
                allow_candidate_plan=not case.workflow_event)
            plan, metadata = compose_plan(case.text, output)
            if case.workflow_event:
                # Explicit workflow events are input, not inferred gold labels.
                plan = None
            row = {'variant': variant, 'id': case.id, 'group': case.group,
                'rule_plan': before is not None and not case.workflow_event,
                'plan_source': metadata['plan_source'] if plan else 'unresolved',
                'actions': [task.action for task in plan.tasks] if plan else [],
                'arguments': [task.arguments for task in plan.tasks] if plan else [],
                'candidate_admission': output.candidate_admission,
                'diagnostics': output.diagnostics, **score_local(case, plan)}
            samples.append(row)
        rows.extend(samples)
        eligible_ids = {case.id for case in cases if case.local_eligible}
        planned = sum(row['local_plan'] for row in samples)
        correct = sum(row['raw_annotation_match'] for row in samples)
        eligible_correct = sum(row['id'] in eligible_ids and row['raw_annotation_match'] for row in samples)
        summaries[variant] = {'status': 'completed', 'cases': len(samples),
            'local_plans': planned, 'raw_annotation_matches': correct,
            'raw_annotation_mismatches': planned-correct,
            'raw_annotation_match_rate': correct/planned if planned else None,
            'action_set_matches': sum(row['action_set_match'] for row in samples),
            'unsafe_local_plans': sum(row['unsafe_local_plan'] for row in samples),
            'single_read_eligible_cases': len(eligible_ids),
            'correct_eligible_local_plans': eligible_correct,
            'eligible_local_coverage': eligible_correct/len(eligible_ids) if eligible_ids else None,
            'unresolved_by_lightweight': len(samples)-planned,
            'candidate_plans': sum(row['candidate_admission'].get('status') == 'accepted' for row in samples),
            'forbidden_action_cases': sum(bool(row['forbidden_actions']) for row in samples),
            'classifier_replay_acquisitions': replay.calls}
    if not classifier:
        summaries['rules_classifier'] = {'status': 'not_run_classifier_unavailable'}
    for variant in ('rules_semantic', 'rules_classifier_semantic_on_demand', 'rules_classifier_semantic_parallel'):
        summaries[variant] = {'status': 'not_run_embedding_unavailable'}
    eligible = [case for case in cases if case.local_eligible]
    diagnostics = []
    for case in cases:
        prediction = predictions.get(case.text, {})
        label = prediction.get('label')
        family = LABEL_ACTIONS.get(label, frozenset())
        strong = classifier_family([{'source': 'classifier', **prediction}], RoutingPolicy()) if prediction else None
        diagnostics.append({'id': case.id, 'group': case.group, 'label': label,
            'confidence': prediction.get('confidence'), 'margin': prediction.get('margin'),
            'truncated': prediction.get('truncated'), 'scores': prediction.get('scores'),
            'mapped_action_family': sorted(family),
            'strong_mapped_family': sorted(strong) if strong is not None else None,
            'target_in_raw_family': bool(case.actions) and set(case.actions) <= family,
            'target_in_strong_family': bool(case.actions) and strong is not None and set(case.actions) <= strong,
            'error': errors.get(case.id)})
    eligible_diag = [row for row in diagnostics if row['id'] in {case.id for case in eligible}]
    baseline = {row['id'] for row in rows if row['variant'] == 'rules_only' and row['raw_annotation_match']}
    added = {row['id'] for row in rows if row['variant'] == 'rules_classifier' and row['raw_annotation_match']} - baseline
    sources = ['careguide/evaluation/business_intents.py', 'scripts/evaluate_business_intents.py',
        'careguide/orchestration/understanding.py', 'careguide/orchestration/query_parser.py',
        'careguide/orchestration/candidate_routing.py', 'careguide/orchestration/intent_examples.json']
    return {'protocol': 'business-intent-local-replay-v1', 'at': datetime.now(timezone.utc).isoformat(),
        'dataset_version': dataset['version'], 'dataset_sha256': digest, 'dataset_split': dataset['split'],
        'group_counts': dict(Counter(case.group for case in cases)),
        'operation_counts': dict(Counter(case.operation for case in cases)),
        'provider_requests': 0, 'planner_calls': 0, 'tools_executed': 0,
        'native_classifier_predictions': len(predictions), 'native_classifier_errors': errors,
        'summary': summaries, 'incremental_correct_case_ids': sorted(added),
        'classifier_diagnostic': {'eligible_cases': len(eligible),
            'raw_family_target_covered': sum(row['target_in_raw_family'] for row in eligible_diag) if classifier else None,
            'strong_family_target_covered': sum(row['target_in_strong_family'] for row in eligible_diag) if classifier else None,
            'top_label_counts': dict(Counter(row['label'] for row in diagnostics)) if classifier else {},
            'rows': diagnostics if classifier else []},
        'rows': rows, 'corpus_fingerprint': corpus.fingerprint,
        'source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},
        'limitations': ['Synthetic evaluation-only engineering set, not independently reviewed or real-user data.',
            'Gold actions are not used as classifier inputs or candidate predictions.',
            'Actual local predictions are replayed equally across arms; no latency or token savings are measured.',
            'Argument annotations are compared to raw TaskSpec.arguments. Later Worker preparation, dependency binding and tool execution are not scored; a raw mismatch is not an execution failure.',
            'Unresolved light-module cases can go to upstream direct/boundary handling or the Planner; no end-to-end Planner quality was measured.',
            'Raw/strong legacy action-family coverage is not fine-grained action accuracy or calibrated confidence.',
            'Dense retrieval and policy comparison were not run; no conclusion about semantic-only superiority or optimal scheduling is supported.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    classifier = LocalClassifier(args.model_path, args.device) if args.model_path else None
    result = asyncio.run(evaluate(classifier))
    result['checkpoint_sha256'] = checkpoint_hashes(args.model_path) if args.model_path else {}
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'summary': result['summary'], 'incremental_correct_case_ids': result['incremental_correct_case_ids'],
                      'classifier_target_coverage': {k:v for k,v in result['classifier_diagnostic'].items() if k != 'rows'}}, ensure_ascii=False))
