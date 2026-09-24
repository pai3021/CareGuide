"""Explicit real LLM + embedding integration gate, separate from dataset evaluation.

Reads process variables or an explicitly selected private configuration file.
Never runs held-out evaluation, indexes collections, or promotes a baseline.
A successful HTTP fallback cannot pass.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ('CAREGUIDE_LLM_API_KEY', 'CAREGUIDE_LLM_BASE_URL', 'CAREGUIDE_LLM_MODEL',
            'CAREGUIDE_EMBEDDING_API_KEY', 'CAREGUIDE_EMBEDDING_BASE_URL', 'CAREGUIDE_EMBEDDING_MODEL')


def preflight(environment):
    missing = [name for name in REQUIRED if not environment.get(name, '').strip()
               or environment[name].strip().upper().startswith('YOUR_')]
    invalid = []
    for name in ('CAREGUIDE_LLM_BASE_URL', 'CAREGUIDE_EMBEDDING_BASE_URL'):
        if name not in missing:
            value = urlsplit(environment[name])
            if value.scheme not in {'https', 'http'} or not value.hostname or value.username or value.password or value.query:
                invalid.append(name)
    return {'ready': not missing and not invalid, 'missing_variables': missing,
            'invalid_endpoint_variables': invalid, 'credentials_displayed': False,
            'external_calls_made': False}


def check_result(reply, trace, expected_intent, expected_status):
    understanding = trace['task_understanding']['intent_result']
    scores = understanding['source_scores']
    errors = []
    if scores.get('llm', {}).get('failed', True):
        errors.append('llm_branch_failed_or_missing')
    if scores.get('embedding_mode') != 'remote' or scores.get('embedding', {}).get('failed', True):
        errors.append('remote_embedding_branch_failed_or_missing')
    if reply.get('execution_mode') != 'provider_tool_loop':
        errors.append('offline_execution_fallback')
    if reply.get('intent') != expected_intent:
        errors.append('unexpected_intent')
    if reply.get('workflow_status') != expected_status:
        errors.append('unexpected_workflow_status')
    agents = [agent for wave in trace['execution']['execution_waves'] for agent in wave['agents']]
    if not agents or any(agent['error'] for agent in agents):
        errors.append('agent_execution_failed_or_missing')
    calls = trace['execution']['tool_calls']
    if not calls or any(not call['success'] for call in calls):
        errors.append('tool_execution_failed_or_missing')
    if expected_status == 'completed' and not reply.get('evidence'):
        errors.append('missing_source_evidence')
    if trace['side_effects'] or reply.get('medical_answering_enabled'):
        errors.append('medical_scope_violation')
    return errors


def source_hashes():
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for directory in ('data/medical_v1', 'data/eval/medical_intent_v1')
            for path in (ROOT/directory).glob('*') if path.is_file()}


def run(output):
    from careguide.api import app
    from careguide.runtime import get_chat_runtime, get_registry
    before = source_hashes()
    # Independent disposable runtime. Source queries stay BM25; real embedding
    # is exercised by intent recognition without writing an index.
    os.environ.update(CAREGUIDE_INTENT_EMBEDDING_MODE='remote', CAREGUIDE_RETRIEVAL_MODE='bm25',
                      CAREGUIDE_MEMORY_MODE='session', CAREGUIDE_PERSIST_TRACES='false')
    get_chat_runtime.cache_clear()
    get_registry.cache_clear()
    checks = []
    with TestClient(app) as client:
        for name, message, conversation, intent, status in [
            ('triage_query', '头痛就诊前应记录哪些信息？', 'real-triage', 'symptom_query', 'completed'),
            ('drug_exact_spec', '查询精氨酸布洛芬片0.2g说明书中的注意事项。', 'real-drug', 'medication_query', 'completed'),
            ('drug_missing_spec', '查精氨酸布洛芬片说明书，但我还没确认规格。', 'real-drug-missing', 'medication_query', 'needs_clarification'),
            ('hospital_missing_campus', '查询北京大学国际医院的挂号指南，我还没说院区。', 'real-hospital', 'appointment', 'needs_clarification'),
            ('hospital_followup', '本院，继续查刚才的挂号指南。', 'real-hospital', 'appointment', 'completed')]:
            response = client.post('/chat', json={'message': message, 'user_id': 'synthetic-real-provider-validation',
                                                  'conv_id': conversation})
            if response.status_code != 200:
                errors = ['chat_http_failure']
                diagnostics = {'http_status': response.status_code}
            else:
                reply = response.json()
                trace = client.get('/trace/tool/'+reply['request_id']).json()['trace']
                errors = check_result(reply, trace, intent, status)
                understanding = trace['task_understanding']['intent_result']
                diagnostics = {'intent': reply['intent'], 'workflow_status': reply['workflow_status'],
                    'branches': understanding['source_scores'],
                    'entity_fields': sorted(understanding['entities']),
                    'tool_calls': [{'tool': call['tool_name'], 'success': call['success'],
                                   'output_status': call['output'].get('status'),
                                   'argument_fields': sorted(call['input'])}
                                  for call in trace['execution']['tool_calls']]}
            checks.append({'name': name, 'passed': not errors, 'errors': errors, 'diagnostics': diagnostics})
            print(json.dumps(checks[-1]), flush=True)
    unchanged = source_hashes() == before
    report = {'scope': 'five synthetic integration scenarios, not recognition accuracy evaluation',
              'external_calls_made': True, 'checks': checks,
              'passed': all(c['passed'] for c in checks) and unchanged,
              'source_and_dataset_bytes_unchanged': unchanged,
              'heldout_recognition_evaluation_run': False, 'clinical_validity_evaluated': False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='\n') as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write('\n')
    return 0 if report['passed'] else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--config-file', type=Path, help='Explicitly authorized private configuration file; never implicit')
    args = parser.parse_args()
    if args.config_file:
        from dotenv import dotenv_values
        for name, value in dotenv_values(args.config_file, interpolate=False).items():
            if name.startswith('CAREGUIDE_') and value is not None:
                os.environ[name] = value
    result = preflight(os.environ)
    print(json.dumps(result), flush=True)
    if args.preflight_only or not result['ready']:
        return 0 if result['ready'] else 2
    if args.output is None or args.output.exists():
        parser.error('--output requires a new report path')
    # No provider exception text is printed: it may contain request credentials.
    try:
        return run(args.output)
    except Exception as error:
        print(json.dumps({'passed': False, 'failure_type': type(error).__name__}), flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
