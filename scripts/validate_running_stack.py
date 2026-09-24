"""Verify the deployed Vue/nginx/API stack using synthetic requests."""
import argparse
import json
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expected-mode', choices=['offline_fallback', 'provider_tool_loop'], default='offline_fallback')
    parser.add_argument('--output', type=Path, required=True, help='New report path; existing evidence is never overwritten')
    parser.add_argument('--runtime-label', default='real Docker Vue/nginx/API')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output report already exists')
    checks = []
    with httpx.Client(base_url='http://127.0.0.1:5175', trust_env=False, timeout=120) as client:
        page = client.get('/')
        assert page.status_code == 200 and 'CareGuide' in page.text
        checks.append('vue_static_page')
        assert client.get('/api/health').json()['stage'] == 'medical_chat_runtime'
        checks.append('nginx_api_proxy')
        assert client.get('/api/skills').json()['count'] == 3
        checks.append('medical_skills')
        for name, message, conv, expected in [
            ('drug_source_query', '精氨酸布洛芬片0.2g说明书', 'stack-drug', 'completed'),
            ('hospital_missing_campus', '北京大学国际医院挂号流程', 'stack-hospital', 'needs_clarification'),
            ('hospital_context_followup', '本院', 'stack-hospital', 'completed')]:
            reply = client.post('/api/chat', json={'message': message, 'user_id': 'synthetic-stack-validation', 'conv_id': conv})
            assert reply.status_code == 200, name
            result = reply.json()
            assert result['workflow_status'] == expected, (name, result)
            assert result['execution_mode'] == args.expected_mode
            traced = client.get('/api/trace/tool/'+result['request_id']).json()
            assert traced['found']
            if args.expected_mode == 'provider_tool_loop':
                scores = traced['trace']['task_understanding']['intent_result']['source_scores']
                assert not scores['llm']['failed'] and not scores['embedding']['failed']
                assert scores['embedding_mode'] == 'remote'
                assert all(not agent['error'] for wave in traced['trace']['execution']['execution_waves'] for agent in wave['agents'])
            checks.append(name)
        assert client.get('/api/metrics').status_code == 200
        checks.append('prometheus_metrics')
    output = args.output
    report = {'passed': len(checks), 'checks': checks, 'runtime': args.runtime_label,
              'execution_mode': args.expected_mode, 'recognition_accuracy_evaluated': False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='\n') as file:
        file.write(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
