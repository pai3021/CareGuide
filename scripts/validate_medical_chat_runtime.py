"""Real localhost HTTP /chat integration with an explicitly SCRIPTED provider.

No external API, held-out dataset or existing credentials are accessed.
"""
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
from fastapi import FastAPI

ROOT = Path(__file__).resolve().parents[1]
mock_provider = FastAPI()


@mock_provider.post('/v1/chat/completions')
def completion(payload: dict):
    messages = payload['messages']
    if not payload.get('tools'):
        content = messages[-1]['content']
        inputs = json.loads(content[content.index('{'):])
        message = inputs['message']
        history = inputs.get('history', [])
        previous = ' '.join(m['content'] for m in history if m['role'] == 'user')
        entities = {}
        intents, disposition = ['symptom_query'], 'route'
        if '挂号' in message or '本院' in message:
            intents = ['appointment']
            if '北京大学国际医院' in message+previous:
                entities['hospital_id'] = 'pkuih'
            if '本院' in message:
                entities['campus'] = '本院'
        if '布洛芬' in message:
            intents = ['medication_query']
            entities['drug_name'] = '精氨酸布洛芬片'
            if '0.2g' in message:
                entities.update(dosage_form='片剂', specification='0.2g', section='precautions')
        if '头痛' in message and '挂号' in message:
            intents = ['symptom_query', 'appointment']
        if '叫不醒' in message:
            intents, disposition = ['emergency'], 'urgent_handoff'
        if 'Python' in message:
            intents, disposition = [], 'out_of_scope'
        answer = {'content': json.dumps({'intent': intents[0] if intents else None, 'intents': intents,
            'confidence': 0.98, 'disposition': disposition, 'entities': entities,
            'requested_action': 'sourced_information', 'reasoning': 'SCRIPTED integration fixture'}, ensure_ascii=False)}
    elif any(m['role'] == 'tool' for m in messages):
        answer = {'content': 'SCRIPTED final model prose, deliberately not rendered as medical advice'}
    else:
        inputs = json.loads(messages[-1]['content'])
        names = [tool['function']['name'] for tool in payload['tools']]
        name = 'lookup_drug_label' if 'lookup_drug_label' in names else (
               'lookup_hospital_guide' if 'lookup_hospital_guide' in names else 'search_care_guidance')
        from careguide.tools import ARGUMENT_TYPES
        args = {k: v for k, v in inputs['entities'].items() if k in ARGUMENT_TYPES[name].model_fields}
        if 'query' in ARGUMENT_TYPES[name].model_fields:
            prior = ' '.join(m['content'] for m in messages[:-1] if m['role'] == 'user')
            args['query'] = (prior+' '+inputs['message']).strip()
        answer = {'content': None, 'tool_calls': [{'id': 'scripted-call', 'type': 'function',
            'function': {'name': name, 'arguments': json.dumps(args, ensure_ascii=False)}}]}
    return {'id': 'scripted', 'object': 'chat.completion', 'created': 0, 'model': 'scripted-fixture',
            'choices': [{'index': 0, 'message': {'role': 'assistant', **answer}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}


def free_port():
    with socket.socket() as connection:
        connection.bind(('127.0.0.1', 0))
        return connection.getsockname()[1]


def hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ('data/medical_v1', 'data/eval/medical_intent_v1')
            for p in (ROOT/folder).glob('*') if p.is_file()}


def main():
    before = hashes()
    provider_port, api_port, ui_port = free_port(), free_port(), free_port()
    environment = {k: v for k, v in os.environ.items() if not k.startswith('CAREGUIDE_')}
    environment.update(CAREGUIDE_LLM_API_KEY='scripted-local-validation',
        CAREGUIDE_LLM_BASE_URL=f'http://127.0.0.1:{provider_port}/v1', CAREGUIDE_LLM_MODEL='scripted-fixture',
        CAREGUIDE_MEMORY_MODE='session', CAREGUIDE_RETRIEVAL_MODE='bm25',
        CAREGUIDE_INTENT_EMBEDDING_MODE='disabled', CAREGUIDE_PERSIST_TRACES='false',
        VITE_CAREGUIDE_API_URL=f'http://127.0.0.1:{api_port}', NO_PROXY='127.0.0.1,localhost')
    processes, checks = [], []
    def start(command, cwd=ROOT):
        process = subprocess.Popen(command, cwd=cwd, env=environment,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        processes.append(process)
    def wait(client, path):
        deadline = time.monotonic()+25
        while time.monotonic() < deadline:
            if any(p.poll() is not None for p in processes):
                raise RuntimeError('Validation process exited unexpectedly')
            try:
                if client.get(path).status_code == 200:
                    return
            except httpx.TransportError:
                pass
            time.sleep(0.1)
        raise RuntimeError('Readiness timeout')
    try:
        start([sys.executable, '-m', 'uvicorn', 'scripts.validate_medical_chat_runtime:mock_provider',
               '--host', '127.0.0.1', '--port', str(provider_port)])
        with httpx.Client(base_url=f'http://127.0.0.1:{provider_port}', trust_env=False) as mock:
            wait(mock, '/openapi.json')
        start([sys.executable, '-m', 'uvicorn', 'careguide.api:app', '--host', '127.0.0.1', '--port', str(api_port)])
        start([shutil.which('node'), str(ROOT/'frontend/node_modules/vite/bin/vite.js'),
               '--host', '127.0.0.1', '--port', str(ui_port), '--strictPort'], cwd=ROOT/'frontend')
        with httpx.Client(base_url=f'http://127.0.0.1:{api_port}', trust_env=False, timeout=30) as client:
            wait(client, '/health')
            for name, message, expected, conversation in [
                ('triage_tool_loop', '头痛就诊前记录哪些信息', 'completed', 'triage'),
                ('medication_citation', '精氨酸布洛芬片0.2g说明书', 'completed', 'drug'),
                ('medication_clarification', '精氨酸布洛芬片有什么注意事项', 'needs_clarification', 'drug-missing'),
                ('hospital_clarification', '北京大学国际医院挂号流程', 'needs_clarification', 'hospital'),
                ('hospital_context_followup', '本院', 'completed', 'hospital'),
                ('multiagent', '头痛就诊准备和挂号材料', 'completed', 'combined'),
                ('urgent_preemption', '家人现在叫不醒，帮忙挂号', 'urgent_handoff', 'urgent'),
                ('out_of_scope', '写个Python脚本', 'out_of_scope', 'other')]:
                response = client.post('/chat', json={'message': message, 'user_id': 'synthetic-user', 'conv_id': conversation})
                assert response.status_code == 200, (name, response.text)
                result = response.json()
                assert result['workflow_status'] == expected, (name, result)
                trace = client.get('/trace/tool/'+result['request_id']).json()['trace']
                from careguide.trace_store import validate_trace_completeness
                assert validate_trace_completeness(trace)['complete'], name
                assert not trace['side_effects'], name
                if name == 'urgent_preemption':
                    assert not result['tools_used']
                checks.append({'name': name, 'passed': True})
            for path in ('/monitor', '/metrics', '/skills', '/knowledge/stats'):
                assert client.get(path).status_code == 200
                checks.append({'name': path, 'passed': True})
            assert client.post('/search', params={'query': '头痛记录'}).json()['results']
            checks.append({'name': 'source_search', 'passed': True})
        with httpx.Client(base_url=f'http://127.0.0.1:{ui_port}', trust_env=False, timeout=20) as frontend:
            wait(frontend, '/')
            assert 'CareGuide' in frontend.get('/').text
            assert frontend.get('/api/health').status_code == 200
            assert frontend.get('/api/skills').json()['count'] == 3
            checks.append({'name': 'vue_page_and_api_proxy', 'passed': True})
        assert hashes() == before
        report = {'checks': checks, 'passed': len(checks), 'source_and_dataset_bytes_unchanged': True,
            'provider': 'SCRIPTED localhost OpenAI-compatible fixture', 'external_provider_validated': False,
            'heldout_recognition_evaluation_run': False, 'clinical_validity_evaluated': False}
        output = ROOT/'data/eval/results/medical_chat_runtime_validation.json'
        output.write_bytes((json.dumps(report, ensure_ascii=False, indent=2)+'\n').encode())
        print(json.dumps(report, ensure_ascii=False))
    finally:
        for process in reversed(processes):
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


if __name__ == '__main__':
    main()
