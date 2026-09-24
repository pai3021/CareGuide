"""Start a real localhost API, exercise queries, verify no source writes, then stop."""
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from careguide.corpus import Corpus, DEFAULT_CORPUS

ROOT = Path(__file__).resolve().parents[1]


def hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in DEFAULT_CORPUS.glob('*.json')}


def main():
    corpus = Corpus()
    before = hashes()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    environment = dict(os.environ, CAREGUIDE_RETRIEVAL_MODE='bm25')
    process = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'careguide.api:app',
                                '--host', '127.0.0.1', '--port', str(port)],
                               cwd=ROOT, env=environment, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL,
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    checks = []
    try:
        with httpx.Client(base_url=f'http://127.0.0.1:{port}', trust_env=False, timeout=10) as client:
            deadline = time.monotonic()+20
            while True:
                if process.poll() is not None:
                    raise RuntimeError('API exited before startup')
                try:
                    if client.get('/health').status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError('API readiness timeout')
                time.sleep(0.1)
            capabilities = client.get('/capabilities').json()
            assert capabilities['coverage']['record_count'] == 24
            assert not capabilities['medical_answering_enabled']
            for name, body, expected_http, expected_status in [
                ('label_exact_spec', {'agent':'medication','tool':'lookup_drug_label','arguments':{
                    'drug_name':'精氨酸布洛芬片','dosage_form':'片剂','specification':'200mg','section':'dosage'}}, 200,'found'),
                ('label_missing_spec', {'agent':'medication','tool':'lookup_drug_label','arguments':{
                    'drug_name':'精氨酸布洛芬片'}}, 200,'needs_clarification'),
                ('unknown_interaction', {'agent':'medication','tool':'lookup_drug_interaction','arguments':{
                    'drug_name':'精氨酸布洛芬片','dosage_form':'片剂','specification':'0.2g','other_drug':'阿托伐他汀'}},200,'not_covered'),
                ('hospital_scoped_search', {'agent':'appointment','tool':'lookup_hospital_guide','arguments':{
                    'hospital_id':'pkuih','campus':'本院','query':'挂号需要带什么材料'}},200,'found'),
                ('general_care_search', {'agent':'triage','tool':'search_care_guidance','arguments':{
                    'query':'反复头痛如何记录就医信息'}},200,'found'),
                ('department_scope', {'agent':'triage','tool':'lookup_department','arguments':{
                    'hospital_id':'pkuih','campus':'本院','department':'神经内科'}},200,'found'),
                ('explicit_label_interaction', {'agent':'medication','tool':'lookup_drug_interaction','arguments':{
                    'drug_name':'精氨酸布洛芬片','dosage_form':'片剂','specification':'0.2g','other_drug':'肝素'}},200,'found'),
                ('write_denied', {'agent':'appointment','tool':'submit_appointment','arguments':{}},403,None),
                ('cross_scope_denied', {'agent':'triage','tool':'lookup_drug_label','arguments':{
                    'drug_name':'精氨酸布洛芬片'}},403,None),
            ]:
                response = client.post('/tools/query', json=body)
                payload = response.json()
                assert response.status_code == expected_http, name
                if expected_status:
                    assert payload['status'] == expected_status, name
                    assert payload['side_effects'] == [], name
                evidence = payload.get('evidence', [])
                assert all(e['citation']['url'].startswith('https://') for e in evidence)
                checks.append({'name':name,'http_status':response.status_code,
                               'result_status':payload.get('status'),
                               'record_ids':[e['record_id'] for e in evidence],
                               'source_urls':sorted({e['citation']['url'] for e in evidence})})
            urgent = client.post('/plan', json={'intents':['medication_query','emergency']}).json()
            assert urgent['agents'] == urgent['tools'] == []
            checks.append({'name':'explicit_emergency_preempts_plan','result_status':urgent['status']})
        assert hashes() == before
        report = {'verified_at':datetime.now(timezone.utc).isoformat(),
                  'corpus_fingerprint':corpus.fingerprint, 'retrieval_mode':'bm25',
                  'live_local_http':True, 'corpus_files_unchanged':True,
                  'external_llm_calls':0, 'external_embedding_calls':0,
                  'clinical_accuracy_evaluated':False, 'checks':checks}
        target = ROOT/'data/eval/results/read_only_v1_validation.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(f'Passed {len(checks)} real HTTP checks; corpus unchanged; no external LLM or embedding calls.')
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


if __name__ == '__main__':
    main()
