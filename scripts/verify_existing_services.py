"""Verify existing local Docker services; optionally add isolated QA resources.

No service is started/stopped and no existing object or volume is deleted.
The write mode only touches identifiers generated for this one verification.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
import redis.asyncio as redis
import numpy as np

from careguide.action_store import ActionStore
from careguide.appointment_service import AppointmentService
from careguide.hospital_workflow import FHIRHospitalBackend


class FixtureEmbedding:
    """A deterministic test vector; never a retrieval-quality benchmark."""
    def __call__(self, input):
        return [np.asarray([(byte+1)/256 for byte in hashlib.sha256(text.encode()).digest()[:8]],
                           dtype=np.float32) for text in input]


async def run(args):
    report = {'kind': 'existing_docker_services_integration', 'paid_calls': 0,
              'mode': 'isolated_writes' if args.write_isolated else 'read_only', 'checks': {}}
    async with httpx.AsyncClient(timeout=10, trust_env=False) as http:
        for label, url in [('backend', 'http://127.0.0.1:8010/health'),
                ('frontend', 'http://127.0.0.1:5175/'), ('fhir', args.fhir_url+'/metadata'),
                ('chroma', 'http://127.0.0.1:8002/api/v1/heartbeat'),
                ('prometheus', 'http://127.0.0.1:19091/-/ready')]:
            started = time.monotonic()
            response = await http.get(url)
            response.raise_for_status()
            report['checks'][label] = {'http_status': response.status_code,
                                      'elapsed_ms': round((time.monotonic()-started)*1000, 3)}
    redis_client = redis.from_url(args.redis_url, decode_responses=True)
    report['checks']['redis'] = {'ping': bool(await redis_client.ping())}
    if not args.write_isolated:
        await redis_client.aclose()
        return report
    import chromadb
    from careguide.conversation_memory import MemoryManager
    from careguide.task_context import PersistentMemory
    from scripts.reproduce_orchestration import ScriptedProvider
    folder = Path(args.state_dir) if args.state_dir else Path(tempfile.mkdtemp(prefix='careguide-docker-qa-'))
    folder.mkdir(parents=True, exist_ok=True)
    nonce = uuid.uuid4().hex[:12]
    actor, conversation = 'cg-qa-patient-'+nonce, 'cg-qa-conversation-'+nonce
    practitioner, schedule = 'cg-qa-doctor-'+nonce, 'cg-qa-schedule-'+nonce
    slot_ids = ['cg-qa-slot-a-'+nonce, 'cg-qa-slot-b-'+nonce]
    resources = [dict(resourceType='Practitioner', id=practitioner, name=[{'text': 'CareGuide QA 合成医生'}]),
        dict(resourceType='Patient', id=actor, name=[{'text': 'CareGuide QA 合成患者'}]),
        dict(resourceType='Schedule', id=schedule, active=True, actor=[{'reference': 'Practitioner/'+practitioner}])]
    for index, identity in enumerate(slot_ids):
        resources.append(dict(resourceType='Slot', id=identity, status='free',
            schedule={'reference': 'Schedule/'+schedule},
            start=f'2030-01-02T{9+index:02}:00:00+08:00', end=f'2030-01-02T{9+index:02}:15:00+08:00'))
    requests = []
    async def record(request):
        requests.append({'method': request.method, 'resource': request.url.path.split('/')[-1]})
    async with httpx.AsyncClient(timeout=15, trust_env=False, event_hooks={'request': [record]}) as http:
        for resource in resources:
            before = await http.get(args.fhir_url+'/'+resource['resourceType']+'/'+resource['id'])
            assert before.status_code == 404, 'QA identity already exists; refusing to overwrite'
        response = await http.post(args.fhir_url, json={'resourceType': 'Bundle', 'type': 'transaction',
            'entry': [{'resource': resource, 'request': {'method': 'PUT',
                       'url': resource['resourceType']+'/'+resource['id']}} for resource in resources]})
        response.raise_for_status()
        assert response.json()['type'] == 'transaction-response'
    backend = FHIRHospitalBackend(args.fhir_url)
    actions = ActionStore(folder/'integration.sqlite3')
    service = AppointmentService(backend, actions)
    action = service.prepare('appointment_create', actor, conversation, {'slot_id': slot_ids[0]})
    claimed, ownership = actions.claim_confirmation(action['id'], actor, conversation)
    assert ownership
    completed = await backend.execute(claimed['type'], claimed['arguments'], claimed['idempotency_key'])
    assert completed['status'] == 'completed', completed.get('status')
    appointment_id = completed['after']['appointment']['id']
    # Lose the write acknowledgement before persisting its business outcome.
    # A new backend has no in-memory idempotency cache and must read FHIR proof.
    await backend.close()
    reopened = FHIRHospitalBackend(args.fhir_url)
    reconciliation_requests = []
    async def reconciliation_record(request):
        reconciliation_requests.append(request.method)
    reopened.client.event_hooks['request'] = [reconciliation_record]
    service = AppointmentService(reopened, ActionStore(folder/'integration.sqlite3'))
    recovered = await service.confirm(action['id'], actor, conversation)
    assert recovered['status'] == 'completed'
    assert recovered['result']['current_state_only']
    assert reconciliation_requests and set(reconciliation_requests) == {'GET'}
    count_before_replay = len(reconciliation_requests)
    assert (await service.confirm(action['id'], actor, conversation))['status'] == 'completed'
    assert len(reconciliation_requests) == count_before_replay
    changed = service.prepare('appointment_reschedule', actor, conversation,
                             {'appointment_id': appointment_id, 'slot_id': slot_ids[1]})
    assert (await service.confirm(changed['id'], actor, conversation))['status'] == 'completed'
    cancelled = service.prepare('appointment_cancel', actor, conversation, {'appointment_id': appointment_id})
    assert (await service.confirm(cancelled['id'], actor, conversation))['status'] == 'completed'
    appointments = await reopened.list_appointments(actor)
    assert len(appointments) == 1 and appointments[0]['status'] == 'cancelled'
    await reopened.close()
    report['checks']['fhir_write_recovery'] = {'created_fixture_resources': len(resources),
        'appointment_id': appointment_id, 'actor': actor, 'lost_response_recovered': True,
        'recovery_methods': reconciliation_requests[:count_before_replay],
        'repeat_confirm_network_requests': 0, 'reschedule': 'completed', 'cancel': 'completed',
        'final_appointments': len(appointments), 'existing_resources_overwritten': 0}
    report['state_dir'] = str(folder)
    report['retained_qa_resources'] = [resource['resourceType']+'/'+resource['id'] for resource in resources]
    (folder/'integration-progress.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    # This verification talks only to existing loopback services. Do not send
    # local Chroma traffic through a provider proxy inherited by the terminal.
    os.environ['NO_PROXY'] = ','.join(filter(None, [os.environ.get('NO_PROXY', ''), '127.0.0.1', 'localhost']))
    chroma = chromadb.HttpClient(host='127.0.0.1', port=8002,
                               settings=chromadb.Settings(anonymized_telemetry=False))
    provider = ScriptedProvider([])
    prefix = 'careguide_qa_'+nonce
    manager = MemoryManager(redis_client=redis_client, chroma_client=chroma,
        chroma_path=str(folder/'unused-chroma-fallback'), client=provider, model='offline-script',
        embedding_function=FixtureEmbedding(), collection_prefix=prefix,
        profile_path=str(folder/'integration.sqlite3'))
    projection = manager._profile
    class DiagnosedProjection:
        def upsert(self, **kwargs):
            try:
                return projection.upsert(**kwargs)
            except Exception:
                # All documents belong to this isolated synthetic QA run.
                traceback.print_exc()
                raise
        def get(self, **kwargs):
            return projection.get(**kwargs)
    manager._profile = DiagnosedProjection()
    memory = PersistentMemory(manager)
    async def update(text, value=None):
        await memory.add(actor, conversation, 'user', text)
        if value:
            source = (await manager._get_working_memory(actor, conversation))[-1]
            provider.replies.append({'candidates': [dict(key='preferred_time', operation='set', value=value,
                source_message_id=source.metadata['message_id'], source_quote=text)]})
        outcome = await memory.add(actor, conversation, 'assistant', '本次合成业务已完成。')
        await redis_client.expire(manager._wm_key(actor, conversation), 120)
        return outcome
    assert (await update('以后优先上午', 'morning'))['status'] == 'updated'
    assert (await update('这次下午'))['status'] == 'skipped'
    assert (await manager._get_profile(actor))['preferred_time']['value'] == 'morning'
    assert (await update('以后优先下午', 'afternoon'))['status'] == 'updated'
    stored = manager._profile.get(ids=[manager._profile_doc_id(actor)])
    assert json.loads(stored['documents'][0])['fields']['preferred_time']['value'] == 'afternoon'
    assert (await manager._get_profile('cg-qa-other-'+nonce)) == {}
    await redis_client.expire(manager._wm_key(actor, conversation), 120)
    report['checks']['persistent_memory'] = {'redis_ttl_seconds': await redis_client.ttl(
        manager._wm_key(actor, conversation)), 'chroma_collections': [prefix+'_episodic', prefix+'_user_profile'],
        'model_calls_scripted': len(provider.calls), 'projection_documents': len(stored['documents']),
        'temporary_not_persisted': True, 'explicit_correction': 'afternoon', 'user_isolation': True}
    await memory.close()
    report['state_dir'] = str(folder)
    report['retained_qa_resources'] = [resource['resourceType']+'/'+resource['id'] for resource in resources]
    report['retention'] = 'QA FHIR resources and two QA Chroma collections retained; own Redis key expires in 120s. No existing data deleted.'
    return report


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-isolated', action='store_true')
    parser.add_argument('--fhir-url', default='http://127.0.0.1:8080/fhir')
    parser.add_argument('--redis-url', default='redis://127.0.0.1:6380/0')
    parser.add_argument('--state-dir')
    print(json.dumps(asyncio.run(run(parser.parse_args())), ensure_ascii=False, indent=2))
