import asyncio
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from careguide.action_store import ActionStore
from careguide.agent_orchestrator import AgentOrchestrator
from careguide.appointment_service import AppointmentService
from careguide.corpus import Corpus
from careguide.hospital_workflow import InMemoryHospitalBackend
from careguide.intent_v3_dataset import (FINE_LABELS, load_v3_boundary_development_cases,
                                         load_v3_boundary_evaluation_cases,
                                         load_v3_development_cases, load_v3_evaluation_cases)
from careguide.intent_v3_recognizer import V3IntentRecognizer
from careguide.knowledge_backends import WeKnoraMCPBackend
from careguide.retrieval.knowledge_base import MedicalKnowledgeBase
from careguide.tools import MedicalToolRegistry
from careguide.api import app
from careguide.runtime import get_chat_runtime, get_registry


def test_v3_dataset_is_balanced_and_heldout_is_separate():
    dev, heldout = load_v3_development_cases(), load_v3_evaluation_cases()
    assert len(FINE_LABELS) == 21
    assert len(dev) == 630 and len(heldout) == 210
    assert {item.id for item in dev}.isdisjoint(item.id for item in heldout)
    boundary_dev, boundary_heldout = load_v3_boundary_development_cases(), load_v3_boundary_evaluation_cases()
    assert len(boundary_dev) == 80 and len(boundary_heldout) == 40
    assert {item.id for item in boundary_dev}.isdisjoint(item.id for item in boundary_heldout)


def test_v3_routes_synthetic_write_and_extracts_confirmed_ids():
    result = asyncio.run(V3IntentRecognizer(embedding_mode="local_ngram").recognize(
        "帮我预约模拟号源SLOT-GASTRO-001"))
    assert result.fine_intent == "appointment_create"
    assert result.requested_action == "booking_transaction"
    assert result.entities["slot_id"] == "SLOT-GASTRO-001"


def test_confirmation_gate_prevents_writes_and_is_idempotent(tmp_path):
    backend = InMemoryHospitalBackend()
    service = AppointmentService(backend, ActionStore(tmp_path/"actions.sqlite3"))
    action = service.prepare("appointment_create", "u001", "c001", {"slot_id": "SLOT-GASTRO-001"})
    assert action["status"] == "pending_confirmation"
    assert asyncio.run(backend.list_appointments("u001")) == []
    completed = asyncio.run(service.confirm(action["id"], "u001", "c001"))
    assert completed["status"] == "completed"
    assert len(asyncio.run(backend.list_appointments("u001"))) == 1
    prepared_again = service.prepare("appointment_create", "u001", "c001", {"slot_id": "SLOT-GASTRO-001"})
    assert prepared_again["id"] == action["id"] and prepared_again["status"] == "completed"
    replay = asyncio.run(service.confirm(action["id"], "u001", "c001"))
    assert replay["status"] == "completed"
    assert len(asyncio.run(backend.list_appointments("u001"))) == 1


def test_v3_boundary_states_are_separate_from_fine_intents():
    recognizer = V3IntentRecognizer(embedding_mode="local_ngram")
    examples = {"我突然胸痛而且呼吸困难": "urgent_handoff",
                "请直接判断我得了什么病": "diagnosis_request",
                "直接告诉我今天该吃几片药": "individualized_medication",
                "帮我分析一只股票明天会不会涨": "out_of_scope"}
    for message, expected in examples.items():
        result = asyncio.run(recognizer.recognize(message))
        assert result.disposition == expected
        assert result.fine_intent == ""
def test_action_is_bound_to_user_and_conversation(tmp_path):
    service = AppointmentService(InMemoryHospitalBackend(), ActionStore(tmp_path/"actions.sqlite3"))
    action = service.prepare("appointment_cancel", "u001", "c001", {"appointment_id": "APT-X"})
    with pytest.raises(LookupError):
        asyncio.run(service.confirm(action["id"], "u002", "c001"))
    with pytest.raises(LookupError):
        asyncio.run(service.confirm(action["id"], "u001", "c002"))


def test_changed_arguments_invalidate_prior_confirmation(tmp_path):
    service = AppointmentService(InMemoryHospitalBackend(), ActionStore(tmp_path/"actions.sqlite3"))
    first = service.prepare("appointment_create", "u001", "c001", {"slot_id":"SLOT-GASTRO-001"})
    second = service.prepare("appointment_create", "u001", "c001", {"slot_id":"SLOT-CARD-001"})
    assert first["id"] != second["id"]
    assert service.actions.get(first["id"], "u001")["status"] == "expired"
    with pytest.raises(ValueError, match="expired"):
        asyncio.run(service.confirm(first["id"], "u001", "c001"))


def test_chat_trace_records_prepared_action_without_executing(tmp_path):
    corpus = Corpus()
    registry = MedicalToolRegistry(MedicalKnowledgeBase(corpus))
    backend = InMemoryHospitalBackend()
    service = AppointmentService(backend, ActionStore(tmp_path/"actions.sqlite3"))
    engine = AgentOrchestrator(V3IntentRecognizer(embedding_mode="local_ngram"), registry,
                               appointment_service=service)
    response = asyncio.run(engine.process("帮我预约模拟号源SLOT-GASTRO-001", "u001", "c001"))
    assert response["workflow_status"] == "pending_confirmation"
    assert response["simulation"] is True
    assert response["action_requests"][0]["status"] == "pending_confirmation"
    assert asyncio.run(backend.list_appointments("u001")) == []
    trace = engine.traces.get(response["request_id"])
    assert trace["side_effects"][0]["executed"] is False


def test_weknora_mcp_outage_never_falls_back():
    backend = WeKnoraMCPBackend("http://weknora.invalid/mcp")
    async def fail(_request):
        raise httpx.ConnectError("offline")
    async def run():
        await backend.client.aclose()
        backend.client = httpx.AsyncClient(transport=httpx.MockTransport(fail))
        result = await backend.search("挂号材料")
        await backend.close()
        return result
    result = asyncio.run(run())
    assert result["status"] == "source_unavailable"
    assert result["evidence"] == []


def test_weknora_mcp_streamable_http_contract():
    seen = []
    def handle(request):
        payload = json.loads(request.content); seen.append(payload)
        method = payload["method"]
        if method == "initialize":
            return httpx.Response(200, headers={"mcp-session-id":"session-1"},
                                  json={"jsonrpc":"2.0", "id":payload["id"], "result":{}})
        if method == "notifications/initialized":
            assert request.headers["mcp-session-id"] == "session-1"
            return httpx.Response(202)
        if method == "tools/list":
            return httpx.Response(200, json={"jsonrpc":"2.0", "id":payload["id"],
                "result":{"tools":[{"name":"hybrid_search"}]}})
        assert method == "tools/call"
        assert payload["params"]["arguments"] == {
            "kb_id":"kb-careguide", "query":"挂号", "match_count":5}
        return httpx.Response(200, json={"jsonrpc":"2.0", "id":payload["id"],
            "result":{"structuredContent":{"results":[{"chunk_id":"c1", "content":"挂号资料",
                "document_name":"门诊指南", "source_url":"https://example.test/guide"}]}}})
    async def run():
        backend = WeKnoraMCPBackend("https://weknora.test/mcp", knowledge_base_id="kb-careguide")
        await backend.client.aclose()
        backend.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        result = await backend.search("挂号", filters={"campus":"本部"})
        await backend.close()
        return result
    result = asyncio.run(run())
    assert [item["method"] for item in seen] == ["initialize", "notifications/initialized", "tools/list", "tools/call"]
    assert result["status"] == "found" and result["evidence"][0]["record_id"] == "c1"


def test_v3_assets_are_explicitly_synthetic():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root/"data/synthetic_hospital/manifest.json").read_text(encoding="utf-8"))
    rag = json.loads((root/"data/eval/careguide_rag_v3/manifest.json").read_text(encoding="utf-8"))
    assert manifest["real_patient_data"] is False and manifest["seed"] == 9999
    assert rag["files"]["dev.jsonl"]["count"] == 240
    assert rag["files"]["heldout.jsonl"]["count"] == 120


def test_v3_http_confirmation_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setenv("CAREGUIDE_INTENT_TAXONOMY", "v3")
    monkeypatch.setenv("CAREGUIDE_HOSPITAL_BACKEND", "memory")
    monkeypatch.setenv("CAREGUIDE_KNOWLEDGE_BACKEND", "embedded")
    monkeypatch.setenv("CAREGUIDE_BUSINESS_DB", str(tmp_path/"api.sqlite3"))
    get_chat_runtime.cache_clear(); get_registry.cache_clear()
    try:
        with TestClient(app) as client:
            reply = client.post("/chat", json={"message":"帮我预约模拟号源SLOT-GASTRO-001",
                                                "user_id":"u001", "conv_id":"api-c1"})
            assert reply.status_code == 200
            payload = reply.json(); action = payload["action_requests"][0]
            assert payload["simulation"] is True and action["status"] == "pending_confirmation"
            assert payload["integrations"]["hospital"] == {
                "backend":"synthetic_memory", "status":"used", "simulation":True}
            assert client.get("/appointments", params={"user_id":"u001"}).json()["appointments"] == []
            confirmed = client.post(f"/actions/{action['id']}/confirm",
                json={"user_id":"u001", "conv_id":"api-c1"})
            assert confirmed.status_code == 200 and confirmed.json()["status"] == "completed"
            action_trace = client.get(f"/trace/action/{action['id']}", params={"user_id":"u001"})
            assert action_trace.status_code == 200
            assert [event["status"] for event in action_trace.json()["events"]] == [
                "pending_confirmation", "executing", "completed"]
            assert len(client.get("/appointments", params={"user_id":"u001"}).json()["appointments"]) == 1
    finally:
        get_chat_runtime.cache_clear(); get_registry.cache_clear()


def test_fhir_appointment_query_uses_participant_actor_parameter():
    from careguide.hospital_workflow import FHIRHospitalBackend

    seen = {}
    async def run():
        backend = FHIRHospitalBackend("https://fhir.test")
        async def fake_get(resource, params=None):
            seen.update(resource=resource, params=params)
            return {"resourceType": "Bundle", "entry": []}
        backend._get = fake_get
        try:
            assert await backend.list_appointments("u001") == []
        finally:
            await backend.close()
    asyncio.run(run())
    assert seen == {"resource": "Appointment", "params": {"actor": "Patient/u001"}}


def test_fhir_client_disables_reused_search_results():
    from careguide.hospital_workflow import FHIRHospitalBackend
    backend = FHIRHospitalBackend("https://fhir.test")
    try:
        assert backend.client.headers["Cache-Control"] == "no-store"
    finally:
        asyncio.run(backend.close())
