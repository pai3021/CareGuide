import pytest
from fastapi.testclient import TestClient

from careguide.api import app
from careguide.domain import AgentType, IntentCategory, plan, require_tool

client = TestClient(app)


def test_emergency_preempts_multi_domain_execution():
    result = plan([IntentCategory.MEDICATION_QUERY, IntentCategory.EMERGENCY,
                   IntentCategory.APPOINTMENT])
    assert result['status'] == 'urgent_handoff'
    assert result['agents'] == ['EscalationAgent']
    assert result['tools'] == []
    assert result['requires_professional_help']


def test_combined_consultation_deduplicates_agents():
    result = plan([IntentCategory.SYMPTOM_QUERY, IntentCategory.MEDICATION_QUERY,
                   IntentCategory.SYMPTOM_QUERY, IntentCategory.APPOINTMENT])
    assert result['agents'] == ['TriageAgent', 'MedicationAgent', 'AppointmentAgent']
    assert result['status'] == 'planned_only'


@pytest.mark.parametrize('agent,tool', [
    (AgentType.TRIAGE, 'lookup_drug_label'),
    (AgentType.MEDICATION, 'lookup_hospital_guide'),
    (AgentType.APPOINTMENT, 'create_ticket'),
    (AgentType.APPOINTMENT, 'submit_appointment'),
])
def test_cross_domain_and_write_tools_are_denied(agent, tool):
    with pytest.raises(PermissionError):
        require_tool(agent, tool)


def test_api_rejects_legacy_and_deferred_intents():
    for intent in ['incident', 'report_query', 'create_ticket']:
        assert client.post('/plan', json={'intents': [intent]}).status_code == 422
    assert client.post('/plan', json={'intents': []}).status_code == 422


def test_capabilities_do_not_claim_unimplemented_integrations():
    result = client.get('/capabilities').json()
    assert not result['medical_answering_enabled']
    assert not result['hospital_integration_enabled']
    assert not result['interaction_database_connected']
    assert result['intent_classifier_backend'] == 'legacy'
    health = client.get('/health')
    assert health.status_code == 200
    assert health.json()['integrations']['intent_classifier']['status'] == 'available'
    assert health.json()['integrations']['memory']['status'] == 'ok'
    assert health.json()['integrations']['memory']['backend'] == 'session'
    assert result['memory']['backend'] == 'session'
