"""Explicit V1 contracts. These are task plans, not clinical decisions."""
from dataclasses import dataclass
from enum import Enum


class IntentCategory(str, Enum):
    SYMPTOM_QUERY = 'symptom_query'
    MEDICATION_QUERY = 'medication_query'
    APPOINTMENT = 'appointment'
    EMERGENCY = 'emergency'


class UrgencyLevel(str, Enum):
    LOW = 'LOW'
    MEDIUM = 'MEDIUM'
    HIGH = 'HIGH'
    CRITICAL = 'CRITICAL'


class MedicalTask(str, Enum):
    SYMPTOM_INFORMATION = 'symptom_information'
    CARE_PREPARATION = 'care_preparation'
    DEPARTMENT_QUERY = 'department_query'
    DRUG_LABEL = 'drug_label'
    DRUG_DOSAGE_INFORMATION = 'drug_dosage_information'
    DRUG_CONTRAINDICATIONS = 'drug_contraindications'
    DRUG_PRECAUTIONS = 'drug_precautions'
    DRUG_INTERACTIONS = 'drug_interactions'
    APPOINTMENT_CHANNELS = 'appointment_channels'
    APPOINTMENT_PROCESS = 'appointment_process'
    APPOINTMENT_MATERIALS = 'appointment_materials'
    APPOINTMENT_CANCELLATION = 'appointment_cancellation'
    FOLLOWUP_GUIDANCE = 'followup_guidance'
    HANDOFF_REQUEST = 'handoff_request'
    HANDOFF_QUERY = 'handoff_query'
    DEPARTMENT_NAVIGATION = 'department_navigation'
    REPORT_PROCESS_GUIDANCE = 'report_process_guidance'
    HOSPITAL_INFORMATION = 'hospital_information'
    APPOINTMENT_POLICY = 'appointment_policy'
    DOCTOR_SEARCH = 'doctor_search'
    SLOT_SEARCH = 'slot_search'
    APPOINTMENT_CREATE = 'appointment_create'
    APPOINTMENT_RESCHEDULE = 'appointment_reschedule'
    APPOINTMENT_CANCEL = 'appointment_cancel'
    APPOINTMENT_STATUS = 'appointment_status'


class AgentType(str, Enum):
    TRIAGE = 'triage'
    MEDICATION = 'medication'
    APPOINTMENT = 'appointment'
    ESCALATION = 'escalation'


@dataclass(frozen=True)
class AgentContract:
    name: str
    mission: str
    tools: tuple[str, ...]
    forbidden: tuple[str, ...]


AGENTS = {
    AgentType.TRIAGE: AgentContract(
        'TriageAgent', '收集症状、必要追问、依据资料提供科室引导、整理就诊摘要',
        ('search_care_guidance', 'lookup_department'),
        ('诊断结论', '排除急症', '处方')),
    AgentType.MEDICATION: AgentContract(
        'MedicationAgent', '查询指定药品和规格的说明书信息；仅在可靠数据覆盖时查询相互作用',
        ('lookup_drug_label', 'lookup_drug_interaction'),
        ('个体化剂量决定', '调整用药', '未命中即认定无相互作用')),
    AgentType.APPOINTMENT: AgentContract(
        'AppointmentAgent', '查询医院资料，并在合成医院中查询号源和准备受控预约操作',
        ('lookup_hospital_guide', 'search_doctors', 'list_available_slots',
         'get_appointment', 'prepare_appointment', 'prepare_reschedule',
         'prepare_cancellation'),
        ('真实医院号源声明', '未经确认执行写操作', '真实预约成功声明')),
    AgentType.ESCALATION: AgentContract(
        'EscalationAgent', '处理当前急症风险提示，终止普通咨询并引导用户立即寻求当地急救或医护帮助',
        (),
        ('诊断结论', '声称已联系急救或医院', '用普通资料查询延迟紧急提示')),
}

ROUTES = {
    IntentCategory.SYMPTOM_QUERY: AgentType.TRIAGE,
    IntentCategory.MEDICATION_QUERY: AgentType.MEDICATION,
    IntentCategory.APPOINTMENT: AgentType.APPOINTMENT,
    IntentCategory.EMERGENCY: AgentType.ESCALATION,
}


def plan(intents: list[IntentCategory]) -> dict:
    """Accept explicitly identified intents; no claim of symptom detection accuracy."""
    if not intents:
        raise ValueError('At least one intent is required')
    if IntentCategory.EMERGENCY in intents:
        return {'status': 'urgent_handoff', 'agents': [AGENTS[AgentType.ESCALATION].name], 'tools': [],
                'requires_professional_help': True,
                'message': '优先寻求当地急救或医护帮助；本服务不能联系急救或替代急救处理。'}
    agents = list(dict.fromkeys(ROUTES[intent] for intent in intents))
    return {'status': 'planned_only', 'agents': [AGENTS[a].name for a in agents],
            'tools': list(dict.fromkeys(t for a in agents for t in AGENTS[a].tools)),
            'requires_professional_help': False}


def require_tool(agent: AgentType, tool: str) -> None:
    if tool not in AGENTS[agent].tools:
        raise PermissionError(f'{tool} is outside {AGENTS[agent].name} scope')


def capabilities() -> dict:
    from careguide.external_data import CAREGUIDE_INTENT_LABELS
    return {'name': 'CareGuide', 'stage': 'medical_chat_runtime',
            'natural_language_intent_enabled': True,
            'agent_tool_execution_enabled': True,
            'conversation_memory_enabled': True,
            'medical_answering_enabled': False,
            'sourced_information_queries_enabled': True,
            'hospital_integration_enabled': False,
            'synthetic_hospital_supported': True,
            'appointment_confirmation_required': True,
            'interaction_database_connected': False,
            'local_handoff_workflow_enabled': True,
            'knowledge_review_workflow_enabled': True,
            'development_evaluation_api_enabled': True,
            'fine_intents': [task.value for task in MedicalTask],
            'semantic_intent_taxonomy': 'CareGuide-13 (KUAKE-QIC-11 + appointment_action + emergency)',
            'semantic_intents': list(CAREGUIDE_INTENT_LABELS),
            'business_actions': ['health_information', 'department_navigation',
                'medication_information', 'drug_interaction', 'hospital_service_information',
                'doctor_search', 'slot_search', 'appointment_create', 'appointment_reschedule',
                'appointment_cancel', 'appointment_status', 'handoff_create', 'handoff_status'],
            'boundary_states': ['urgent_handoff', 'diagnosis_request',
                'individualized_medication', 'out_of_scope', 'needs_clarification'],
            'urgency_levels': [level.value for level in UrgencyLevel],
            'agents': {kind.value: {'name': spec.name, 'mission': spec.mission,
                                   'tool_scope': list(spec.tools),
                                   'forbidden': list(spec.forbidden)}
                       for kind, spec in AGENTS.items()}}
