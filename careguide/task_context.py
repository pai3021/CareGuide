"""Shared medical task state for router-driven Agent execution."""
from dataclasses import dataclass, field, asdict
from enum import Enum

from careguide.intent_recognizer import IntentResult


class WorkflowStatus(str, Enum):
    RECEIVED = 'received'
    UNDERSTOOD = 'understood'
    EXECUTING = 'executing'
    READY = 'ready'
    COMPOSING = 'composing'
    ESCALATED = 'escalated'
    COMPLETED = 'completed'
    PARTIAL_COMPLETED = 'partial_completed'
    RECONCILING = 'reconciling'
    NEEDS_CLARIFICATION = 'needs_clarification'
    OUT_OF_SCOPE = 'out_of_scope'
    URGENT_HANDOFF = 'urgent_handoff'
    FAILED = 'failed'
    NOT_COVERED = 'not_covered'
    RESTRICTED_GUIDANCE = 'restricted_guidance'
    PENDING_CONFIRMATION = 'pending_confirmation'


@dataclass
class TaskContext:
    task_type: str
    requested_action: str
    entities: dict = field(default_factory=dict)
    missing_fields: list = field(default_factory=list)
    workflow_status: WorkflowStatus = WorkflowStatus.UNDERSTOOD
    request_id: str = ''
    task_id: str = ''
    planned_action: str | None = None
    workflow_binding: dict = field(default_factory=dict)
    actor_user_id: str = ''
    conversation_id: str = ''
    primary_agent: str | None = None
    supporting_agents: list = field(default_factory=list)
    evidence: list = field(default_factory=list)
    tool_history: list = field(default_factory=list)
    state_history: list = field(default_factory=lambda: ['received', 'understood'])
    urgency: str = 'LOW'
    fine_intent: str = 'symptom_information'
    qic_intent: str | None = None
    agent_route: str | None = None
    business_action: str | None = None
    boundary: str | None = None
    careguide_intent: str | None = None
    task_brief: dict = field(default_factory=dict)

    @classmethod
    def from_intent_result(cls, result: IntentResult):
        context = cls(result.intent.value if result.intent else
                      result.fine_intent if result.disposition == 'route' and result.fine_intent else result.disposition,
                      result.requested_action, dict(result.entities))
        context.urgency = getattr(result, 'urgency', 'LOW')
        context.fine_intent = getattr(result, 'fine_intent', 'symptom_information')
        context.qic_intent = getattr(result, 'qic_intent', None)
        context.agent_route = getattr(result, 'agent_route', None)
        context.business_action = getattr(result, 'business_action', None)
        context.boundary = getattr(result, 'boundary', None)
        context.careguide_intent = getattr(result, 'careguide_intent', None)
        required = {'medication_query': ('drug_name', 'dosage_form', 'specification'),
                    'appointment': ('hospital_id', 'campus')}.get(context.task_type, ())
        context.missing_fields = [field for field in required if not context.entities.get(field)]
        context.transition(WorkflowStatus.NEEDS_CLARIFICATION if context.missing_fields else WorkflowStatus.READY)
        return context

    def transition(self, status):
        self.workflow_status = WorkflowStatus(status)
        self.state_history.append(self.workflow_status.value)

    def execution_payload(self):
        data = asdict(self)
        data['workflow_status'] = self.workflow_status.value
        return data


class SessionMemory:
    """Bounded process-local working memory; optional Redis/Chroma backend is separate."""
    def __init__(self, max_sessions=1000, max_messages=20):
        self.max_sessions, self.max_messages = max_sessions, max_messages
        self.sessions = {}

    async def history(self, user_id, conv_id, query=''):
        return [dict(item) for item in self.sessions.get((user_id, conv_id), [])]

    async def add(self, user_id, conv_id, role, content):
        key = (user_id, conv_id)
        if key not in self.sessions and len(self.sessions) >= self.max_sessions:
            del self.sessions[next(iter(self.sessions))]
        messages = self.sessions.setdefault(key, [])
        messages.append({'role': role, 'content': content})
        del messages[:-self.max_messages]

    async def close(self):
        pass

    def info(self):
        return {'backend': 'session', 'persistent': False,
                'working_memory': 'process_local', 'episodic_memory': False,
                'user_profile': False}

    async def health(self):
        return {'status': 'ok', **self.info()}


class PersistentMemory:
    """Adapter for Redis working memory and Chroma episodic/profile memory."""
    def __init__(self, manager):
        self.manager = manager

    async def history(self, user_id, conv_id, query=''):
        context = await self.manager.get_context(user_id, conv_id, query=query)
        history = [{'role': m.role.value, 'content': m.content} for m in context.recent_messages[-6:]]
        if context.summary:
            history.insert(0, {'role': 'assistant', 'content': '[历史摘要] '+context.summary})
        if context.relevant_history or context.user_profile:
            import json
            history.insert(0, {'role': 'assistant', 'content': '[历史背景，不是当前症状或医疗证据] '+json.dumps(
                {'relevant_history': context.relevant_history, 'user_profile': context.user_profile}, ensure_ascii=False),
                'durable_preferences': context.user_profile})
        return history

    async def add(self, user_id, conv_id, role, content):
        from careguide.conversation_memory import MsgRole
        await self.manager.add_message(user_id, conv_id, MsgRole(role), content)
        if role == 'assistant':
            return await self.manager.update_profile(user_id, conv_id)

    async def close(self):
        await self.manager.close()

    def info(self):
        return {'backend': 'redis_chroma', 'persistent': True,
                'working_memory': 'redis', 'episodic_memory': 'chroma',
                'user_profile': 'chroma'}

    async def health(self):
        return await self.manager.health()
