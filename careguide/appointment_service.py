"""Application service joining confirmation actions to the hospital backend."""
from __future__ import annotations

from typing import Any

from careguide.action_store import ActionStore
from careguide.hospital_workflow import HospitalWorkflowBackend


WRITE_ACTIONS = {"appointment_create", "appointment_reschedule", "appointment_cancel"}
REQUIRED_ARGUMENTS = {
    "appointment_create": ("patient_id", "slot_id"),
    "appointment_reschedule": ("patient_id", "appointment_id", "slot_id"),
    "appointment_cancel": ("patient_id", "appointment_id"),
}


class AppointmentService:
    def __init__(self, backend: HospitalWorkflowBackend, actions: ActionStore):
        self.backend, self.actions = backend, actions

    def prepare(self, action_type: str, actor: str, conversation: str,
                arguments: dict[str, Any]) -> dict[str, Any]:
        if action_type not in WRITE_ACTIONS:
            raise ValueError("Unsupported appointment action")
        clean = {key: value for key, value in arguments.items() if value not in {None, ""}}
        clean["patient_id"] = actor
        missing = [key for key in REQUIRED_ARGUMENTS[action_type] if not clean.get(key)]
        if missing:
            return {"status": "needs_clarification", "missing_fields": missing,
                    "simulation": True, "message": "请补充模拟预约所需信息。"}
        item, reused = self.actions.prepare(action_type, actor, conversation, clean)
        return {**item, "reused": reused,
                "message": "这是 CareGuide 合成医院操作。确认后才会执行，不能用于真实就诊。"}

    async def confirm(self, identity: str, actor: str, conversation: str) -> dict[str, Any]:
        item = self.actions.get(identity, actor)
        if not item or item["conversation_id"] != conversation:
            raise LookupError("Action not found")
        if item["status"] == "completed":
            return item
        if item["status"] != "pending_confirmation":
            raise ValueError(f"Action is {item['status']}")
        self.actions.transition(identity, actor, conversation, "executing")
        result = await self.backend.execute(item["type"], item["arguments"], item["idempotency_key"])
        final = "completed" if result.get("status") == "completed" else "failed"
        return self.actions.transition(identity, actor, conversation, final, result=result)

    def reject(self, identity: str, actor: str, conversation: str) -> dict[str, Any]:
        return self.actions.transition(identity, actor, conversation, "rejected")
