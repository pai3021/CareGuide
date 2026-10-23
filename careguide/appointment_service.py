"""Application service joining confirmation actions to the hospital backend."""
from __future__ import annotations

import asyncio
from typing import Any

from careguide.action_store import ActionStore
from careguide.hospital_workflow import HospitalWorkflowBackend
from careguide.orchestration.budget import (BudgetExhausted, BudgetLedger,
    budgeted_call, current_budget, use_budget)


WRITE_ACTIONS = {"appointment_create", "appointment_reschedule", "appointment_cancel"}
REQUIRED_ARGUMENTS = {
    "appointment_create": ("patient_id", "slot_id"),
    "appointment_reschedule": ("patient_id", "appointment_id", "slot_id"),
    "appointment_cancel": ("patient_id", "appointment_id"),
}


class AppointmentService:
    def __init__(self, backend: HospitalWorkflowBackend, actions: ActionStore, *, timeout_s=10):
        self.backend, self.actions = backend, actions
        self.timeout_s = timeout_s

    def prepare(self, action_type: str, actor: str, conversation: str,
                arguments: dict[str, Any], *, workflow_binding=None) -> dict[str, Any]:
        if action_type not in WRITE_ACTIONS:
            raise ValueError("Unsupported appointment action")
        clean = {key: value for key, value in arguments.items()
                 if key in REQUIRED_ARGUMENTS[action_type] and value is not None and value != ''}
        clean["patient_id"] = actor
        if any(not isinstance(value, str) or not 1 <= len(value) <= 100 for value in clean.values()):
            raise ValueError('Invalid appointment argument type or length')
        missing = [key for key in REQUIRED_ARGUMENTS[action_type] if not clean.get(key)]
        if missing:
            return {"status": "needs_clarification", "missing_fields": missing,
                    "simulation": True, "message": "请补充模拟预约所需信息。"}
        item, reused = self.actions.prepare(action_type, actor, conversation, clean,
                                           workflow_binding=workflow_binding)
        return {**item, "reused": reused,
                "message": ('该模拟操作结果待核对；不会重新提交。' if item['status'] in {'executing', 'reconciling'}
                    else '该模拟操作已完成，复用原有记录。' if item['status'] == 'completed'
                    else "这是 CareGuide 合成医院操作。确认后才会执行，不能用于真实就诊。")}

    async def confirm(self, identity: str, actor: str, conversation: str) -> dict[str, Any]:
        item = self.actions.get(identity, actor)
        if item is None or item['conversation_id'] != conversation:
            raise LookupError('Action not found')
        binding = item.get('workflow_binding')
        if binding and item['status'] != 'completed' and current_budget() is None:
            # Enforce persisted limits even when the outer graph is switched
            # off for rollback. A missing ledger must never create fresh quota.
            budget = BudgetLedger(self.actions.path, binding['workflow_id'], actor,
                                  conversation, require_existing=True)
            with use_budget(budget):
                return await self._confirm(identity, actor, conversation)
        return await self._confirm(identity, actor, conversation)

    async def _confirm(self, identity: str, actor: str, conversation: str) -> dict[str, Any]:
        item = self.actions.get(identity, actor)
        if item is None or item['conversation_id'] != conversation:
            raise LookupError('Action not found')
        if item["status"] == "completed":
            return item
        if item['status'] in {'executing', 'reconciling'}:
            return await self.reconcile(identity, actor, conversation)
        if item['status'] != 'pending_confirmation':
            raise ValueError(f"Action is {item['status']}")
        claimed = False
        try:
            # Reserve before the atomic claim. A refused budget leaves the
            # pending contract untouched and cannot create an unknown write.
            async with budgeted_call('tool', self.timeout_s):
                item, claimed = self.actions.claim_confirmation(identity, actor, conversation)
                if claimed:
                    result = await asyncio.wait_for(self.backend.execute(
                        item["type"], item["arguments"], item["idempotency_key"]), timeout=self.timeout_s)
        except BudgetExhausted:
            raise
        except asyncio.CancelledError:
            if claimed:
                self.actions.transition(identity, actor, conversation, 'reconciling')
            raise
        except Exception:
            if not claimed:
                raise
            self.actions.transition(identity, actor, conversation, 'reconciling')
            return await self.reconcile(identity, actor, conversation)
        if not claimed:
            if item['status'] == 'completed':
                return item
            if item['status'] in {'executing', 'reconciling'}:
                return await self.reconcile(identity, actor, conversation)
            raise ValueError(f"Action is {item['status']}")
        if not isinstance(result, dict) or result.get('status') not in {
                'completed', 'failed', 'conflict', 'not_found', 'source_unavailable', 'permission_denied'}:
            self.actions.transition(identity, actor, conversation, 'reconciling')
            return await self.reconcile(identity, actor, conversation)
        final = "completed" if result.get("status") == "completed" else "failed"
        return self.actions.transition(identity, actor, conversation, final, result=result)

    async def reconcile(self, identity, actor, conversation):
        item = self.actions.get(identity, actor)
        if item is None or item['conversation_id'] != conversation:
            raise LookupError('Action not found')
        if item['status'] not in {'executing', 'reconciling'}:
            return item
        lookup = getattr(self.backend, 'reconcile', None)
        if lookup is not None:
            try:
                async with budgeted_call('tool', self.timeout_s):
                    result = await asyncio.wait_for(lookup(item['type'], item['arguments'],
                        item['idempotency_key']), timeout=self.timeout_s)
            except BudgetExhausted:
                raise
            except Exception:
                result = {'status': 'unknown'}
            if isinstance(result, dict) and result.get('status') == 'completed':
                return self.actions.transition(identity, actor, conversation, 'completed', result=result)
        current = self.actions.get(identity, actor)
        if current['status'] == 'completed':
            return current
        return {**self.actions.transition(identity, actor, conversation, 'reconciling'),
                'message': '模拟操作结果尚未核实；不会自动重复提交，请稍后核对。'}

    def reject(self, identity: str, actor: str, conversation: str) -> dict[str, Any]:
        return self.actions.transition(identity, actor, conversation, "rejected")
