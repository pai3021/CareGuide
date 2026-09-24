"""Synthetic hospital workflow backends.

The public contract is deliberately administrative.  It never connects to a
real hospital in this project and every returned object carries simulation=true.
"""
from __future__ import annotations

import asyncio
import copy
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

import httpx


class HospitalWorkflowBackend(Protocol):
    name: str

    async def health(self) -> dict[str, Any]: ...
    async def search_doctors(self, *, department: str | None = None) -> list[dict]: ...
    async def list_available_slots(self, *, practitioner_id: str | None = None,
                                   department: str | None = None,
                                   date: str | None = None) -> list[dict]: ...
    async def list_appointments(self, patient_id: str) -> list[dict]: ...
    async def execute(self, action_type: str, arguments: dict,
                      idempotency_key: str) -> dict[str, Any]: ...
    async def close(self) -> None: ...


class DisabledHospitalBackend:
    name = "disabled"

    async def health(self):
        return {"status": "disabled", "backend": self.name, "simulation": True}

    async def search_doctors(self, **_):
        return []

    async def list_available_slots(self, **_):
        return []

    async def list_appointments(self, patient_id):
        return []

    async def execute(self, action_type, arguments, idempotency_key):
        return {"status": "source_unavailable", "message": "模拟医院服务未启用。",
                "simulation": True}

    async def close(self):
        return None


class InMemoryHospitalBackend:
    """Deterministic backend for unit tests and the lightweight local demo."""

    name = "synthetic_memory"

    def __init__(self, *, doctors: list[dict] | None = None, slots: list[dict] | None = None):
        self.doctors = doctors or [
            {"id": "PRA-CARD-01", "name": "合成医生甲", "department": "心血管内科"},
            {"id": "PRA-GASTRO-01", "name": "合成医生乙", "department": "消化内科"},
        ]
        self.slots = {item["id"]: item for item in (slots or [
            {"id": "SLOT-CARD-001", "practitioner_id": "PRA-CARD-01", "department": "心血管内科",
             "start": "2030-01-02T09:00:00+08:00", "end": "2030-01-02T09:15:00+08:00", "status": "free"},
            {"id": "SLOT-GASTRO-001", "practitioner_id": "PRA-GASTRO-01", "department": "消化内科",
             "start": "2030-01-02T10:00:00+08:00", "end": "2030-01-02T10:15:00+08:00", "status": "free"},
        ])}
        self.appointments: dict[str, dict] = {}
        self.results: dict[str, dict] = {}
        self._lock = asyncio.Lock()

    async def health(self):
        return {"status": "ok", "backend": self.name, "simulation": True}

    async def search_doctors(self, *, department=None):
        values = [copy.deepcopy(item) for item in self.doctors
                  if not department or item["department"] == department]
        return [{**item, "simulation": True} for item in values]

    async def list_available_slots(self, *, practitioner_id=None, department=None, date=None):
        values = [copy.deepcopy(item) for item in self.slots.values()
                  if item["status"] == "free"
                  and (not practitioner_id or item["practitioner_id"] == practitioner_id)
                  and (not department or item["department"] == department)
                  and (not date or item["start"].startswith(date))]
        return [{**item, "simulation": True} for item in values]

    async def list_appointments(self, patient_id):
        return [{**copy.deepcopy(item), "simulation": True} for item in self.appointments.values()
                if item["patient_id"] == patient_id]

    async def execute(self, action_type, arguments, idempotency_key):
        async with self._lock:
            if idempotency_key in self.results:
                return {**copy.deepcopy(self.results[idempotency_key]), "idempotent_replay": True}
            before, after = {}, {}
            if action_type == "appointment_create":
                slot = self.slots.get(arguments.get("slot_id"))
                if not slot or slot["status"] != "free":
                    return {"status": "conflict", "message": "号源已不可用。", "simulation": True}
                before["slot"] = copy.deepcopy(slot); slot["status"] = "busy"
                identity = "APT-" + uuid.uuid4().hex[:12].upper()
                appointment = {"id": identity, "patient_id": arguments["patient_id"],
                               "slot_id": slot["id"], "status": "booked",
                               "department": slot["department"], "start": slot["start"]}
                self.appointments[identity] = appointment
                after = {"slot": copy.deepcopy(slot), "appointment": copy.deepcopy(appointment)}
            elif action_type == "appointment_reschedule":
                appointment = self.appointments.get(arguments.get("appointment_id"))
                new_slot = self.slots.get(arguments.get("slot_id"))
                if not appointment or appointment["status"] != "booked" or not new_slot or new_slot["status"] != "free":
                    return {"status": "conflict", "message": "预约或新号源不可用。", "simulation": True}
                old_slot = self.slots[appointment["slot_id"]]
                before = {"appointment": copy.deepcopy(appointment), "old_slot": copy.deepcopy(old_slot),
                          "new_slot": copy.deepcopy(new_slot)}
                old_slot["status"] = "free"; new_slot["status"] = "busy"
                appointment.update(slot_id=new_slot["id"], department=new_slot["department"], start=new_slot["start"])
                after = {"appointment": copy.deepcopy(appointment), "old_slot": copy.deepcopy(old_slot),
                         "new_slot": copy.deepcopy(new_slot)}
            elif action_type == "appointment_cancel":
                appointment = self.appointments.get(arguments.get("appointment_id"))
                if not appointment or appointment["status"] != "booked":
                    return {"status": "conflict", "message": "预约不存在或无法取消。", "simulation": True}
                slot = self.slots[appointment["slot_id"]]
                before = {"appointment": copy.deepcopy(appointment), "slot": copy.deepcopy(slot)}
                appointment["status"] = "cancelled"; slot["status"] = "free"
                after = {"appointment": copy.deepcopy(appointment), "slot": copy.deepcopy(slot)}
            else:
                return {"status": "invalid", "message": "不支持的模拟操作。", "simulation": True}
            result = {"status": "completed", "action_type": action_type, "simulation": True,
                      "before": before, "after": after, "executed_at": datetime.now(timezone.utc).isoformat()}
            self.results[idempotency_key] = result
            return copy.deepcopy(result)

    async def close(self):
        return None


@dataclass
class FHIRHospitalBackend:
    """FHIR R4 HTTP adapter for one explicitly synthetic hospital."""

    base_url: str
    timeout_s: float = 20
    name: str = "hapi_fhir"

    def __post_init__(self):
        self.base_url = self.base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=self.timeout_s,
                                        headers={"Accept": "application/fhir+json",
                                                 "Cache-Control": "no-store"})
        self._slot_locks: dict[str, asyncio.Lock] = {}
        self._results: dict[str, dict] = {}

    async def _get(self, resource: str, params: dict | None = None) -> dict:
        response = await self.client.get(f"{self.base_url}/{resource}", params=params)
        response.raise_for_status(); return response.json()

    @staticmethod
    def _resources(bundle: dict) -> list[dict]:
        return [entry.get("resource", {}) for entry in bundle.get("entry", []) if entry.get("resource")]

    async def health(self):
        try:
            response = await self.client.get(f"{self.base_url}/metadata")
            response.raise_for_status()
            return {"status": "ok", "backend": self.name, "simulation": True,
                    "fhir_version": response.json().get("fhirVersion")}
        except Exception as exc:
            return {"status": "unavailable", "backend": self.name, "simulation": True,
                    "error_type": type(exc).__name__}

    async def search_doctors(self, *, department=None):
        params = {"specialty:text": department} if department else {}
        roles = self._resources(await self._get("PractitionerRole", params))
        async def expand(role):
            identity = role.get("practitioner", {}).get("reference", "").split("/")[-1]
            practitioner = await self._resource("Practitioner", identity) if identity else {}
            name = next((value.get("text") for value in practitioner.get("name", []) if value.get("text")), identity)
            return {"id": identity, "name": name,
                 "department": next((coding.get("display") for specialty in role.get("specialty", [])
                                     for coding in specialty.get("coding", []) if coding.get("display")), department),
                 "role_id": role.get("id"), "simulation": True}
        return list(await asyncio.gather(*(expand(role) for role in roles)))

    async def list_available_slots(self, *, practitioner_id=None, department=None, date=None):
        params = {"status": "free"}
        if practitioner_id: params["schedule.actor"] = f"Practitioner/{practitioner_id}"
        if date: params["start"] = f"ge{date}"
        slots = self._resources(await self._get("Slot", params))
        values = []
        for slot in slots:
            slot_department = next((coding.get("display") for service in slot.get("serviceType", [])
                                    for coding in service.get("coding", []) if coding.get("display")), None)
            if (slot.get("status") != "free" or
                    (date and not slot.get("start", "").startswith(date)) or
                    (department and slot_department != department)):
                continue
            values.append({"id": slot.get("id"), "status": slot.get("status"), "start": slot.get("start"),
                           "end": slot.get("end"), "schedule": slot.get("schedule", {}).get("reference"),
                           "department": slot_department, "simulation": True})
        return values

    async def list_appointments(self, patient_id):
        # HAPI FHIR JPA v8 indexes Appointment.participant.actor through the
        # standard ``actor`` search parameter.  Its ``patient`` parameter does
        # not return these R4 resources, even when the actor is a Patient.
        resources = self._resources(await self._get(
            "Appointment", {"actor": f"Patient/{patient_id}"}))

        async def expand(item):
            slot_ids = [slot.get("reference", "").split("/")[-1]
                        for slot in item.get("slot", []) if slot.get("reference")]
            value = {"id": item.get("id"), "status": item.get("status"),
                     "start": item.get("start"), "end": item.get("end"),
                     "slot_ids": slot_ids, "simulation": True}
            if not slot_ids:
                return value
            try:
                slot = await self._resource("Slot", slot_ids[0])
                value.update({
                    "start": value["start"] or slot.get("start"),
                    "end": value["end"] or slot.get("end"),
                    "department": next((
                        coding.get("display")
                        for service in slot.get("serviceType", [])
                        for coding in service.get("coding", [])
                        if coding.get("display")), None),
                })
                schedule_id = slot.get("schedule", {}).get("reference", "").split("/")[-1]
                if schedule_id:
                    schedule = await self._resource("Schedule", schedule_id)
                    practitioner_id = next((
                        actor.get("reference", "").split("/")[-1]
                        for actor in schedule.get("actor", [])
                        if actor.get("reference", "").startswith("Practitioner/")), None)
                    if practitioner_id:
                        practitioner = await self._resource("Practitioner", practitioner_id)
                        value["practitioner_id"] = practitioner_id
                        value["practitioner_name"] = next((
                            name.get("text") for name in practitioner.get("name", [])
                            if name.get("text")), practitioner_id)
            except Exception:
                # A stale FHIR reference must not hide the appointment itself.
                pass
            return value

        return list(await asyncio.gather(*(expand(item) for item in resources)))

    async def _resource(self, kind: str, identity: str) -> dict:
        return await self._get(f"{kind}/{identity}")

    @staticmethod
    def _put_entry(resource: dict) -> dict:
        version = resource.get("meta", {}).get("versionId")
        request = {"method": "PUT", "url": f"{resource['resourceType']}/{resource['id']}"}
        if version: request["ifMatch"] = f'W/"{version}"'
        return {"resource": resource, "request": request}

    async def execute(self, action_type, arguments, idempotency_key):
        if idempotency_key in self._results:
            return {**self._results[idempotency_key], "idempotent_replay": True}
        slot_id = arguments.get("slot_id", "global")
        lock = self._slot_locks.setdefault(slot_id, asyncio.Lock())
        async with lock:
            entries, before = [], {}
            if action_type == "appointment_create":
                slot = await self._resource("Slot", arguments["slot_id"])
                if slot.get("status") != "free": return {"status": "conflict", "simulation": True}
                before["slot"] = copy.deepcopy(slot); slot["status"] = "busy"
                appointment = {"resourceType": "Appointment", "id": "APT-" + uuid.uuid4().hex[:12].upper(),
                    "status": "booked",
                    "slot": [{"reference": f"Slot/{slot['id']}"}],
                    "participant": [{"actor": {"reference": f"Patient/{arguments['patient_id']}"},
                                     "status": "accepted"}],
                    "identifier": [{"system": "urn:careguide:idempotency", "value": idempotency_key}]}
                entries = [self._put_entry(slot), self._put_entry(appointment)]
            else:
                appointment = await self._resource("Appointment", arguments["appointment_id"])
                patient_reference = f"Patient/{arguments['patient_id']}"
                if not any(item.get("actor", {}).get("reference") == patient_reference
                           for item in appointment.get("participant", [])):
                    return {"status": "conflict", "message": "预约不属于当前合成患者。",
                            "simulation": True}
                old_slot_id = appointment.get("slot", [{}])[0].get("reference", "/").split("/")[-1]
                old_slot = await self._resource("Slot", old_slot_id)
                before = {"appointment": copy.deepcopy(appointment), "old_slot": copy.deepcopy(old_slot)}
                if action_type == "appointment_cancel":
                    appointment["status"] = "cancelled"; old_slot["status"] = "free"
                    entries = [self._put_entry(old_slot), self._put_entry(appointment)]
                elif action_type == "appointment_reschedule":
                    new_slot = await self._resource("Slot", arguments["slot_id"])
                    if new_slot.get("status") != "free": return {"status": "conflict", "simulation": True}
                    before["new_slot"] = copy.deepcopy(new_slot)
                    old_slot["status"] = "free"; new_slot["status"] = "busy"
                    appointment["slot"] = [{"reference": f"Slot/{new_slot['id']}"}]
                    entries = [self._put_entry(old_slot), self._put_entry(new_slot), self._put_entry(appointment)]
                else: return {"status": "invalid", "simulation": True}
            bundle = {"resourceType": "Bundle", "type": "transaction", "entry": entries}
            response = await self.client.post(self.base_url, json=bundle,
                                              headers={"Content-Type": "application/fhir+json"})
            if response.status_code >= 400:
                return {"status": "failed", "simulation": True, "http_status": response.status_code}
            transaction = response.json()
            response_entries = transaction.get("entry", [])
            versions = [{key: entry.get("response", {}).get(key) for key in ("location", "etag", "lastModified")}
                        for entry in response_entries]
            after = {}
            if action_type == "appointment_create":
                after["slot"] = await self._resource("Slot", arguments["slot_id"])
                location = next((entry.get("response", {}).get("location", "") for entry in response_entries
                                 if entry.get("response", {}).get("location", "").startswith("Appointment/")), "")
                appointment_id = location.split("/")[1] if "/" in location else ""
                if appointment_id:
                    after["appointment"] = await self._resource("Appointment", appointment_id)
            else:
                after["appointment"] = await self._resource("Appointment", arguments["appointment_id"])
                after["old_slot"] = await self._resource("Slot", old_slot_id)
                if action_type == "appointment_reschedule":
                    after["new_slot"] = await self._resource("Slot", arguments["slot_id"])
            result = {"status": "completed", "action_type": action_type, "simulation": True,
                      "before": before, "after": after, "resource_versions": versions,
                      "transaction_response": transaction}
            self._results[idempotency_key] = result
            return result

    async def close(self):
        await self.client.aclose()
