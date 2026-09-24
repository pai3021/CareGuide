"""Structured, request-level execution traces for CareGuide V1."""

from __future__ import annotations

import json
import logging
import threading
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


logger = logging.getLogger(__name__)

TRACE_VERSION = "1.0"
DEFAULT_TRACE_PATH = Path(__file__).resolve().parents[1] / "data" / "traces" / "requests.jsonl"
_SENSITIVE_KEY_PARTS = ("password", "token", "api_key", "authorization", "secret")


def redact_sensitive(value: Any) -> Any:
    """Recursively redact the small V1 set of sensitive mapping keys."""
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            normalized = key.casefold().replace("-", "_")
            redacted[key] = (
                "***REDACTED***"
                if any(part in normalized for part in _SENSITIVE_KEY_PARTS)
                else redact_sensitive(item)
            )
        return redacted
    if isinstance(value, (list, tuple, set)):
        return [redact_sensitive(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass(frozen=True)
class RequestTrace:
    trace_version: str
    request_id: str
    conversation_id: str
    timestamp: str
    input: str
    user_id: str
    task_understanding: dict[str, Any]
    routing: dict[str, Any]
    execution: dict[str, Any]
    evidence: list[dict[str, Any]]
    side_effects: list[dict[str, Any]]
    final: dict[str, Any]
    started_at: str
    finished_at: str
    total_latency_ms: float
    status: str

    def to_dict(self) -> dict[str, Any]:
        return redact_sensitive(asdict(self))


def validate_trace_completeness(trace: RequestTrace | Mapping[str, Any]) -> dict[str, Any]:
    """Check structural completeness only; this does not score correctness."""
    payload = trace.to_dict() if isinstance(trace, RequestTrace) else dict(trace)
    missing: list[str] = []

    def require(path: str, value: Any) -> None:
        if value is None or value == "":
            missing.append(path)

    for field_name in (
        "trace_version",
        "request_id",
        "conversation_id",
        "timestamp",
        "input",
        "user_id",
        "task_understanding",
        "routing",
        "execution",
        "evidence",
        "side_effects",
        "final",
        "started_at",
        "finished_at",
        "total_latency_ms",
        "status",
    ):
        if field_name not in payload:
            missing.append(field_name)

    require("request_id", payload.get("request_id"))
    require("started_at", payload.get("started_at"))
    require("finished_at", payload.get("finished_at"))
    require("status", payload.get("status"))

    understanding = payload.get("task_understanding") or {}
    for field_name in ("task_type", "entities", "missing_fields", "workflow_status_after_understanding"):
        if field_name not in understanding:
            missing.append(f"task_understanding.{field_name}")

    routing = payload.get("routing") or {}
    for field_name in ("primary_agent", "supporting_agents", "routing_reason", "routing_confidence"):
        if field_name not in routing:
            missing.append(f"routing.{field_name}")

    execution = payload.get("execution") or {}
    waves = execution.get("execution_waves")
    if waves is None:
        missing.append("execution.execution_waves")
        waves = []
    for wave_index, wave in enumerate(waves):
        if "wave" not in wave:
            missing.append(f"execution.execution_waves[{wave_index}].wave")
        if "agents" not in wave:
            missing.append(f"execution.execution_waves[{wave_index}].agents")
        for agent_index, agent in enumerate(wave.get("agents", [])):
            for field_name in (
                "agent_type",
                "role",
                "started_at",
                "finished_at",
                "latency_ms",
                "success",
                "tools_used",
                "error",
            ):
                if field_name not in agent:
                    missing.append(
                        f"execution.execution_waves[{wave_index}].agents[{agent_index}].{field_name}"
                    )
            selected_skill = agent.get("selected_skill")
            if selected_skill is not None:
                if not isinstance(selected_skill, Mapping):
                    missing.append(
                        f"execution.execution_waves[{wave_index}].agents[{agent_index}].selected_skill"
                    )
                else:
                    for field_name in ("skill_id", "version", "selection_reason"):
                        if not selected_skill.get(field_name):
                            missing.append(
                                "execution.execution_waves"
                                f"[{wave_index}].agents[{agent_index}]"
                                f".selected_skill.{field_name}"
                            )

    for tool_index, tool in enumerate(execution.get("tool_calls", [])):
        for field_name in ("tool_name", "agent_type", "input", "success", "output", "error", "latency_ms"):
            if field_name not in tool:
                missing.append(f"execution.tool_calls[{tool_index}].{field_name}")

    final = payload.get("final") or {}
    for field_name in (
        "workflow_status",
        "final_response",
        "agent_types",
        "primary_agent",
        "supporting_agents",
        "total_tool_calls",
        "total_latency_ms",
    ):
        if field_name not in final:
            missing.append(f"final.{field_name}")

    unique_missing = list(dict.fromkeys(missing))
    return {"complete": not unique_missing, "missing_fields": unique_missing}


class TraceStore:
    """In-memory request index with optional append-only JSONL persistence.

    One request ID maps to one final trace. Duplicate appends are rejected.
    Malformed or duplicate persisted lines are skipped during reload so one bad
    line cannot make the store unavailable.
    """

    def __init__(self, path: str | Path | None = DEFAULT_TRACE_PATH) -> None:
        self.path = Path(path) if path is not None else None
        self._lock = threading.Lock()
        self._traces: dict[str, dict[str, Any]] = {}
        self._order: list[str] = []
        self.load_errors: list[dict[str, Any]] = []
        if self.path is not None and self.path.exists():
            self._load()

    def _load(self) -> None:
        assert self.path is not None
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    trace = json.loads(line)
                    request_id = trace["request_id"]
                    if not isinstance(trace, dict) or not isinstance(request_id, str) or not request_id:
                        raise ValueError("trace must be an object with a request_id")
                    if request_id in self._traces:
                        raise ValueError(f"duplicate request_id: {request_id}")
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    error = {"line": line_number, "error": str(exc)}
                    self.load_errors.append(error)
                    logger.warning("Skipping malformed trace line %s in %s: %s", line_number, self.path, exc)
                    continue
                self._traces[request_id] = trace
                self._order.append(request_id)

    def append(self, trace: RequestTrace | Mapping[str, Any]) -> dict[str, Any]:
        payload = trace.to_dict() if isinstance(trace, RequestTrace) else redact_sensitive(dict(trace))
        # Normalize through JSON once so in-memory and persisted values are identical.
        payload = json.loads(json.dumps(payload, ensure_ascii=False, default=str))
        request_id = payload.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("trace request_id is required")

        with self._lock:
            if request_id in self._traces:
                raise ValueError(f"trace already exists for request_id: {request_id}")
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._traces[request_id] = payload
            self._order.append(request_id)
        return deepcopy(payload)

    def get(self, request_id: str) -> dict[str, Any] | None:
        trace = self._traces.get(request_id)
        return deepcopy(trace) if trace is not None else None

    def list_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        if not self._order:
            return []
        normalized_limit = max(1, min(int(limit or 20), len(self._order)))
        request_ids = reversed(self._order[-normalized_limit:])
        return [deepcopy(self._traces[request_id]) for request_id in request_ids]
