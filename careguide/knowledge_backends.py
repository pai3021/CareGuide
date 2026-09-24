"""Knowledge backend contracts for embedded retrieval and a real MCP server.

The MCP adapter implements the MCP initialize/tools/call exchange over
Streamable HTTP.  It never falls back to embedded retrieval: callers must
select that backend explicitly so scored runs cannot hide an outage.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

import httpx


class KnowledgeBackend(Protocol):
    name: str

    async def search(self, query: str, *, top_k: int = 5,
                     filters: dict[str, str] | None = None) -> dict[str, Any]: ...

    async def health(self) -> dict[str, Any]: ...

    async def close(self) -> None: ...


@dataclass
class EmbeddedKnowledgeBackend:
    knowledge: Any
    records: list[Any]
    name: str = "embedded"

    async def search(self, query: str, *, top_k: int = 5,
                     filters: dict[str, str] | None = None) -> dict[str, Any]:
        records = list(self.records)
        for key, value in (filters or {}).items():
            if records and value and hasattr(records[0], key):
                records = [record for record in records if getattr(record, key, None) == value]
        hits = self.knowledge.search(query, records,
                                     today=datetime.now(timezone.utc).date(), top_k=top_k)
        return {"status": "found" if hits else "not_covered", "evidence": hits,
                "retrieval_backend": self.name, "message": "" if hits else "未检索到适用资料。"}

    async def health(self) -> dict[str, Any]:
        return {"status": "ok", "backend": self.name, "record_count": len(self.records)}

    async def close(self) -> None:
        return None


class WeKnoraMCPBackend:
    """Small MCP Streamable-HTTP client scoped to one read-only search tool."""

    name = "weknora_mcp"

    def __init__(self, endpoint: str, *, token: str = "",
                 search_tool: str = "hybrid_search", knowledge_base_id: str = "",
                 timeout_s: float = 20):
        if not endpoint:
            raise ValueError("WeKnora MCP endpoint is required")
        headers = {"Accept": "application/json, text/event-stream",
                   "Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.endpoint, self.search_tool = endpoint, search_tool
        self.knowledge_base_id = knowledge_base_id
        self.client = httpx.AsyncClient(headers=headers, timeout=timeout_s)
        self._session_id: str | None = None
        self._initialized = False

    @staticmethod
    def _decode(response: httpx.Response) -> dict[str, Any]:
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            events = []
            for line in response.text.splitlines():
                if line.startswith("data:"):
                    value = line[5:].strip()
                    if value and value != "[DONE]":
                        events.append(json.loads(value))
            if not events:
                raise RuntimeError("MCP server returned no data event")
            return events[-1]
        return response.json()

    async def _rpc(self, method: str, params: dict[str, Any] | None = None,
                   *, notification: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if not notification:
            payload["id"] = uuid.uuid4().hex
        if params is not None:
            payload["params"] = params
        headers = {"Mcp-Session-Id": self._session_id} if self._session_id else {}
        response = await self.client.post(self.endpoint, json=payload, headers=headers)
        if response.headers.get("mcp-session-id"):
            self._session_id = response.headers["mcp-session-id"]
        if notification and response.status_code in {200, 202, 204}:
            return {}
        body = self._decode(response)
        if body.get("error"):
            raise RuntimeError(f"MCP error: {body['error'].get('message', 'unknown')}")
        return body.get("result", {})

    async def _initialize(self) -> None:
        if self._initialized:
            return
        await self._rpc("initialize", {"protocolVersion": "2025-03-26", "capabilities": {},
                                        "clientInfo": {"name": "careguide", "version": "0.3.0"}})
        await self._rpc("notifications/initialized", notification=True)
        tools = await self._rpc("tools/list", {})
        names = {item.get("name") for item in tools.get("tools", [])}
        if self.search_tool not in names:
            raise RuntimeError(f"MCP search tool unavailable: {self.search_tool}")
        self._initialized = True

    @staticmethod
    def _normalize(result: dict[str, Any]) -> list[dict[str, Any]]:
        structured = result.get("structuredContent") or {}
        candidates = structured.get("results") or structured.get("data") or []
        if not candidates:
            for block in result.get("content", []):
                if block.get("type") == "text":
                    try:
                        value = json.loads(block.get("text", ""))
                        candidates = value.get("results", value) if isinstance(value, dict) else value
                    except (ValueError, TypeError):
                        continue
        evidence = []
        for index, item in enumerate(candidates if isinstance(candidates, list) else []):
            if not isinstance(item, dict):
                continue
            evidence.append({
                "record_id": item.get("chunk_id") or item.get("id") or f"weknora-{index}",
                "content": item.get("content") or item.get("text") or item.get("chunk") or "",
                "title": item.get("title") or item.get("document_name") or "WeKnora document",
                "score": item.get("score"),
                "citation": {"url": item.get("url") or item.get("source_url") or "",
                             "source_id": item.get("document_id") or item.get("source_id") or ""},
                "metadata": item.get("metadata") or {},
            })
        return evidence

    async def search(self, query: str, *, top_k: int = 5,
                     filters: dict[str, str] | None = None) -> dict[str, Any]:
        try:
            await self._initialize()
            if self.search_tool == "hybrid_search":
                if not self.knowledge_base_id:
                    raise ValueError("WeKnora knowledge base id is required for hybrid_search")
                arguments: dict[str, Any] = {"kb_id": self.knowledge_base_id,
                                             "query": query, "match_count": top_k}
            else:
                arguments = {"query": query, "limit": top_k, "mode": "hybrid"}
                if filters:
                    arguments["filters"] = {key: value for key, value in filters.items() if value}
            result = await self._rpc("tools/call", {"name": self.search_tool, "arguments": arguments})
            evidence = self._normalize(result)
            return {"status": "found" if evidence else "not_covered", "evidence": evidence,
                    "retrieval_backend": self.name,
                    "message": "" if evidence else "WeKnora 未检索到适用资料。"}
        except Exception as exc:
            return {"status": "source_unavailable", "evidence": [],
                    "retrieval_backend": self.name,
                    "message": "知识服务暂不可用。", "error_type": type(exc).__name__}

    async def health(self) -> dict[str, Any]:
        try:
            await self._initialize()
            return {"status": "ok", "backend": self.name, "tool": self.search_tool}
        except Exception as exc:
            return {"status": "unavailable", "backend": self.name,
                    "error_type": type(exc).__name__}

    async def close(self) -> None:
        await self.client.aclose()
