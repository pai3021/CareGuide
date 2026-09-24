"""Persistent, user-bound confirmation actions for synthetic hospital writes."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ActionStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self.connect() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS pending_actions (
                id TEXT PRIMARY KEY, actor TEXT NOT NULL, conversation TEXT NOT NULL,
                status TEXT NOT NULL, expires_at TEXT NOT NULL, idempotency_key TEXT NOT NULL UNIQUE,
                payload TEXT NOT NULL)""")

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def _digest(action_type: str, actor: str, conversation: str, arguments: dict) -> str:
        raw = json.dumps([action_type, actor, conversation, arguments], ensure_ascii=False,
                         sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()

    def prepare(self, action_type: str, actor: str, conversation: str, arguments: dict,
                *, ttl_minutes: int = 15) -> tuple[dict, bool]:
        idem = self._digest(action_type, actor, conversation, arguments)
        with self._lock, self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT payload FROM pending_actions WHERE idempotency_key=?", (idem,)).fetchone()
            if existing:
                item = json.loads(existing["payload"])
                # Reusing an already completed request is essential: callers may
                # retry after losing the confirmation response.  Executing is
                # also returned as-is so a concurrent retry cannot create a
                # second write.  A live pending request keeps its original TTL.
                if item["status"] in {"executing", "completed"}:
                    return item, True
                if item["status"] == "pending_confirmation" and item["expires_at"] > _now().isoformat():
                    return item, True
            # A changed critical argument creates a new confirmation contract
            # and invalidates older pending confirmations for the same action.
            pending = connection.execute(
                "SELECT id,payload FROM pending_actions WHERE actor=? AND conversation=? AND status='pending_confirmation'",
                (actor, conversation)).fetchall()
            for row in pending:
                old = json.loads(row["payload"])
                if old.get("type") != action_type or old.get("idempotency_key") == idem:
                    continue
                at = _now().isoformat(); old["status"] = "expired"; old["updated_at"] = at
                old["events"].append({"status": "expired", "at": at, "reason": "arguments_changed"})
                connection.execute("UPDATE pending_actions SET status=?,payload=? WHERE id=?",
                                   ("expired", json.dumps(old, ensure_ascii=False), row["id"]))
            created = _now()
            item = {"id": "ACT-" + uuid.uuid4().hex[:12].upper(), "type": action_type,
                    "status": "pending_confirmation", "actor_user_id": actor,
                    "conversation_id": conversation, "arguments": arguments,
                    "arguments_hash": idem, "idempotency_key": idem,
                    "simulation": True, "created_at": created.isoformat(),
                    "expires_at": (created + timedelta(minutes=ttl_minutes)).isoformat(),
                    "events": [{"status": "pending_confirmation", "at": created.isoformat()}]}
            connection.execute("INSERT OR REPLACE INTO pending_actions VALUES (?,?,?,?,?,?,?)",
                (item["id"], actor, conversation, item["status"], item["expires_at"], idem,
                 json.dumps(item, ensure_ascii=False)))
            return item, False

    def get(self, identity: str, actor: str | None = None) -> dict | None:
        query, values = "SELECT payload FROM pending_actions WHERE id=?", [identity]
        if actor is not None:
            query += " AND actor=?"; values.append(actor)
        with self.connect() as connection:
            row = connection.execute(query, values).fetchone()
        if not row:
            return None
        item = json.loads(row["payload"])
        if item["status"] == "pending_confirmation" and item["expires_at"] <= _now().isoformat():
            return self.transition(identity, item["actor_user_id"], item["conversation_id"], "expired")
        return item

    def transition(self, identity: str, actor: str, conversation: str, status: str,
                   *, result: dict | None = None) -> dict:
        allowed = {"pending_confirmation": {"executing", "rejected", "expired"},
                   "executing": {"completed", "failed"}}
        with self._lock, self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT payload FROM pending_actions WHERE id=? AND actor=? AND conversation=?",
                                     (identity, actor, conversation)).fetchone()
            if not row:
                raise LookupError("Action not found")
            item = json.loads(row["payload"])
            if status not in allowed.get(item["status"], set()):
                raise ValueError("Invalid action transition")
            at = _now().isoformat(); item["status"] = status
            event = {"status": status, "at": at}
            if result is not None:
                event["result"] = result; item["result"] = result
            item["events"].append(event); item["updated_at"] = at
            connection.execute("UPDATE pending_actions SET status=?,payload=? WHERE id=?",
                               (status, json.dumps(item, ensure_ascii=False), identity))
            return item

    def list_for_actor(self, actor: str) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute("SELECT payload FROM pending_actions WHERE actor=? ORDER BY rowid DESC LIMIT 100",
                                      (actor,)).fetchall()
        return [self.get(json.loads(row["payload"])["id"], actor) for row in rows]
