"""Persistent atomic quotas shared by concurrent tasks and resumed requests."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
import time
from typing import Any
import uuid

from pydantic import Field

from careguide.orchestration.contracts import Contract


class BudgetExhausted(RuntimeError):
    pass


_CURRENT_BUDGET = ContextVar('careguide_workflow_budget', default=None)
_REQUEST_DEADLINE = ContextVar('careguide_request_deadline', default=None)


def current_budget():
    return _CURRENT_BUDGET.get()


@contextmanager
def use_budget(ledger, *, deadline=None):
    token = _CURRENT_BUDGET.set(ledger)
    deadline_token = _REQUEST_DEADLINE.set(deadline) if deadline is not None else None
    try:
        yield ledger
    finally:
        _CURRENT_BUDGET.reset(token)
        if deadline_token is not None:
            _REQUEST_DEADLINE.reset(deadline_token)


@asynccontextmanager
async def budgeted_call(kind, timeout_s=None):
    ledger = _CURRENT_BUDGET.get()
    if ledger is None:
        yield None
    else:
        async with ledger.call(kind, timeout_s) as reservation:
            yield reservation


class BudgetLimits(Contract):
    model: int = Field(default=20, ge=0, le=100)
    tool: int = Field(default=30, ge=0, le=200)
    retry: int = Field(default=3, ge=0, le=20)
    replan: int = Field(default=3, ge=0, le=10)
    execution_seconds: float = Field(default=120, gt=0, le=3600)
    request_seconds: float = Field(default=45, gt=0, le=180)
    closing_ratio: float = Field(default=0.8, gt=0, lt=1)


@dataclass
class Reservation:
    id: str
    kind: str
    timeout_s: float
    usage: dict[str, int] | None = None

    def record_response(self, response: Any):
        usage = response.get('usage') if isinstance(response, dict) else getattr(response, 'usage', None)
        if usage is None:
            return
        def field(name, default=0):
            value = usage.get(name, default) if isinstance(usage, dict) else getattr(usage, name, default)
            return max(0, int(value or 0))
        try:
            prompt, completion = field('prompt_tokens'), field('completion_tokens')
            self.usage = {'prompt_tokens': prompt, 'completion_tokens': completion,
                          'total_tokens': field('total_tokens', prompt+completion)}
        except (TypeError, ValueError):
            self.usage = None


class BudgetLedger:
    """Calls remain charged on failure; unknown outcomes consume reserved time.

    Execution seconds sum active call durations, including parallel work. The
    separate request deadline is wall time; user waiting is neither duration.
    """
    def __init__(self, path: str | Path, workflow_id: str, actor: str,
                 conversation: str, limits: BudgetLimits | None = None, *, require_existing=False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.workflow_id, self.actor, self.conversation = workflow_id, actor, conversation
        limits = limits or BudgetLimits()
        with self.connect() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS workflow_budgets (
                id TEXT PRIMARY KEY, actor TEXT NOT NULL, conversation TEXT NOT NULL,
                limits_json TEXT NOT NULL, used_json TEXT NOT NULL,
                seconds REAL NOT NULL, tokens_json TEXT NOT NULL)''')
            db.execute('''CREATE TABLE IF NOT EXISTS workflow_reservations (
                id TEXT PRIMARY KEY, workflow TEXT NOT NULL, kind TEXT NOT NULL,
                timeout REAL NOT NULL, status TEXT NOT NULL)''')
            if not require_existing:
                db.execute('INSERT OR IGNORE INTO workflow_budgets VALUES (?,?,?,?,?,?,?)',
                    (workflow_id, actor, conversation, limits.model_dump_json(),
                     json.dumps({key: 0 for key in ('model', 'tool', 'retry', 'replan')}), 0.0,
                     json.dumps({'prompt_tokens': 0, 'completion_tokens': 0,
                                 'total_tokens': 0, 'calls_without_usage': 0})))
            row = self._row(db)
            self.limits = BudgetLimits.model_validate_json(row['limits_json'])

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def _row(self, db):
        row = db.execute('SELECT * FROM workflow_budgets WHERE id=?',
                         (self.workflow_id,)).fetchone()
        if row is None or row['actor'] != self.actor or row['conversation'] != self.conversation:
            raise PermissionError('Budget does not belong to this user and conversation')
        return row

    def reserve(self, kind: str, timeout_s: float = 0, *, exploration=False):
        if kind not in {'model', 'tool', 'retry', 'replan'}:
            raise ValueError('Unknown budget dimension')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = self._row(db)
            limits = BudgetLimits.model_validate_json(row['limits_json'])
            used = json.loads(row['used_json'])
            if used[kind] >= getattr(limits, kind):
                raise BudgetExhausted(kind)
            if exploration and self._closing(used, row['seconds'], limits):
                raise BudgetExhausted('closing')
            pending = db.execute("SELECT COALESCE(SUM(timeout),0) FROM workflow_reservations "
                "WHERE workflow=? AND status='reserved'", (self.workflow_id,)).fetchone()[0]
            remaining = limits.execution_seconds-row['seconds']-pending
            effective_timeout = min(max(0.0, timeout_s), max(0.0, remaining))
            if timeout_s > 0 and effective_timeout <= 0:
                raise BudgetExhausted('execution_seconds')
            used[kind] += 1
            ticket = Reservation(uuid.uuid4().hex, kind, effective_timeout)
            db.execute('UPDATE workflow_budgets SET used_json=? WHERE id=?',
                       (json.dumps(used), self.workflow_id))
            db.execute('INSERT INTO workflow_reservations VALUES (?,?,?,?,?)',
                (ticket.id, self.workflow_id, kind, ticket.timeout_s, 'reserved'))
            return ticket

    def finish(self, ticket: Reservation, elapsed: float, *, unknown=False):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = self._row(db)
            reservation = db.execute('SELECT * FROM workflow_reservations WHERE id=? AND workflow=?',
                (ticket.id, self.workflow_id)).fetchone()
            if reservation is None:
                raise LookupError('Reservation not found')
            if reservation['status'] != 'reserved':
                return
            charge = reservation['timeout'] if unknown else min(max(0, elapsed), reservation['timeout'])
            tokens = json.loads(row['tokens_json'])
            if reservation['kind'] == 'model':
                if ticket.usage is None:
                    tokens['calls_without_usage'] += 1
                else:
                    for key, value in ticket.usage.items():
                        tokens[key] += value
            db.execute('UPDATE workflow_budgets SET seconds=seconds+?,tokens_json=? WHERE id=?',
                       (charge, json.dumps(tokens), self.workflow_id))
            db.execute('UPDATE workflow_reservations SET status=? WHERE id=?',
                       ('unknown' if unknown else 'settled', ticket.id))

    @staticmethod
    def _closing(used, seconds, limits):
        return (seconds >= limits.execution_seconds*limits.closing_ratio or any(
            getattr(limits, key) > 0 and used[key] >= getattr(limits, key)*limits.closing_ratio
            for key in ('model', 'tool')))

    def summary(self):
        with self.connect() as db:
            row = self._row(db)
            used, tokens = json.loads(row['used_json']), json.loads(row['tokens_json'])
            limits = BudgetLimits.model_validate_json(row['limits_json'])
            pending = db.execute('SELECT status,COUNT(*) AS n FROM workflow_reservations '
                                 'WHERE workflow=? GROUP BY status', (self.workflow_id,)).fetchall()
            return {'limits': limits.model_dump(), 'used': used,
                    'active_execution_seconds': row['seconds'], 'tokens': tokens,
                    'usage_accounting': 'provider_reported; missing usage is counted separately',
                    'closing': self._closing(used, row['seconds'], limits),
                    'reservations': {item['status']: item['n'] for item in pending}}

    def recover_interrupted_reservations(self):
        """Only call after proving the prior workflow execution is no longer live."""
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            self._row(db)
            pending = db.execute("SELECT * FROM workflow_reservations WHERE workflow=? AND status='reserved'",
                                 (self.workflow_id,)).fetchall()
            charge = sum(item['timeout'] for item in pending)
            db.execute('UPDATE workflow_budgets SET seconds=seconds+? WHERE id=?',
                       (charge, self.workflow_id))
            db.execute("UPDATE workflow_reservations SET status='unknown' WHERE workflow=? AND status='reserved'",
                       (self.workflow_id,))
            return len(pending)

    @asynccontextmanager
    async def call(self, kind, timeout_s=None, *, exploration=False):
        if timeout_s is None:
            timeout_s = min(30, self.limits.request_seconds) if kind in {'model', 'tool'} else 0
        deadline = _REQUEST_DEADLINE.get()
        if deadline is not None:
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise BudgetExhausted('request_deadline')
            if timeout_s > 0:
                timeout_s = min(timeout_s, remaining)
        ticket = await asyncio.to_thread(self.reserve, kind, timeout_s, exploration=exploration)
        started = time.monotonic()
        from careguide.timing import start_span, end_span
        span = start_span(kind)
        try:
            if ticket.timeout_s:
                async with asyncio.timeout(ticket.timeout_s):
                    yield ticket
            else:
                yield ticket
        except (TimeoutError, asyncio.CancelledError):
            await asyncio.shield(asyncio.to_thread(self.finish, ticket,
                time.monotonic()-started, unknown=True))
            raise
        except BaseException:
            await asyncio.to_thread(self.finish, ticket, time.monotonic()-started)
            raise
        else:
            await asyncio.to_thread(self.finish, ticket, time.monotonic()-started)
        finally:
            end_span(span)
