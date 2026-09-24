"""BusinessStore lookup and guarded-write helpers for CareGuide.

SQLite stores local consultation handoffs, never hospital bookings or dispatches.
Creation is transactional; open records are reused for the same actor/session.
"""
import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager


class BusinessStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self.connect() as connection:
            connection.executescript('''
                CREATE TABLE IF NOT EXISTS handoffs (
                    id TEXT PRIMARY KEY, actor TEXT NOT NULL, conversation TEXT NOT NULL,
                    status TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE UNIQUE INDEX IF NOT EXISTS open_handoff
                ON handoffs(actor, conversation) WHERE status IN ('open', 'in_progress');
                CREATE TABLE IF NOT EXISTS knowledge (
                    id TEXT PRIMARY KEY, status TEXT NOT NULL, payload TEXT NOT NULL);
            ''')

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def create_handoff(self, actor, conversation, request_id, message, evidence, tool_history, urgency):
        with self._lock, self.connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            existing = connection.execute(
                "SELECT payload FROM handoffs WHERE actor=? AND conversation=? AND status IN ('open','in_progress')",
                (actor, conversation)).fetchone()
            if existing:
                return json.loads(existing['payload']), True
            timestamp = datetime.now(timezone.utc).isoformat()
            item = {'id': 'CG-'+uuid.uuid4().hex[:12].upper(), 'actor_user_id': actor,
                    'conversation_id': conversation, 'request_id': request_id, 'status': 'open',
                    'summary': message, 'evidence': evidence, 'tool_history': tool_history,
                    'urgency': urgency, 'created_at': timestamp, 'updated_at': timestamp,
                    'events': [{'status': 'open', 'at': timestamp}],
                    'destination': 'local_consultation_queue', 'hospital_dispatch_performed': False}
            connection.execute('INSERT INTO handoffs VALUES (?,?,?,?,?)',
                (item['id'], actor, conversation, 'open', json.dumps(item, ensure_ascii=False)))
            return item, False

    def get_handoff(self, identity, actor):
        with self.connect() as connection:
            item = connection.execute('SELECT payload FROM handoffs WHERE id=? AND actor=?', (identity, actor)).fetchone()
            return json.loads(item['payload']) if item else None

    def list_handoffs(self, actor):
        with self.connect() as connection:
            return [json.loads(item['payload']) for item in connection.execute(
                'SELECT payload FROM handoffs WHERE actor=? ORDER BY rowid DESC LIMIT 100', (actor,))]

    def transition_handoff(self, identity, actor, status, note=''):
        transitions = {'open': {'in_progress', 'cancelled'}, 'in_progress': {'resolved', 'cancelled'}}
        with self._lock, self.connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            row = connection.execute('SELECT payload FROM handoffs WHERE id=? AND actor=?', (identity, actor)).fetchone()
            if row is None:
                raise LookupError('Consultation record not found')
            item = json.loads(row['payload'])
            if status not in transitions.get(item['status'], set()):
                raise ValueError('Invalid consultation status transition')
            item['status'] = status
            item['updated_at'] = datetime.now(timezone.utc).isoformat()
            item['events'].append({'status': status, 'at': item['updated_at'], 'note': note})
            connection.execute('UPDATE handoffs SET status=?,payload=? WHERE id=?',
                (status, json.dumps(item, ensure_ascii=False), identity))
            return item
