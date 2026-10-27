"""Scoped, append-only chat archive. Context snapshots never replace raw history.

This store is not a workflow executor and cannot commit hospital actions.
All access is scoped by the authenticated actor and conversation pair.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import uuid


def now():
    return datetime.now(timezone.utc).isoformat()


class TurnConflict(ValueError):
    pass


class SessionStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS memory_sessions (
                    actor TEXT NOT NULL, conversation TEXT NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    PRIMARY KEY(actor,conversation));
                CREATE TABLE IF NOT EXISTS memory_messages (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE,
                    actor TEXT NOT NULL, conversation TEXT NOT NULL,
                    request_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL,
                    payload TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL,
                    UNIQUE(actor,conversation,request_id,role));
                CREATE INDEX IF NOT EXISTS memory_message_scope
                    ON memory_messages(actor,conversation,seq);
                CREATE TABLE IF NOT EXISTS memory_turns (
                    actor TEXT NOT NULL, conversation TEXT NOT NULL, id TEXT NOT NULL,
                    input TEXT NOT NULL, status TEXT NOT NULL, result TEXT,
                    PRIMARY KEY(actor,conversation,id));
                CREATE TABLE IF NOT EXISTS memory_snapshots (
                    actor TEXT NOT NULL, conversation TEXT NOT NULL, kind TEXT NOT NULL,
                    payload TEXT NOT NULL, revision INTEGER NOT NULL,
                    PRIMARY KEY(actor,conversation,kind));
                CREATE TABLE IF NOT EXISTS memory_episodes (
                    id TEXT PRIMARY KEY, actor TEXT NOT NULL, conversation TEXT NOT NULL,
                    payload TEXT NOT NULL, indexed INTEGER NOT NULL DEFAULT 0);
            ''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def _append(self, db, actor, conversation, role, content, request_id, payload, status):
        stamp = now()
        db.execute('INSERT INTO memory_sessions VALUES (?,?,?,?) ON CONFLICT(actor,conversation) '
                   'DO UPDATE SET updated_at=excluded.updated_at', (actor, conversation, stamp, stamp))
        identity = hashlib.sha256(json.dumps([actor, conversation, request_id, role]).encode()).hexdigest()
        db.execute('INSERT OR IGNORE INTO memory_messages '
                   '(id,actor,conversation,request_id,role,content,payload,status,created_at) '
                   'VALUES (?,?,?,?,?,?,?,?,?)',
                   (identity, actor, conversation, request_id, role, content,
                    json.dumps(payload, ensure_ascii=False), status, stamp))
        row = db.execute('SELECT * FROM memory_messages WHERE id=?', (identity,)).fetchone()
        if row['content'] != content or json.loads(row['payload']) != payload:
            raise TurnConflict('Message identity was reused with different content')
        return dict(row)

    def append(self, actor, conversation, role, content, *, request_id=None, payload=None):
        with self.connect() as db:
            return self._append(db, actor, conversation, role, content, request_id or uuid.uuid4().hex,
                                payload or {}, 'completed')

    def begin(self, actor, conversation, request_id, text):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            previous = db.execute('SELECT * FROM memory_turns WHERE actor=? AND conversation=? AND id=?',
                                  (actor, conversation, request_id)).fetchone()
            if previous:
                if previous['input'] != text:
                    raise TurnConflict('Request identity was reused with different input')
                if previous['status'] == 'completed':
                    return json.loads(previous['result'])
                # Never automatically replay an interrupted request with possible side effects.
                raise TurnConflict('Request already admitted; inspect task state before retrying')
            db.execute('INSERT INTO memory_turns VALUES (?,?,?,?,?,NULL)',
                       (actor, conversation, request_id, text, 'pending'))
            self._append(db, actor, conversation, 'user', text, request_id, {}, 'pending')
        return None

    def complete(self, actor, conversation, request_id, result):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT status FROM memory_turns WHERE actor=? AND conversation=? AND id=?',
                             (actor, conversation, request_id)).fetchone()
            if row is None:
                raise TurnConflict('Unknown request')
            self._append(db, actor, conversation, 'assistant', result['response'], request_id, {}, 'completed')
            db.execute('UPDATE memory_messages SET status=? WHERE actor=? AND conversation=? AND request_id=?',
                       ('completed', actor, conversation, request_id))
            db.execute('UPDATE memory_turns SET status=?,result=? WHERE actor=? AND conversation=? AND id=?',
                       ('completed', json.dumps(result, ensure_ascii=False), actor, conversation, request_id))

    def fail(self, actor, conversation, request_id):
        with self.connect() as db:
            db.execute("UPDATE memory_turns SET status='failed' WHERE actor=? AND conversation=? AND id=? "
                       "AND status='pending'", (actor, conversation, request_id))
            db.execute("UPDATE memory_messages SET status='failed' WHERE actor=? AND conversation=? "
                       "AND request_id=? AND status='pending'", (actor, conversation, request_id))

    def messages(self, actor, conversation, *, after=0, include_pending=False):
        with self.connect() as db:
            rows = db.execute('SELECT * FROM memory_messages WHERE actor=? AND conversation=? AND seq>? '
                              + ('' if include_pending else "AND status!='pending' ") + 'ORDER BY seq',
                              (actor, conversation, after)).fetchall()
        return [dict(row) for row in rows]

    def snapshot(self, actor, conversation, kind='history'):
        with self.connect() as db:
            row = db.execute('SELECT payload,revision FROM memory_snapshots '
                             'WHERE actor=? AND conversation=? AND kind=?', (actor, conversation, kind)).fetchone()
        return (json.loads(row['payload']), row['revision']) if row else ({}, 0)

    def save_snapshot(self, actor, conversation, payload, *, kind='history', expected_revision=0, episode=None):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute('INSERT OR IGNORE INTO memory_snapshots VALUES (?,?,?,?,0)',
                       (actor, conversation, kind, '{}'))
            updated = db.execute('UPDATE memory_snapshots SET payload=?,revision=revision+1 '
                                 'WHERE actor=? AND conversation=? AND kind=? AND revision=?',
                                 (json.dumps(payload, ensure_ascii=False), actor, conversation, kind, expected_revision))
            if updated.rowcount != 1:
                raise TurnConflict('Context changed during compression')
            if episode:
                identity = hashlib.sha256(json.dumps([actor, conversation, episode['start_seq'],
                                                      episode['end_seq']]).encode()).hexdigest()
                db.execute('INSERT OR IGNORE INTO memory_episodes VALUES (?,?,?,?,0)',
                           (identity, actor, conversation, json.dumps(episode, ensure_ascii=False)))

    def pending_episodes(self, actor, conversation):
        with self.connect() as db:
            rows = db.execute('SELECT * FROM memory_episodes WHERE actor=? AND conversation=? AND indexed=0 LIMIT 3',
                              (actor, conversation)).fetchall()
        return [dict(row) for row in rows]

    def mark_episode(self, actor, conversation, identity):
        with self.connect() as db:
            db.execute('UPDATE memory_episodes SET indexed=1 WHERE actor=? AND conversation=? AND id=?',
                       (actor, conversation, identity))

    def turn_count(self, actor, conversation):
        with self.connect() as db:
            return db.execute("SELECT count(*) FROM memory_messages WHERE actor=? AND conversation=? "
                              "AND role='user' AND status='completed'", (actor, conversation)).fetchone()[0]

    def source_order(self, actor, message_id):
        with self.connect() as db:
            row = db.execute("SELECT seq FROM memory_messages WHERE actor=? AND id=? AND role='user'",
                             (actor, message_id)).fetchone()
        return row['seq'] if row else 0

    def delete(self, actor, conversation):
        with self.connect() as db:
            for table in ('memory_messages', 'memory_snapshots', 'memory_turns', 'memory_sessions', 'memory_episodes'):
                db.execute(f'DELETE FROM {table} WHERE actor=? AND conversation=?', (actor, conversation))
