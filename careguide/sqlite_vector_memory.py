"""Rebuildable Memory vectors in the canonical SQLite database.

Uses sqlite-vec scalar cosine distance over scoped, valid rows (exact KNN).
Embedding identity includes model, dimensions, endpoint and query instruction.
Vectors never replace canonical text or its version/deletion checks.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import struct


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


class SQLiteMemoryVectors:
    def __init__(self, path, embedding):
        self.path = Path(path)
        self.dimensions = int(embedding.dimensions)
        if not 0 < self.dimensions <= 65536:
            raise ValueError('Invalid memory embedding dimensions')
        identity = {'model': embedding.model, 'dimensions': self.dimensions,
                    'provider': getattr(embedding, 'provider_name', ''),
                    'endpoint': getattr(embedding, 'base_url', ''),
                    'instruction': getattr(embedding, 'query_instruction', '')}
        self.space = digest(json.dumps(identity, sort_keys=True, ensure_ascii=False))
        with self.connect() as db:
            self.version = db.execute('SELECT vec_version()').fetchone()[0]
            db.execute('''CREATE TABLE IF NOT EXISTS memory_vectors (
                actor TEXT NOT NULL, kind TEXT NOT NULL CHECK(kind IN ('content','episode')),
                identity TEXT NOT NULL, space TEXT NOT NULL, revision INTEGER NOT NULL,
                content_hash TEXT NOT NULL, dimensions INTEGER NOT NULL,
                embedding BLOB NOT NULL CHECK(length(embedding)=4*dimensions),
                PRIMARY KEY(actor,kind,identity,space))''')

    @contextmanager
    def connect(self):
        import sqlite_vec
        db = sqlite3.connect(self.path, timeout=1)
        db.row_factory = sqlite3.Row
        try:
            db.enable_load_extension(True)
            try:
                sqlite_vec.load(db)
            finally:
                db.enable_load_extension(False)
            db.create_function('memory_sha256', 1, digest, deterministic=True)
            with db:
                yield db
        finally:
            db.close()

    def pack(self, vector):
        values = [float(value) for value in vector]
        if len(values) != self.dimensions or not all(math.isfinite(value) for value in values):
            raise ValueError('Invalid memory vector dimension or values')
        if not any(value != 0 for value in values):
            raise ValueError('Zero memory vector cannot use cosine distance')
        blob = struct.pack(f'<{self.dimensions}f', *values)
        rounded = struct.unpack(f'<{self.dimensions}f', blob)
        if not all(math.isfinite(value) for value in rounded) or not any(rounded):
            raise ValueError('Memory vector overflows float32')
        return blob

    @staticmethod
    def now():
        return datetime.now(timezone.utc).isoformat()

    def pending(self, actor, limit=16):
        with self.connect() as db:
            # Canonical joins also remove late, deleted and superseded vectors.
            db.execute('''DELETE FROM memory_vectors AS v WHERE v.actor=? AND (
                (v.kind='content' AND NOT EXISTS(SELECT 1 FROM memory_content m
                  WHERE m.actor=v.actor AND m.id=v.identity AND m.status='active'
                    AND m.expires_at>? AND m.version=v.revision AND m.content_hash=v.content_hash))
                OR (v.kind='episode' AND NOT EXISTS(SELECT 1 FROM memory_episodes e
                  WHERE e.actor=v.actor AND e.id=v.identity
                    AND memory_sha256(json_extract(e.payload,'$.summary'))=v.content_hash)))''',
                       (actor, self.now()))
            content = [dict(row) for row in db.execute('''SELECT 'content' AS kind,
                m.id AS identity,m.version AS revision,m.content_hash,m.content AS text
                FROM memory_content m LEFT JOIN memory_vectors v ON v.actor=m.actor
                  AND v.kind='content' AND v.identity=m.id AND v.space=?
                  AND v.revision=m.version AND v.content_hash=m.content_hash
                WHERE m.actor=? AND m.status='active' AND m.expires_at>? AND v.identity IS NULL
                ORDER BY m.updated_at DESC,m.id LIMIT ?''', (self.space, actor, self.now(), limit))]
            remaining = limit-len(content)
            episodes = [dict(row) for row in db.execute('''SELECT 'episode' AS kind,e.id AS identity,
                1 AS revision, memory_sha256(json_extract(e.payload,'$.summary')) AS content_hash,
                json_extract(e.payload,'$.summary') AS text FROM memory_episodes e
                LEFT JOIN memory_vectors v ON v.actor=e.actor AND v.kind='episode'
                  AND v.identity=e.id AND v.space=?
                  AND v.content_hash=memory_sha256(json_extract(e.payload,'$.summary'))
                WHERE e.actor=? AND v.identity IS NULL ORDER BY e.rowid DESC LIMIT ?''',
                (self.space, actor, remaining))] if remaining else []
        return content+episodes

    def put(self, actor, rows, vectors):
        if len(rows) != len(vectors):
            raise ValueError('Embedding batch length mismatch')
        packed = [self.pack(vector) for vector in vectors]
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            for row, blob in zip(rows, packed):
                # Recheck after embedding: the user may have changed/deleted text.
                if row['kind'] == 'content':
                    valid = db.execute('''SELECT 1 FROM memory_content WHERE actor=? AND id=?
                        AND status='active' AND expires_at>? AND version=? AND content_hash=?''',
                        (actor, row['identity'], self.now(), row['revision'], row['content_hash'])).fetchone()
                else:
                    valid = db.execute('''SELECT 1 FROM memory_episodes WHERE actor=? AND id=?
                        AND memory_sha256(json_extract(payload,'$.summary'))=?''',
                        (actor, row['identity'], row['content_hash'])).fetchone()
                if valid:
                    db.execute('INSERT OR REPLACE INTO memory_vectors VALUES (?,?,?,?,?,?,?,?)',
                        (actor, row['kind'], row['identity'], self.space, row['revision'],
                         row['content_hash'], self.dimensions, blob))

    def has_vectors(self, actor):
        with self.connect() as db:
            return db.execute('SELECT 1 FROM memory_vectors WHERE actor=? AND space=? '
                              'AND dimensions=? LIMIT 1', (actor, self.space, self.dimensions)).fetchone() is not None

    def search_content(self, actor, vector, *, limit=5, min_score=.3):
        with self.connect() as db:
            rows = db.execute('''WITH matches AS MATERIALIZED (
                SELECT m.*, vec_distance_cosine(v.embedding,?) AS distance
                FROM memory_vectors v JOIN memory_content m ON m.actor=v.actor AND m.id=v.identity
                  AND m.version=v.revision AND m.content_hash=v.content_hash
                WHERE v.actor=? AND v.kind='content' AND v.space=? AND v.dimensions=?
                  AND m.status='active' AND m.expires_at>?)
                SELECT * FROM matches WHERE distance<=? ORDER BY distance,id LIMIT ?''',
                (self.pack(vector), actor, self.space, self.dimensions, self.now(), 1-min_score, limit))
            return [dict(row) for row in rows]

    def search_episodes(self, actor, conversation, vector, *, limit=5, min_score=.3):
        with self.connect() as db:
            rows = db.execute('''WITH matches AS MATERIALIZED (
                SELECT e.id,e.conversation,e.payload,vec_distance_cosine(v.embedding,?) AS distance
                FROM memory_vectors v JOIN memory_episodes e ON e.actor=v.actor AND e.id=v.identity
                  AND memory_sha256(json_extract(e.payload,'$.summary'))=v.content_hash
                WHERE v.actor=? AND v.kind='episode' AND v.space=? AND v.dimensions=?)
                SELECT * FROM matches WHERE distance<=?
                ORDER BY distance,(conversation=?) DESC,id LIMIT ?''',
                (self.pack(vector), actor, self.space, self.dimensions, 1-min_score, conversation, limit))
            return [json.loads(row['payload'])['summary'] for row in rows]
