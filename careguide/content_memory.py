"""Source-backed, content-first long-term memories and their rebuildable index."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr


class MemoryConflict(ValueError):
    pass


class MemoryCandidate(BaseModel):
    model_config = ConfigDict(extra='forbid')
    content: StrictStr = Field(default='', max_length=800)
    operation: Literal['set', 'delete'] = 'set'
    memory_id: StrictStr | None = Field(default=None, min_length=1, max_length=100)
    target_quote: StrictStr = Field(default='', max_length=800)
    source_message_id: StrictStr = Field(min_length=1, max_length=100)
    source_quote: StrictStr = Field(min_length=1, max_length=1000)


_REMEMBER = re.compile(r'记住|记一下|记下来|记录一下|忘记|不再记|删除.*(?:记忆|记录)')
_PERSONAL = re.compile(r'(?:我|本人)(?:的|有|对|不|在|从|会|每|平时|曾经|之前|现在|通常|一般|经常|一直|长期|习惯|偏好|喜欢|更喜欢|希望|需要|住在|居住|使用|用|是|负责|看不|听不|第一次)')
_TEMPORARY = re.compile(r'今天|明天|这次|本次|临时|仅这|只这|现在.*(?:头痛|发烧|疼|难受)')
_UNCERTAIN = re.compile(r'可能|假如|假设|如果|不确定|医生说|助手建议|资料说|网上说')
_UNSAFE = re.compile(r'密钥|api[ _-]?key|密码|系统提示|忽略.{0,8}(?:系统|指令)|自动.{0,8}(?:挂号|预约|支付)', re.I)
_DELETE = re.compile(r'忘记|删除|不再记|清除')
_CHANGE = re.compile(r'改为|改成|更正|更新|不再|现在|以前|之前|原来|纠正|换成|其实')


def unsafe_memory_content(text):
    return bool(_UNSAFE.search(text))


def eligible_memory_sources(messages):
    """A cheap gate for user-authored durable facts; no business-field whitelist."""
    sources = {}
    for message in messages:
        if getattr(message.role, 'value', message.role) != 'user':
            continue
        identity = message.metadata.get('message_id')
        if not identity:
            continue
        clauses = [part.strip() for part in re.split(r'[。；;\n]|但是|不过', message.content)]
        clauses = [part for part in clauses if part and len(part) <= 1000
                   and (_REMEMBER.search(part) or _PERSONAL.search(part))
                   and not (_TEMPORARY.search(part) or _UNCERTAIN.search(part) or _UNSAFE.search(part))]
        if clauses:
            sources[identity] = {'content': message.content, 'clauses': clauses}
    return sources


def normalize(text):
    return ' '.join(text.split())


def validate_memories(candidates, sources, current, *, check_targets=True):
    """Keep literal user wording; a target must be uniquely identified in the source."""
    by_id = {item['id']: item for item in current}
    seen = set()
    for item in candidates:
        source = sources.get(item.source_message_id)
        if source is None or not any(item.source_quote in clause for clause in source['clauses']):
            raise PermissionError('Memory requires a literal eligible user source')
        if item.operation == 'set':
            if not item.content.strip() or item.content not in item.source_quote or _UNSAFE.search(item.content):
                raise PermissionError('Memory content must be an extractive user statement')
        elif item.content or not item.memory_id or not _DELETE.search(item.source_quote):
            raise PermissionError('Deletion requires an explicit target and user intent')
        if item.memory_id and check_targets:
            target = by_id.get(item.memory_id)
            quote = item.target_quote.strip()
            if (target is None or len(quote) < 2 or quote not in target['content']
                    or quote not in item.source_quote
                    or sum(quote in old['content'] for old in current) != 1):
                raise PermissionError('Memory target is absent or ambiguous in the user source')
            if item.operation == 'set' and not _CHANGE.search(item.source_quote):
                raise PermissionError('Replacement requires explicit correction intent')
        elif not item.memory_id and item.target_quote:
            raise ValueError('A target quote requires a memory id')
        identity = item.memory_id or normalize(item.content)
        if identity in seen:
            raise ValueError('Conflicting operations for one memory')
        seen.add(identity)
    return candidates


def search_tokens(text):
    """Index Latin words and Chinese bigrams without an embedding dependency."""
    tokens = []
    for part in re.findall(r'[a-z0-9_]+|[\u3400-\u9fff]+', text.casefold()):
        if re.fullmatch(r'[\u3400-\u9fff]+', part):
            tokens.extend(part[i:i+2] for i in range(len(part)-1))
        else:
            tokens.append(part)
    return list(dict.fromkeys(tokens))


class ContentMemoryStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS memory_content_heads ('
                       'actor TEXT PRIMARY KEY, revision INTEGER NOT NULL)')
            db.execute('''CREATE TABLE IF NOT EXISTS memory_content (
                actor TEXT NOT NULL, id TEXT NOT NULL, content TEXT NOT NULL,
                content_hash TEXT NOT NULL, status TEXT NOT NULL, version INTEGER NOT NULL,
                source_message_id TEXT NOT NULL, source_quote TEXT NOT NULL,
                conversation TEXT NOT NULL, source_order INTEGER NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL, expires_at TEXT NOT NULL,
                indexed_version INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(actor,id))''')
            db.execute('CREATE INDEX IF NOT EXISTS memory_content_active '
                       'ON memory_content(actor,status,expires_at,updated_at)')
            db.execute("CREATE UNIQUE INDEX IF NOT EXISTS memory_content_unique "
                       "ON memory_content(actor,content_hash) WHERE status='active'")
            db.execute('CREATE VIRTUAL TABLE IF NOT EXISTS memory_content_fts USING fts5('
                       'actor UNINDEXED, memory_id UNINDEXED, tokens)')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _now():
        return datetime.now(timezone.utc).isoformat()

    def snapshot(self, actor):
        with self.connect() as db:
            # Keep the catalog and its CAS revision in the same read transaction.
            db.execute('BEGIN')
            head = db.execute('SELECT revision FROM memory_content_heads WHERE actor=?', (actor,)).fetchone()
            rows = db.execute("SELECT * FROM memory_content WHERE actor=? AND status='active' "
                              'AND expires_at>? ORDER BY updated_at DESC,id', (actor, self._now())).fetchall()
        return (head['revision'] if head else 0), [dict(row) for row in rows]

    def apply(self, db, actor, candidates, *, expected_revision, conversation='', source_order=None):
        """Run inside the caller's write transaction, together with auxiliary preferences."""
        head = db.execute('SELECT revision FROM memory_content_heads WHERE actor=?', (actor,)).fetchone()
        revision = head[0] if head else 0
        if revision != expected_revision:
            raise MemoryConflict('Long-term memory changed during extraction')
        stamp = self._now()
        changed = []
        for item in candidates:
            content = normalize(item.content)
            digest = hashlib.sha256(content.encode()).hexdigest()
            identity = item.memory_id or hashlib.sha256(json.dumps([actor, content], ensure_ascii=False).encode()).hexdigest()
            if not item.memory_id:
                same = db.execute('SELECT id FROM memory_content WHERE actor=? AND content_hash=? '
                                  "ORDER BY (status='active') DESC,updated_at DESC LIMIT 1", (actor, digest)).fetchone()
                if same:
                    identity = same[0]
            cursor = db.execute('SELECT * FROM memory_content WHERE actor=? AND id=?', (actor, identity))
            row = cursor.fetchone()
            old = dict(zip([column[0] for column in cursor.description], row)) if row else None
            order = source_order.get(item.source_message_id, 0) if source_order is not None else 0
            if source_order is not None and (not order or old and order <= old['source_order']):
                raise MemoryConflict('Older sources cannot replace or revive a memory')
            if item.memory_id and (old is None or old['status'] != 'active' or old['expires_at'] <= stamp):
                raise MemoryConflict('Target memory is no longer active')
            if (old and old['status'] == 'active' and old['content'] == content
                    and old['expires_at'] > stamp and source_order is None):
                continue
            if item.operation == 'delete':
                content, digest = '', old['content_hash']
            version = old['version']+1 if old else 1
            created = old['created_at'] if old else stamp
            expires = (datetime.now(timezone.utc)+timedelta(days=180)).isoformat()
            try:
                db.execute('''INSERT INTO memory_content
                    (actor,id,content,content_hash,status,version,source_message_id,source_quote,
                     conversation,source_order,created_at,updated_at,expires_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(actor,id) DO UPDATE SET
                    content=excluded.content,content_hash=excluded.content_hash,status=excluded.status,
                    version=excluded.version,source_message_id=excluded.source_message_id,
                    source_quote=excluded.source_quote,conversation=excluded.conversation,
                    source_order=excluded.source_order,updated_at=excluded.updated_at,expires_at=excluded.expires_at''',
                    (actor, identity, content, digest, 'deleted' if item.operation == 'delete' else 'active',
                     version, item.source_message_id, item.source_quote, conversation, order, created, stamp, expires))
            except sqlite3.IntegrityError as exc:
                raise MemoryConflict('An identical active memory already exists') from exc
            db.execute('DELETE FROM memory_content_fts WHERE actor=? AND memory_id=?', (actor, identity))
            if item.operation != 'delete':
                db.execute('INSERT INTO memory_content_fts VALUES (?,?,?)',
                           (actor, identity, ' '.join(search_tokens(content))))
            changed.append(identity)
        if changed:
            db.execute('INSERT INTO memory_content_heads VALUES (?,?) '
                       'ON CONFLICT(actor) DO UPDATE SET revision=excluded.revision', (actor, revision+1))
        return changed

    def search(self, actor, query='', limit=5):
        tokens = search_tokens(query)[:40]
        with self.connect() as db:
            rows = []
            if tokens:
                expression = ' OR '.join('"'+token+'"' for token in tokens)
                rows = db.execute('''SELECT m.* FROM memory_content_fts f JOIN memory_content m
                    ON m.actor=f.actor AND m.id=f.memory_id
                    WHERE memory_content_fts MATCH ? AND m.actor=? AND m.status='active'
                    AND m.expires_at>? ORDER BY bm25(memory_content_fts),m.updated_at DESC LIMIT ?''',
                    (expression, actor, self._now(), limit)).fetchall()
            if not rows:
                rows = db.execute("SELECT * FROM memory_content WHERE actor=? AND status='active' "
                                  'AND expires_at>? ORDER BY updated_at DESC,id LIMIT ?',
                                  (actor, self._now(), limit)).fetchall()
        return [dict(row) for row in rows]

    def pending(self, actor, limit=20):
        with self.connect() as db:
            return [dict(row) for row in db.execute('SELECT * FROM memory_content WHERE actor=? '
                'AND indexed_version<>version ORDER BY updated_at,id LIMIT ?', (actor, limit))]

    def newer_candidates(self, actor, candidates, source_order):
        """Drop obsolete review sources, including targets whose content was deleted."""
        fresh = []
        with self.connect() as db:
            for item in candidates:
                content = normalize(item.content)
                identity = item.memory_id or hashlib.sha256(json.dumps([actor, content], ensure_ascii=False).encode()).hexdigest()
                digest = hashlib.sha256(content.encode()).hexdigest()
                row = db.execute('SELECT source_order FROM memory_content WHERE actor=? AND '
                    '(id=? OR (? AND content_hash=?)) ORDER BY source_order DESC LIMIT 1',
                    (actor, identity, item.memory_id is None, digest)).fetchone()
                order = source_order.get(item.source_message_id, 0)
                if not order:
                    raise MemoryConflict('Memory source is not in this user archive')
                if row is None or order > row['source_order']:
                    fresh.append(item)
        return fresh

    def mark_indexed(self, actor, identity, version):
        with self.connect() as db:
            db.execute('UPDATE memory_content SET indexed_version=? WHERE actor=? AND id=? AND version=?',
                       (version, actor, identity, version))

    def resolve(self, actor, identities):
        if not identities:
            return []
        with self.connect() as db:
            placeholders = ','.join('?' for _ in identities)
            rows = db.execute(f"SELECT * FROM memory_content WHERE actor=? AND id IN ({placeholders}) "
                              "AND status='active' AND expires_at>?", (actor, *identities, self._now())).fetchall()
        by_id = {row['id']: dict(row) for row in rows}
        return [by_id[identity] for identity in identities if identity in by_id]
