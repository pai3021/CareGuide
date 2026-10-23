"""Bounded, source-checked durable workflow preferences with optimistic updates."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from contextlib import contextmanager
import json
from pathlib import Path
import re
import sqlite3
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr


KEYS = Literal['preferred_time', 'preferred_hospital', 'preferred_department',
               'preferred_campus', 'response_language']
DURABLE = re.compile(r'以后|长期|一直|通常|一般|每次|记住|忘记|删除.*偏好|不再记')
TEMPORARY = re.compile(r'今天|明天|这次|本次|临时|仅这|只这')
PREFERENCE = re.compile(r'偏好|喜欢|习惯|默认|倾向|优先|记住|忘记|用中文|用英文|一般.*(?:上午|下午|晚上)')


class Candidate(BaseModel):
    model_config = ConfigDict(extra='forbid')
    key: KEYS
    operation: Literal['set', 'delete'] = 'set'
    value: StrictStr | None = Field(default=None, max_length=100)
    source_message_id: str = Field(min_length=1, max_length=100)
    source_quote: str = Field(min_length=1, max_length=500)


class Extraction(BaseModel):
    model_config = ConfigDict(extra='forbid')
    candidates: list[Candidate] = Field(default_factory=list, max_length=5)


def eligible_sources(messages):
    """Gate user statements only; each clause must express durable intent."""
    sources = {}
    for message in messages:
        role = getattr(message.role, 'value', message.role)
        identity = message.metadata.get('message_id')
        if role != 'user' or not identity:
            continue
        clauses = re.split(r'[，,。；;\n]|但是|不过', message.content)
        clauses = [clause.strip() for clause in clauses
                   if DURABLE.search(clause) and PREFERENCE.search(clause)
                   and not TEMPORARY.search(clause)
                   and not re.search(r'可能|假如|假设|如果|不确定|妈妈|爸爸|父亲|母亲|家人|朋友|同事|'
                                     r'太太|丈夫|妻子|对象|孩子|儿子|女儿|医生|建议|推荐|助手|资料说|网上说|他喜欢|她喜欢', clause)]
        if clauses:
            sources[identity] = {'content': message.content, 'clauses': clauses}
    return sources


def validate_candidates(extraction, sources):
    seen = set()
    for item in extraction.candidates:
        if item.key in seen:
            raise ValueError('Conflicting operations for one preference')
        seen.add(item.key)
        source = sources.get(item.source_message_id)
        quote = item.source_quote
        if source is None or not any(quote in clause for clause in source['clauses']) or not DURABLE.search(quote):
            raise PermissionError('Preference requires an eligible literal user source')
        if item.operation == 'delete':
            if item.value is not None or not re.search(r'忘记|删除|不再记|清除', quote):
                raise PermissionError('Deletion requires explicit user intent')
            labels = {'preferred_time': r'时间|时段|上午|下午|晚上',
                'preferred_hospital': r'医院', 'preferred_department': r'科室|内科|外科',
                'preferred_campus': r'院区', 'response_language': r'语言|中文|英文'}
            if not re.search(labels[item.key], quote):
                raise PermissionError('Deletion source does not identify this preference')
        elif not item.value:
            raise ValueError('A durable value cannot be empty')
        elif item.key == 'preferred_time':
            words = {'morning': '上午', 'afternoon': '下午', 'evening': '晚上'}
            word = words.get(item.value)
            if word is None or word not in quote or re.search(r'(?:不喜欢|不要|不选|避免).{0,4}'+word, quote):
                raise PermissionError('Time preference is not supported by its source')
        elif item.key == 'response_language':
            words = {'zh': '中文', 'en': '英文'}
            if (item.value not in words or words[item.value] not in quote
                    or re.search(r'(?:不要|不用|不喜欢).{0,4}'+words.get(item.value, '$^'), quote)):
                raise PermissionError('Language is not supported by its source')
        else:
            labels = {'preferred_hospital': r'医院|医疗中心',
                'preferred_department': r'科|急诊', 'preferred_campus': r'院区'}
            if (item.value not in quote or not re.search(labels[item.key], item.value)
                    or re.search(r'诊断|患有|剂量|药方|忽略|指令|密钥|系统|调用|[\r\n{}<>]', item.value)
                    or re.search(r'(?:不喜欢|不要|不选|避免).{0,4}'+re.escape(item.value), quote)):
                raise PermissionError('Unsupported or unsafe durable preference value')
    return extraction


class ProfileConflict(ValueError):
    pass


class ProfileStore:
    """SQLite owns revisions; Chroma is an atomic upsert projection.

    A failed projection prevents the authoritative update. Network I/O never
    holds a SQLite write lock. Reads prefer this store even if a projection is
    late or concurrent; only the final revision CAS can publish a preference.
    """
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS durable_profiles ('
                'actor TEXT PRIMARY KEY, revision INTEGER NOT NULL, payload TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS profile_events ('
                'actor TEXT NOT NULL, event_id TEXT NOT NULL, status TEXT NOT NULL, '
                'payload TEXT NOT NULL, PRIMARY KEY(actor,event_id))')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        try:
            with db:
                yield db
        finally:
            db.close()

    def ensure(self, actor, previous):
        with self.connect() as db:
            db.execute('INSERT OR IGNORE INTO durable_profiles VALUES (?,?,?)',
                       (actor, 0, json.dumps(previous, ensure_ascii=False)))
        return self.get(actor)

    def get(self, actor):
        with self.connect() as db:
            row = db.execute('SELECT revision,payload FROM durable_profiles WHERE actor=?', (actor,)).fetchone()
        return (row[0], json.loads(row[1])) if row else None

    def processed(self, actor, event_id):
        with self.connect() as db:
            return db.execute('SELECT status FROM profile_events WHERE actor=? AND event_id=?',
                              (actor, event_id)).fetchone() is not None

    def prune(self, actor, *, now=None):
        """Discard expired v2 fields only; legacy snapshots are never inferred."""
        now = now or datetime.now(timezone.utc).isoformat()
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT revision,payload FROM durable_profiles WHERE actor=?', (actor,)).fetchone()
            if row is None:
                return
            profile = json.loads(row[1])
            if profile.get('schema_version') != 2:
                return
            fields = profile.get('fields', {})
            active = {key: field for key, field in fields.items() if field.get('expires_at', '') > now}
            if active != fields:
                profile.update(fields=active, revision=row[0]+1)
                db.execute('UPDATE durable_profiles SET revision=?,payload=? WHERE actor=?',
                    (row[0]+1, json.dumps(profile, ensure_ascii=False), actor))

    def event(self, actor, event_id, status, payload):
        with self.connect() as db:
            db.execute('INSERT OR REPLACE INTO profile_events VALUES (?,?,?,?)',
                       (actor, event_id, status, json.dumps(payload, ensure_ascii=False)))

    def merge(self, actor, extraction, *, expected_revision, projection, deadline=None):
        current = self.get(actor)
        if current is None or current[0] != expected_revision:
            raise ProfileConflict('Profile was updated concurrently; extraction must be refreshed')
        old = current[1]
        now = datetime.now(timezone.utc)
        fields = dict(old.get('fields', {})) if old.get('schema_version') == 2 else {}
        for item in extraction.candidates:
            if item.operation == 'delete':
                fields.pop(item.key, None)
            else:
                fields[item.key] = {'value': item.value, 'source_message_id': item.source_message_id,
                    'source_quote': item.source_quote, 'updated_at': now.isoformat(),
                    'version': expected_revision+1, 'scope': 'nonclinical_workflow_preference',
                    'expires_at': (now+timedelta(days=180)).isoformat()}
        fields = {key: value for key, value in fields.items() if value['expires_at'] > now.isoformat()}
        payload = {'schema_version': 2, 'revision': expected_revision+1, 'fields': fields}
        projection(payload)
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError('Projection arrived after the extraction deadline')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT revision,payload FROM durable_profiles WHERE actor=?', (actor,)).fetchone()
            if row is None or row[0] != expected_revision:
                raise ProfileConflict('Profile was updated concurrently; extraction must be refreshed')
            db.execute('UPDATE durable_profiles SET revision=?,payload=? WHERE actor=?',
                       (expected_revision+1, json.dumps(payload, ensure_ascii=False), actor))
            return payload


def active_profile(profile):
    if profile.get('schema_version') != 2:
        # Preserve legacy storage, but do not reuse unverified legacy prose as a
        # durable instruction or clinical fact in a new conversation.
        return {}
    now = datetime.now(timezone.utc).isoformat()
    return {key: field for key, field in profile.get('fields', {}).items()
            if field.get('expires_at', '') > now}
