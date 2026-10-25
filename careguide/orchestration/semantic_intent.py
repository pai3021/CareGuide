"""Budgeted dense retrieval of versioned business examples, never execution authority."""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from contextvars import ContextVar
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time

from careguide.orchestration.budget import BudgetExhausted
from careguide.orchestration.contracts import ACTION_ARGUMENTS

INSTRUCTION = 'Given a user request, retrieve example requests with the same business intent and conversational operation.'
BANK_PATH = Path(__file__).with_name('intent_examples.json')
_ticket = ContextVar('intent_embedding_response_ticket', default=None)


def record_response(response):
    ticket = _ticket.get()
    if ticket is not None:
        ticket.record_response(response)


def load_bank(path=BANK_PATH):
    bank = json.loads(Path(path).read_text(encoding='utf-8'))
    examples = bank['examples']
    if not bank.get('version') or not 1 <= len(examples) <= 64:
        raise ValueError('Invalid intent example bank')
    identities = set()
    for example in examples:
        if (set(example) - {'id', 'text', 'operation', 'actions', 'note'} or example['id'] in identities
                or not isinstance(example['text'], str) or not 1 <= len(example['text']) <= 500
                or example['operation'] not in {'new', 'change', 'answer', 'cancel', 'clarify', 'boundary'}
                or any(action not in ACTION_ARGUMENTS for action in example['actions'])):
            raise ValueError('Invalid business intent reference')
        identities.add(example['id'])
    return bank


def fingerprint(embedding, bank):
    identity = {'schema': 1, 'bank': bank, 'model': embedding.model, 'dimensions': embedding.dimensions,
        'base_url': embedding.base_url, 'instruction': embedding.query_instruction}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def unit_vector(vector, dimensions):
    if len(vector) != dimensions or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in vector):
        raise ValueError('Invalid intent embedding dimensions or values')
    norm = math.sqrt(sum(v*v for v in vector))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError('Zero or invalid intent vector')
    return [v/norm for v in vector]


class SemanticIntentRetriever:
    def __init__(self, embedding, *, bank=None, timeout=3.0, cache_size=256, cache_ttl=300,
                 cooldown=30, index_path=None):
        if not 0 < timeout <= 10 or not 1 <= cache_size <= 4096 or cache_ttl <= 0:
            raise ValueError('Invalid semantic retrieval limits')
        self.embedding = embedding
        self.bank = deepcopy(bank) if bank is not None else load_bank()
        if embedding.query_instruction != INSTRUCTION or embedding.batch_size < len(self.bank['examples']):
            raise ValueError('Intent retrieval requires its own instruction and one bounded bank batch')
        self.identity = fingerprint(embedding, self.bank)
        self.timeout, self.cache_size, self.cache_ttl, self.cooldown = timeout, cache_size, cache_ttl, cooldown
        self.index_path = Path(index_path) if index_path else None
        self._vectors = None
        self._cache = OrderedDict()
        self._index_lock = asyncio.Lock()
        self._slots = asyncio.Semaphore(4)
        self._unavailable_until = 0

    async def _call(self, fn, budget, calls):
        if budget is None:
            raise BudgetExhausted('missing_intent_embedding_budget')
        async with budget.call('model', self.timeout, exploration=True) as ticket:
            calls.append(1)
            token = _ticket.set(ticket)
            try:
                return await asyncio.to_thread(fn, ticket.timeout_s)
            finally:
                _ticket.reset(token)

    async def _index(self, budget, calls):
        async with self._index_lock:
            if self._vectors is not None:
                return self._vectors
            if self.index_path is not None:
                artifact = json.loads(self.index_path.read_text(encoding='utf-8'))
                if artifact.get('schema') != 1 or artifact.get('fingerprint') != self.identity:
                    raise ValueError('Intent index configuration or example version mismatch')
                vectors = artifact['vectors']
            else:
                vectors = await self._call(lambda timeout: self.embedding.embed_documents(
                    [example['text'] for example in self.bank['examples']], timeout=timeout), budget, calls)
            if len(vectors) != len(self.bank['examples']):
                raise ValueError('Intent index length mismatch')
            self._vectors = [unit_vector(vector, self.embedding.dimensions) for vector in vectors]
            return self._vectors

    async def retrieve(self, text, *, budget, scope):
        metadata = {'source': 'semantic_embedding', 'model': self.embedding.model,
            'bank_version': self.bank['version'], 'fingerprint': self.identity,
            'direct_acceptance': False, 'remote_calls': 0}
        key = hashlib.sha256(json.dumps([self.identity, list(scope), text], ensure_ascii=False).encode()).hexdigest()
        now = time.monotonic()
        cached = self._cache.get(key)
        if cached and cached[0] > now:
            self._cache.move_to_end(key)
            return deepcopy(cached[1]), {**metadata, 'status': 'cache_hit'}
        self._cache.pop(key, None)
        if now < self._unavailable_until:
            return [], {**metadata, 'status': 'cooldown'}
        calls = []
        try:
            async with asyncio.timeout(self.timeout):
                async with self._slots:
                    vectors = await self._index(budget, calls)
                    query = unit_vector(await self._call(lambda timeout: self.embedding.embed_query(text, timeout=timeout), budget, calls),
                                        self.embedding.dimensions)
            scores = [max(-1., min(1., sum(a*b for a, b in zip(query, vector)))) for vector in vectors]
            ranked = sorted(zip(scores, self.bank['examples']), key=lambda row: row[0], reverse=True)[:4]
            rows = [{'source': 'semantic_embedding', 'example_id': example['id'], 'example': example['text'],
                'operation': example['operation'], 'actions': list(example['actions']),
                'note': example.get('note', ''), 'similarity': score, 'calibrated': False,
                'reference_only': True, 'bank_version': self.bank['version']} for score, example in ranked]
            self._cache[key] = (time.monotonic()+self.cache_ttl, rows)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
            return deepcopy(rows), {**metadata, 'status': 'retrieved', 'remote_calls': len(calls)}
        except BudgetExhausted:
            return [], {**metadata, 'status': 'budget_exhausted', 'remote_calls': len(calls)}
        except Exception as exc:
            self._unavailable_until = time.monotonic()+self.cooldown
            return [], {**metadata, 'status': 'unavailable', 'diagnostic_type': type(exc).__name__,
                        'remote_calls': len(calls)}

    async def close(self):
        await asyncio.to_thread(self.embedding.close)

    async def export_index(self, *, budget):
        async with asyncio.timeout(self.timeout):
            vectors = await self._index(budget, [])
        return {'schema': 1, 'fingerprint': self.identity, 'bank_version': self.bank['version'],
            'model': self.embedding.model, 'dimensions': self.embedding.dimensions,
            'instruction': self.embedding.query_instruction, 'vectors': vectors}
