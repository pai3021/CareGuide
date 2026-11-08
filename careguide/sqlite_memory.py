"""SQLite memory backend with sqlite-vec retrieval and an optional Redis cache.

The runtime wraps this backend in ArchivedMemory for durable turns, recovery,
compaction and incremental review. Canonical memory uses the same database.
"""
import asyncio
import hashlib
import json
import logging
import math
from datetime import datetime, timezone

from careguide.long_term_memory import LongTermMemory
from careguide.memory_types import Message, MsgRole
from careguide.orchestration.budget import BudgetLedger, BudgetLimits

logger = logging.getLogger(__name__)


class SQLiteMemoryManager(LongTermMemory):
    records_episodes = True

    def __init__(self, path, *, client=None, model='', redis_url=None, redis_client=None,
                 embedding=None, vector_timeout=1.5, min_score=.3):
        super().__init__(path, client=client, model=model)
        if not math.isfinite(vector_timeout) or not 0 < vector_timeout <= 2:
            raise ValueError('Memory vector timeout must be within (0, 2] seconds')
        if not math.isfinite(min_score) or not -1 <= min_score <= 1:
            raise ValueError('Memory vector min score must be within [-1, 1]')
        self.embedding = embedding
        self.vector_timeout, self.min_score = vector_timeout, min_score
        self.vectors = None
        self.vector_status = 'disabled'
        if embedding is not None:
            try:
                from careguide.sqlite_vector_memory import SQLiteMemoryVectors
                self.vectors = SQLiteMemoryVectors(path, embedding)
                self.vector_status = 'ready'
            except Exception as exc:
                self.vector_status = 'unavailable'
                logger.warning('SQLite vector extension unavailable: %s', type(exc).__name__)
        self._redis = redis_client
        if redis_url and self._redis is None:
            import redis.asyncio as redis
            self._redis = redis.from_url(redis_url, decode_responses=True,
                socket_connect_timeout=2, socket_timeout=2, retry_on_timeout=False)

    async def _get_working_memory(self, user_id, conv_id):
        rows = self.archive_store.messages(user_id, conv_id)
        return [Message(MsgRole(row['role']), row['content'], metadata={'message_id': row['id']})
                for row in rows if row['role'] in {'user', 'assistant'} and row['status'] == 'completed']

    async def _get_long_term_memories(self, user_id, query=''):
        semantic, _ = await self._semantic_search(user_id, '', query)
        return self._select_memories(user_id, query, semantic)

    def _select_memories(self, user_id, query, semantic):
        lexical = self._profile_store.memories.search(user_id, query, limit=self.HISTORY_TOP_K)
        rows = list({row['id']: row for row in semantic+lexical}.values())
        selected, used = [], 0
        for row in rows:
            if len(selected) >= self.HISTORY_TOP_K:
                break
            if used + len(row['content']) > 2400:
                continue
            selected.append({key: row[key] for key in ('id', 'content', 'source_message_id', 'version')})
            used += len(row['content'])
        return selected

    async def read_background(self, user_id, conv_id, query):
        profile = await self._get_profile(user_id)
        semantic, episodes = await self._semantic_search(user_id, conv_id, query)
        return profile, episodes, self._select_memories(user_id, query, semantic)

    async def _semantic_search(self, user_id, conv_id, query):
        if self.vectors is None or not query.strip():
            return [], []
        try:
            # At most one document batch and one query call per read, both inside
            # the same wall-clock budget. No late writes after a timeout.
            async with asyncio.timeout(self.vector_timeout):
                day = datetime.now(timezone.utc).date().isoformat()
                identity = 'memory-vector-'+hashlib.sha256(json.dumps([user_id, day]).encode()).hexdigest()
                budget = BudgetLedger(self._profile_store.path, identity, user_id, 'memory-vectors',
                    BudgetLimits(model=100, tool=0, retry=0, replan=0,
                                 execution_seconds=600, request_seconds=self.vector_timeout))
                pending = self.vectors.pending(user_id)
                if pending:
                    async with budget.call('model', self.vector_timeout) as ticket:
                        vectors = await asyncio.to_thread(self.embedding.embed_documents,
                            [row['text'] for row in pending], timeout=ticket.timeout_s)
                    self.vectors.put(user_id, pending, vectors)
                if not self.vectors.has_vectors(user_id):
                    self.vector_status = 'ready'
                    return [], []
                async with budget.call('model', self.vector_timeout) as ticket:
                    query_vector = await asyncio.to_thread(self.embedding.embed_query,
                        query[:2000], timeout=ticket.timeout_s)
                content = self.vectors.search_content(user_id, query_vector,
                    limit=self.HISTORY_TOP_K, min_score=self.min_score)
                episodes = self.vectors.search_episodes(user_id, conv_id, query_vector,
                    limit=self.HISTORY_TOP_K, min_score=self.min_score)
                self.vector_status = 'ready'
                return content, list(dict.fromkeys(episodes))
        except Exception as exc:
            self.vector_status = 'unavailable'
            logger.warning('SQLite semantic memory fallback: %s', type(exc).__name__)
            return [], []

    async def _search_episodic(self, user_id, conv_id, query):
        _, episodes = await self._semantic_search(user_id, conv_id, query)
        return episodes

    async def add_message(self, user_id, conv_id, role, content, metadata=None):
        """Optionally mirror recent messages; SQLite commits always come first."""
        if self._redis is None:
            return
        key = self._wm_key(user_id, conv_id)
        pipe = self._redis.pipeline(transaction=True)
        pipe.lpush(key, json.dumps({'role': role.value, 'content': content,
                                   'ts': datetime.now().isoformat(),
                                   'metadata': metadata or {}}, ensure_ascii=False))
        pipe.ltrim(key, 0, 19)
        pipe.expire(key, 86400)
        await pipe.execute()

    @staticmethod
    def _wm_key(user_id, conv_id):
        actor = user_id.replace('%', '%25').replace(':', '%3A')
        conversation = conv_id.replace('%', '%25').replace(':', '%3A')
        return f'careguide:wm:{actor}:{conversation}'

    @staticmethod
    def _summary_key(user_id, conv_id):
        actor = user_id.replace('%', '%25').replace(':', '%3A')
        conversation = conv_id.replace('%', '%25').replace(':', '%3A')
        return f'careguide:summary:{actor}:{conversation}'

    async def close(self):
        if self._redis is not None:
            await self._redis.aclose()
        if self.embedding is not None:
            await asyncio.to_thread(self.embedding.close)


class SQLiteMemoryBackend:
    """Long-term memory adapter, used underneath the durable session archive."""
    def __init__(self, path, *, client=None, model='', redis_url=None, redis_client=None,
                 embedding=None, vector_timeout=1.5, min_score=.3):
        self.manager = SQLiteMemoryManager(path, client=client, model=model,
            redis_url=redis_url, redis_client=redis_client, embedding=embedding,
            vector_timeout=vector_timeout, min_score=min_score)

    def info(self):
        return {'backend': 'sqlite', 'persistent': True, 'working_memory': 'sqlite',
                'episodic_memory': 'sqlite_vec' if self.manager.vectors is not None else False,
                'long_term_memory': 'sqlite_vec_with_fts5' if self.manager.vectors is not None else 'sqlite_fts5',
                'vector_configured': self.manager.embedding is not None,
                'vector_status': self.manager.vector_status,
                'user_profile': 'sqlite',
                'cache': 'redis' if self.manager._redis is not None else 'none'}

    async def health(self):
        with self.manager._profile_store.memories.connect() as db:
            db.execute('SELECT 1').fetchone()
        cache = 'disabled'
        if self.manager._redis is not None:
            try:
                cache = 'ok' if await self.manager._redis.ping() else 'unavailable'
            except Exception:
                cache = 'unavailable'
        return {**self.info(), 'status': 'degraded' if cache == 'unavailable' else 'ok',
                'cache_status': cache}

    async def close(self):
        await self.manager.close()
