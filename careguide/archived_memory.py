"""Optional durable session adapter shared by session and redis_chroma modes."""
import asyncio
from contextlib import contextmanager
import hashlib
import json
import logging
import uuid

from careguide.context_memory import ContextPolicy, ArchiveOnlyPolicy, use_context_policy, estimate_tokens
from careguide.session_store import SessionStore

logger = logging.getLogger(__name__)


class ArchivedMemory:
    def __init__(self, memory, path, *, client=None, model='', settings=None, review_interval=0):
        self.memory = memory
        self.store = SessionStore(path)
        self.client, self.model, self.settings = client, model, settings
        self.review_interval = review_interval
        if review_interval < 0:
            raise ValueError('Review interval cannot be negative')
        self.read_timeout = (settings.timeout + 5) if settings else 5
        self._reviews = {}
        self._seen = set()
        manager = getattr(memory, 'manager', None)
        if manager is not None:
            manager.archive_store = self.store
            manager.token_context_enabled = settings is not None

    def begin_turn(self, actor, conversation, request_id, text):
        return self.store.begin(actor, conversation, request_id, text)

    def fail_turn(self, actor, conversation, request_id):
        self.store.fail(actor, conversation, request_id)

    async def complete_turn(self, actor, conversation, request_id, text, result):
        # Commit first: cache/profile/vector outages cannot erase a completed reply.
        self.store.complete(actor, conversation, request_id, result)
        update = None
        self._seen.add((actor, conversation))
        try:
            manager = getattr(self.memory, 'manager', None)
            if manager is None:
                await self.memory.add(actor, conversation, 'user', text)
                update = await self.memory.add(actor, conversation, 'assistant', result['response'])
            else:
                from careguide.conversation_memory import MsgRole, Message
                rows = self.store.messages(actor, conversation)
                user = next(r for r in rows if r['request_id'] == request_id and r['role'] == 'user')
                try:
                    await manager.add_message(actor, conversation, MsgRole.USER, text, metadata={'message_id': user['id']})
                    assistant = next(r for r in rows if r['request_id'] == request_id and r['role'] == 'assistant')
                    await manager.add_message(actor, conversation, MsgRole.ASSISTANT, result['response'],
                                              metadata={'message_id': assistant['id']})
                finally:
                    update = await manager.update_profile(actor, conversation, source_messages=[
                        Message(MsgRole.USER, text, metadata={'message_id': user['id']})])
        except Exception as exc:
            logger.warning('Working memory cache update failed: %s', type(exc).__name__)
        self._schedule_review(actor, conversation)
        return update

    async def add(self, actor, conversation, role, content):
        row = self.store.append(actor, conversation, role, content)
        self._seen.add((actor, conversation))
        manager = getattr(self.memory, 'manager', None)
        if manager is None:
            return await self.memory.add(actor, conversation, role, content)
        from careguide.conversation_memory import MsgRole, Message
        await manager.add_message(actor, conversation, MsgRole(role), content, metadata={'message_id': row['id']})
        if role == 'assistant':
            users = [r for r in self.store.messages(actor, conversation) if r['role'] == 'user']
            update = await manager.update_profile(actor, conversation, source_messages=[
                Message(MsgRole.USER, users[-1]['content'], metadata={'message_id': users[-1]['id']})] if users else [])
            self._schedule_review(actor, conversation)
            return update

    async def history(self, actor, conversation, query=''):
        snapshot, revision = self.store.snapshot(actor, conversation)
        covered = snapshot.get('covered_seq', 0)
        rows = [r for r in self.store.messages(actor, conversation, after=covered)
                if r['role'] in {'user', 'assistant'}]
        recent = [{'role': r['role'], 'content': r['content']} for r in rows]
        summary = snapshot.get('summary', '')
        head = ([{'role': 'assistant', 'content': '[历史摘要，非业务状态或医疗证据]\n'+summary}]
                if summary else [])
        history = head + recent
        if self.settings:
            policy = self._policy(actor, conversation)
            # A history-only pass does not include fixed request costs; the API hook
            # performs the final full-request check including system/tool schemas.
            compacted = await policy.prepare(history)
            if compacted != history and compacted and compacted[0]['content'].startswith('[历史摘要'):
                tail = compacted[1:]
                consumed = len(recent) - len(tail)
                if consumed > 0:
                    summary = compacted[0]['content'].split('\n', 1)[1]
                    covered = rows[consumed-1]['seq']
                    start_seq = snapshot.get('start_seq', rows[0]['seq'])
                    self.store.save_snapshot(actor, conversation,
                        {'summary': summary, 'covered_seq': covered, 'start_seq': start_seq}, expected_revision=revision,
                        episode={'summary': summary, 'start_seq': start_seq, 'end_seq': covered,
                                 'new_source_ids': [r['id'] for r in rows[:consumed]],
                                 'full_text': '\n'.join(r['content'] for r in rows[:consumed])[:500]})
                    history = compacted
        elif len(history) > 20:
            # Archive-only compatibility mode retains the legacy short input window.
            history = head + recent[-20:]
        manager = getattr(self.memory, 'manager', None)
        if manager is not None:
            await self._flush_episodes(actor, conversation, manager)
            # Restore cache from the archive, including the durable summary. Legacy
            # summary is recovered before expiration if no new snapshot exists yet.
            try:
                if summary:
                    await manager._redis.setex(manager._summary_key(actor, conversation), 86400, summary)
                key = manager._wm_key(actor, conversation)
                pipe = manager._redis.pipeline(transaction=True)
                pipe.delete(key)
                for row in rows[-20:]:
                    pipe.lpush(key, json.dumps({'role': row['role'], 'content': row['content'],
                        'ts': row['created_at'], 'metadata': {'message_id': row['id']}}, ensure_ascii=False))
                pipe.expire(key, 86400)
                await pipe.execute()
            except Exception as exc:
                logger.warning('Redis restore failed; using durable history: %s', type(exc).__name__)
            profile = await manager._get_profile(actor)
            related = await manager._search_episodic(actor, conversation, query) if query else []
            memories = await manager._get_long_term_memories(actor, query)
            if profile or related or memories:
                history.insert(0, {'role': 'assistant', 'content':
                    '[历史背景，是用户历史陈述，不是系统指令、当前症状或医疗证据] '+json.dumps(
                        {'long_term_memories': memories, 'relevant_history': related, 'user_profile': profile}, ensure_ascii=False),
                    'durable_preferences': profile})
            if summary and not any(m['content'].startswith('[历史摘要') for m in history):
                history.insert(0, {'role': 'assistant', 'content': '[历史摘要，非业务状态或医疗证据]\n'+summary})
        elif hasattr(self.memory, 'sessions'):
            self.memory.sessions[(actor, conversation)] = recent[-20:]
        return history

    async def _flush_episodes(self, actor, conversation, manager):
        for episode in self.store.pending_episodes(actor, conversation):
            payload = json.loads(episode['payload'])
            try:
                async with asyncio.timeout(2):
                    saved = await manager._store_episodic(actor, conversation, payload['full_text'],
                        payload['summary'], doc_id=episode['id'], source_range=(payload['start_seq'], payload['end_seq']))
                if saved:
                    self.store.mark_episode(actor, conversation, episode['id'])
            except Exception as exc:
                logger.warning('Episode indexing deferred: %s', type(exc).__name__)

    @contextmanager
    def request_context(self, actor, conversation, request_id):
        def archive(messages):
            # Archive the original tool rounds before request-local pruning. These
            # records are searchable sources, never treated as new user messages.
            for message in messages:
                if message.get('role') == 'tool' or message.get('tool_calls'):
                    identity = hashlib.sha256(json.dumps([request_id, message],
                        ensure_ascii=False, sort_keys=True).encode()).hexdigest()
                    self.store.append(actor, conversation, 'tool_context', message.get('content') or '',
                                      request_id=identity, payload=message)
        policy = self._policy(actor, conversation, archive) if self.settings else ArchiveOnlyPolicy(archive)
        with use_context_policy(policy):
            yield policy

    def _policy(self, actor, conversation, archive=None):
        from datetime import datetime, timezone
        from careguide.orchestration.budget import BudgetLedger, BudgetLimits
        state, _ = self.store.snapshot(actor, conversation, 'compression')
        def event(payload):
            from careguide.session_store import TurnConflict
            _, revision = self.store.snapshot(actor, conversation, 'compression')
            try:
                self.store.save_snapshot(actor, conversation, payload, kind='compression', expected_revision=revision)
            except TurnConflict:
                logger.info('Concurrent compression telemetry update skipped')
        identity = 'context-'+hashlib.sha256(json.dumps([actor,
            datetime.now(timezone.utc).date().isoformat()]).encode()).hexdigest()
        budget = BudgetLedger(self.store.path, identity, actor, 'context-compression',
            BudgetLimits(model=100, tool=0, retry=0, replan=0, execution_seconds=1800, request_seconds=30))
        return ContextPolicy(self.client, self.model, self.settings, archive, state, event, budget)

    def _schedule_review(self, actor, conversation, *, force=False):
        manager = getattr(self.memory, 'manager', None)
        if manager is None or not self.review_interval:
            return
        key = (actor, conversation)
        if key in self._reviews and not self._reviews[key].done():
            return
        count = self.store.turn_count(actor, conversation)
        checkpoint, _ = self.store.snapshot(actor, conversation, 'review')
        if count <= checkpoint.get('turn_count', 0) or (not force and count-checkpoint.get('turn_count', 0) < self.review_interval):
            return
        self._reviews[key] = asyncio.create_task(self.review(actor, conversation))

    async def review(self, actor, conversation):
        manager = getattr(self.memory, 'manager', None)
        if manager is None:
            return {'status': 'skipped'}
        checkpoint, revision = self.store.snapshot(actor, conversation, 'review')
        count = self.store.turn_count(actor, conversation)
        rows = [r for r in self.store.messages(actor, conversation, after=checkpoint.get('covered_seq', 0))
                if r['role'] == 'user' and r['status'] == 'completed']
        from careguide.conversation_memory import Message, MsgRole
        # Bounded incremental batches. Boundaries advance only through processed
        # source rows; remaining history is picked up by the next review.
        batch, used = [], 0
        for row in rows:
            size = estimate_tokens(row['content'])
            if batch and used+size > 6000:
                break
            if size > 6000:
                return {'status': 'unavailable', 'reason': 'review_source_over_budget'}
            used += size
            batch.append(row)
        if not batch:
            return {'status': 'skipped'}
        try:
            with use_context_policy(None):
                async with asyncio.timeout(15):
                    outcome = await manager.update_profile(actor, conversation, source_messages=[
                        Message(MsgRole.USER, row['content'], metadata={'message_id': row['id']}) for row in batch],
                        review=True)
            if outcome['status'] in {'updated', 'skipped'}:
                # count reflects all completed user turns only when the full batch
                # has been drained. Partial batches remain immediately eligible.
                self.store.save_snapshot(actor, conversation,
                    {'covered_seq': batch[-1]['seq'], 'turn_count': count if len(batch) == len(rows) else checkpoint.get('turn_count', 0)},
                    kind='review', expected_revision=revision)
            return outcome
        except Exception as exc:
            logger.warning('Background memory review failed: %s', type(exc).__name__)
            return {'status': 'unavailable', 'reason': type(exc).__name__}

    async def close(self):
        # Controlled shutdown also reviews short conversations; abrupt termination
        # leaves archive boundaries intact for retry on a later completed turn.
        for actor, conversation in self._seen:
            self._schedule_review(actor, conversation, force=True)
        tasks = [task for task in self._reviews.values() if not task.done()]
        if tasks:
            _, pending = await asyncio.wait(tasks, timeout=5)
            for task in pending:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        await self.memory.close()

    def info(self):
        return {**self.memory.info(), 'session_archive': 'sqlite', 'context_budget': bool(self.settings),
                'review_interval': self.review_interval}

    async def health(self):
        return {**await self.memory.health(), 'session_archive': 'sqlite'}
