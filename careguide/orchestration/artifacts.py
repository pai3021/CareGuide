"""Owned, versioned JSON result references backed by the existing business DB."""
from __future__ import annotations

import hashlib
import json
import sqlite3

from careguide.orchestration.decisions import ArtifactReference
from careguide.orchestration.store import WorkflowExpired, utc_now


class ResultArtifacts:
    def __init__(self, path):
        self.path = path
        with sqlite3.connect(path) as db:
            db.execute('''CREATE TABLE IF NOT EXISTS task_artifacts (
                id TEXT PRIMARY KEY, actor TEXT, conversation TEXT, workflow TEXT,
                task TEXT, revision INTEGER, payload TEXT, digest TEXT, size INTEGER)''')

    def put(self, state, task, data):
        raw = json.dumps(data, ensure_ascii=False, sort_keys=True, allow_nan=False)
        digest = hashlib.sha256(raw.encode()).hexdigest()
        identity = hashlib.sha256(f'{state.id}:{task.spec.id}:{task.revision}:{digest}'.encode()).hexdigest()
        size = len(raw.encode())
        with sqlite3.connect(self.path) as db:
            db.execute('INSERT OR IGNORE INTO task_artifacts VALUES (?,?,?,?,?,?,?,?,?)',
                (identity, state.actor, state.conversation, state.id, task.spec.id, task.revision, raw, digest, size))
        return ArtifactReference(id=identity, workflow_id=state.id, task_id=task.spec.id,
                                  task_revision=task.revision, sha256=digest, size=size)

    def read(self, reference, actor, conversation):
        reference = ArtifactReference.model_validate(reference)
        with sqlite3.connect(self.path) as db:
            db.row_factory = sqlite3.Row
            row = db.execute('SELECT * FROM task_artifacts WHERE id=? AND actor=? AND conversation=?',
                             (reference.id, actor, conversation)).fetchone()
            workflow = db.execute('SELECT payload FROM orchestration_workflows WHERE id=? AND actor=? AND conversation=?',
                                  (reference.workflow_id, actor, conversation)).fetchone()
        if row is None or workflow is None:
            raise LookupError('Result reference not found')
        state = json.loads(workflow['payload'])
        task = state['tasks'].get(reference.task_id)
        if (task is None or task['revision'] != reference.task_revision or task['status'] in {
                'cancelled', 'invalidated', 'expired'} or state['expires_at'] <= utc_now().isoformat()
                or task.get('result_valid_until') and task['result_valid_until'] <= utc_now().isoformat()):
            raise WorkflowExpired('Result reference is no longer current')
        if ((row['workflow'], row['task'], row['revision'], row['digest'], row['size']) !=
                (reference.workflow_id, reference.task_id, reference.task_revision, reference.sha256, reference.size)
                or hashlib.sha256(row['payload'].encode()).hexdigest() != reference.sha256):
            raise ValueError('Result reference integrity mismatch')
        return json.loads(row['payload'])
