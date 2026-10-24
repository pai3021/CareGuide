import test from 'node:test'
import assert from 'node:assert/strict'
import { acceptWorkflow, answerEvent, currentAction, currentTask } from '../src/lib/workflow.js'

const task = { id: 'booking', revision: 2, status: 'waiting_input', missing_fields: ['slot_id'] }
const snapshot = { id: 'workflow', version: 3, revision: 10, tasks: [task] }

test('a delayed response cannot restore an obsolete plan or question', () => {
  const previous = { ...snapshot, version: 2, revision: 100 }
  assert.equal(acceptWorkflow(snapshot, previous), snapshot)
  assert.equal(acceptWorkflow(snapshot, { ...snapshot, revision: 9 }), snapshot)
  assert.equal(currentTask(snapshot, previous, task), false)
  assert.equal(currentTask(snapshot, snapshot, { ...task, revision: 1 }), false)
  assert.equal(currentTask(snapshot, snapshot, task), true)
})

test('answer payload binds the exact workflow version, task revision and user event', () => {
  assert.deepEqual(answerEvent(snapshot, task, { slot_id: 'SLOT-2' }, 'event-1'), {
    kind: 'answer', workflow_id: 'workflow', plan_version: 3, task_id: 'booking',
    task_revision: 2, values: { slot_id: 'SLOT-2' }, event_id: 'event-1'
  })
  assert.throws(() => answerEvent(snapshot, task, { slot_id: '' }, 'event-2'))
  assert.throws(() => answerEvent(snapshot, task, { patient_id: 'another-user' }, 'event-2'))
})

test('an unrelated plan version keeps a valid confirmation; a changed task invalidates it', () => {
  const action = { id: 'ACT-1', workflow_binding: {
    workflow_id: 'workflow', plan_version: 1, task_id: 'booking', task_revision: 2 } }
  const current = { ...snapshot, tasks: [{ ...task, status: 'waiting_confirmation', action_id: 'ACT-1' }] }
  assert.equal(currentAction(current, action), true)
  assert.equal(currentAction({ ...current, tasks: [{ ...current.tasks[0], revision: 3 }] }, action), false)
  assert.equal(currentAction({ ...current, tasks: [{ ...current.tasks[0], status: 'cancelled' }] }, action), false)
})
