// The server remains authoritative. These guards prevent using obsolete cards
// while a newer snapshot is already present in the browser.
export function acceptWorkflow(current, incoming) {
  if (!incoming) return current
  if (current?.id === incoming.id && (incoming.version < current.version ||
      incoming.version === current.version && incoming.revision < current.revision)) return current
  return incoming
}

export function currentTask(workflow, card, task) {
  return workflow?.id === card?.id && workflow.version === card.version &&
    workflow.tasks.some(item => item.id === task.id && item.revision === task.revision &&
      item.status === task.status)
}

export function currentAction(workflow, action) {
  const binding = action.workflow_binding
  if (!binding) return true
  return workflow?.id === binding.workflow_id && workflow.tasks.some(task =>
    task.id === binding.task_id && task.revision === binding.task_revision &&
    task.action_id === action.id && ['waiting_confirmation', 'reconciling'].includes(task.status))
}

export function answerEvent(workflow, task, values, eventId) {
  if (!workflow || task.status !== 'waiting_input' || !Object.keys(values).length ||
      Object.keys(values).some(field => !task.missing_fields.includes(field) || !String(values[field]).trim())) {
    throw new Error('请填写当前任务所需信息')
  }
  return { kind: 'answer', workflow_id: workflow.id, plan_version: workflow.version,
    task_id: task.id, task_revision: task.revision, values, event_id: eventId }
}
