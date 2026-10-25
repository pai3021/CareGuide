import { postChatStream } from './sse'

const BACKEND = {
  id: 'careguide',
  label: 'CareGuide',
  baseUrl: import.meta.env.VITE_CAREGUIDE_API_URL || '/api',
  port: '8010'
}

export function createInitialSettings() {
  const saved = readSettings()
  return {
    backend: BACKEND.id,
    userId: saved.userId || 'u001',
    conversationId: saved.conversationId || '',
    endpoint: saved.endpoint || BACKEND.baseUrl,
    theme: saved.theme === 'dark' ? 'dark' : 'light'
  }
}

export function saveSettings(settings) {
  localStorage.setItem('careguide.frontend.settings', JSON.stringify(settings))
}

export function backendMeta(_type, settings) {
  return {
    ...BACKEND,
    baseUrl: normalizeBaseUrl(settings.endpoint || BACKEND.baseUrl)
  }
}

export async function requestHealth(type, settings) {
  return requestJson(backendMeta(type, settings).baseUrl, '/health')
}

export async function requestMonitor(type, settings) {
  const raw = await requestJson(backendMeta(type, settings).baseUrl, '/monitor')
  return normalizeMonitorResponse(raw)
}

export async function requestSkills(type, settings) {
  const raw = await requestJson(backendMeta(type, settings).baseUrl, '/skills')
  return normalizeSkillsResponse(raw)
}

export async function reloadSkills(type, settings) {
  const raw = await requestJson(backendMeta(type, settings).baseUrl, '/skills/reload', { method: 'POST' })
  return normalizeSkillsResponse(raw)
}

export async function requestKnowledgeStats(type, settings) {
  return requestJson(backendMeta(type, settings).baseUrl, '/knowledge/stats')
}

export async function requestSearch(type, settings, query, topK = 5, rewrite = false, rerank = false) {
  const params = new URLSearchParams({ query, top_k: String(topK) })
  params.set('rewrite', String(rewrite))
  params.set('rerank', String(rerank))
  const raw = await requestJson(backendMeta(type, settings).baseUrl, `/search?${params}`, { method: 'POST', timeoutMs: 120000 })
  return normalizeSearchResponse(raw)
}

export async function requestChat(type, settings, message, signal, workflowEvent, onEvent) {
  const payload = buildChatPayload(type, settings, message)
  if (workflowEvent) payload.workflow_event = workflowEvent
  const raw = await postChatStream(`${backendMeta(type, settings).baseUrl}/chat/stream`, payload, { signal, onEvent })
  return normalizeChatResponse(type, raw)
}

export async function requestChatJson(type, settings, message, signal, workflowEvent) {
  const meta = backendMeta(type, settings)
  const payload = buildChatPayload(type, settings, message)
  if (workflowEvent) payload.workflow_event = workflowEvent
  const raw = await requestJson(meta.baseUrl, '/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal,
    timeoutMs: 90000
  })
  return normalizeChatResponse(type, raw)
}

export async function requestToolTrace(type, settings, requestId) {
  if (!requestId) return null
  const raw = await requestJson(backendMeta(type, settings).baseUrl, `/trace/tool/${encodeURIComponent(requestId)}`)
  return normalizeToolTraceResponse(raw)
}

export function requestAction(type, settings, actionId, operation) {
  return requestJson(backendMeta(type, settings).baseUrl,
    `/actions/${encodeURIComponent(actionId)}/${operation}`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: settings.userId, conv_id: settings.conversationId })
    })
}

export function requestAppointments(type, settings) {
  const user = encodeURIComponent(settings.userId || 'demo-user')
  return requestJson(backendMeta(type, settings).baseUrl, `/appointments?user_id=${user}`)
}

export async function addKnowledge(type, settings, bundle) {
  return requestJson(backendMeta(type, settings).baseUrl, '/knowledge/add', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(bundle)
  })
}

export async function uploadKnowledge(type, settings, file) {
  const bundle = JSON.parse(await file.text())
  return requestJson(backendMeta(type, settings).baseUrl, '/knowledge/upload', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(bundle)
  })
}

export function requestOperations(type, settings, path, body) {
  return requestJson(backendMeta(type, settings).baseUrl, path, body === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), timeoutMs: 600000
  })
}

function buildChatPayload(type, settings, message) {
  return {
    message,
    user_id: settings.userId || 'anonymous',
    conv_id: settings.conversationId || undefined
  }
}

function normalizeChatResponse(type, raw) {
  const payload = asObject(raw)
  return {
    backend: type,
    conversationId: payload.conversation_id || payload.conversationId || payload.conv_id || '',
    requestId: payload.request_id || payload.requestId || '',
    response: payload.response || '',
    intent: payload.intent || 'other',
    intentGroup: payload.intent_group || payload.intentGroup || 'other',
    agentType: payload.agent_type || payload.agentType || '',
    agentTypes: asArray(payload.agent_types ?? payload.agentTypes),
    primaryAgent: payload.primary_agent || payload.primaryAgent || '',
    supportingAgents: asArray(payload.supporting_agents ?? payload.supportingAgents),
    toolsUsed: asArray(payload.tools_used ?? payload.toolsUsed),
    routingReason: payload.routing_reason || payload.routingReason || '',
    routingConfidence: Number(payload.routing_confidence ?? payload.routingConfidence ?? 0),
    entities: asObject(payload.entities),
    intentConfidence: Number(payload.intent_confidence ?? payload.intentConfidence ?? 0),
    intentSourceScores: asObject(payload.intent_source_scores ?? payload.intentSourceScores),
    escalated: toBoolean(payload.escalated),
    latencyMs: Number(payload.latency_ms ?? payload.latencyMs ?? 0),
    knowledgeUsed: toBoolean(payload.knowledge_used ?? payload.knowledgeUsed),
    retrievalBackend: payload.retrieval_backend || payload.retrievalBackend || '',
    simulation: toBoolean(payload.simulation),
    actionRequests: asArray(payload.action_requests ?? payload.actionRequests),
    workflow: payload.workflow || null,
    budget: payload.budget || null,
    appointment: asArray(payload.appointment),
    evidence: asArray(payload.evidence),
    verified: payload.verified,
    grounded: payload.grounded,
    raw: payload
  }
}

function normalizeToolTraceResponse(raw) {
  const payload = asObject(raw)
  const trace = asObject(payload.trace)
  const execution = asObject(trace.execution)
  const final = asObject(trace.final)
  const toolCalls = firstArray(
    trace.tool_calls,
    trace.toolCalls,
    execution.tool_calls,
    execution.toolCalls
  ) || []
  const explicitToolsUsed = [
    trace.tools_used,
    trace.toolsUsed,
    execution.tools_used,
    execution.toolsUsed
  ].find(Array.isArray)
  const waveToolsUsed = asArray(execution.execution_waves ?? execution.executionWaves)
    .flatMap((wave) => asArray(wave?.agents))
    .flatMap((agent) => asArray(agent?.tools_used ?? agent?.toolsUsed))
  const toolsUsed = uniqueStrings([
    ...(explicitToolsUsed || []),
    ...waveToolsUsed,
    ...toolCalls.map((call) => call?.tool_name || call?.toolName)
  ])
  return {
    requestId: payload.request_id || payload.requestId || '',
    found: toBoolean(payload.found),
    trace: {
      ...trace,
      conversationId: trace.conversation_id || trace.conversationId || '',
      workflowStatus: final.workflow_status || final.workflowStatus || trace.status || '',
      toolsUsed,
      toolCalls
    },
    raw: payload
  }
}

function normalizeMonitorResponse(raw) {
  const payload = asObject(raw)
  const activeAlerts = firstArray(payload.active_alerts, payload.activeAlerts).map((alert) => {
    const item = asObject(alert)
    const message = item.message || item.detail || item.title || ''
    return { ...item, message }
  })
  return {
    ...payload,
    agentStats: asObject(payload.agent_stats ?? payload.agentStats),
    toolStats: asObject(payload.tool_stats ?? payload.toolStats),
    activeAlerts,
    suggestions: asArray(payload.suggestions)
  }
}

function normalizeSkillsResponse(raw) {
  const payload = asObject(raw)
  const skills = asArray(payload.skills).map((skill) => {
    const item = asObject(skill)
    const skillId = item.skill_id || item.skillId || item.id || item.name || ''
    const version = item.version || ''
    const eligibleAgents = asArray(item.eligible_agents ?? item.eligibleAgents ?? item.agents)
    return {
      ...item,
      skillId,
      version,
      versionLabel: version ? `v${version}` : '未版本化',
      name: item.name || skillId,
      description: item.description || '',
      eligibleAgents,
      applicableTaskTypes: asArray(item.applicable_task_types ?? item.applicableTaskTypes),
      applicableActions: asArray(item.applicable_actions ?? item.applicableActions),
      scopeLabel: eligibleAgents.length ? `适用 Agent：${eligibleAgents.join('、')}` : '业务规范能力'
    }
  })
  return {
    ...payload,
    rootDir: payload.root_dir || payload.rootDir || '',
    count: Number(payload.count ?? skills.length),
    skills,
    errors: asArray(payload.errors),
    selection: payload.selection || '',
    externalLlmCalls: Number(payload.external_llm_calls ?? payload.externalLlmCalls ?? 0)
  }
}

function normalizeSearchResponse(raw) {
  const payload = asObject(raw)
  const results = asArray(payload.results ?? payload.data).map((result, index) => {
    const item = asObject(result)
    return {
      ...item,
      id: item.id || item.chunk_id || item.chunkId || item.document_id || item.documentId || `result-${index}`,
      documentId: item.document_id || item.documentId || '',
      chunkId: item.chunk_id || item.chunkId || ''
    }
  })
  return { ...payload, results }
}

function asObject(value) {
  return value && typeof value === 'object' && !Array.isArray(value) ? value : {}
}

function asArray(value) {
  return Array.isArray(value) ? value : []
}

function firstArray(...values) {
  return values.find(Array.isArray) || []
}

function uniqueStrings(values) {
  return [...new Set(values.filter((value) => typeof value === 'string' && value))]
}

function toBoolean(value) {
  if (typeof value === 'string') return value.toLowerCase() === 'true'
  return Boolean(value)
}

async function requestJson(baseUrl, path, options = {}) {
  const url = `${normalizeBaseUrl(baseUrl)}${path}`
  const { timeoutMs = 30000, signal: callerSignal, ...fetchOptions } = options
  const controller = new AbortController()
  const timeout = window.setTimeout(() => controller.abort('timeout'), timeoutMs)
  const abortFromCaller = () => controller.abort('cancelled')
  callerSignal?.addEventListener('abort', abortFromCaller, { once: true })

  let response
  let text
  try {
    response = await fetch(url, { ...fetchOptions, signal: controller.signal })
    text = await response.text()
  } catch (error) {
    if (controller.signal.aborted) {
      throw new Error(callerSignal?.aborted ? '请求已取消' : '请求超时，请检查模型服务后重试')
    }
    throw error
  } finally {
    window.clearTimeout(timeout)
    callerSignal?.removeEventListener('abort', abortFromCaller)
  }

  let data = null
  try {
    data = text ? JSON.parse(text) : null
  } catch {
    data = text
  }
  if (!response.ok) {
    const detail = typeof data === 'string' ? data : JSON.stringify(data)
    throw new Error(`${response.status} ${response.statusText}: ${detail}`)
  }
  return data
}

function normalizeBaseUrl(value) {
  return String(value || '').replace(/\/+$/, '')
}

function readSettings() {
  try {
    const saved = localStorage.getItem('careguide.frontend.settings')
      || '{}'
    return JSON.parse(saved)
  } catch {
    return {}
  }
}
