// POST SSE: one execution per request. Never reconnect or resubmit implicitly.
export function applyAnswerEvent(message, event, data) {
  if (event === 'answer_reset') {
    if (message.answerId === data.answer_id) {
      message.content = ''; message.answerId = null; message.answerOffset = 0
    }
    return
  }
  if (event !== 'delta') return
  if (typeof data.answer_id !== 'string' || !data.answer_id || typeof data.text !== 'string' ||
      !Number.isSafeInteger(data.offset) || data.offset < 0) throw new Error('回答增量格式无效')
  if (!message.answerId) {
    if (data.offset !== 0) throw new Error('回答增量不完整，请恢复会话任务')
    message.answerId = data.answer_id; message.answerOffset = 0; message.content = ''
  }
  if (message.answerId !== data.answer_id || data.offset !== message.answerOffset) {
    throw new Error('回答增量顺序无效，请恢复会话任务')
  }
  message.content += data.text
  message.answerOffset += Array.from(data.text).length
  if (message.answerOffset > 12000) throw new Error('回答过长，请恢复会话任务')
}

export function createSSEParser(onEvent) {
  let buffer = '', event = '', data = [], id = '', size = 0
  const line = (value) => {
    if (!value) {
      if (data.length) onEvent({ event: event || 'message', id, data: data.join('\n') })
      event = ''; data = []; size = 0
      return
    }
    if (value.startsWith(':')) return
    const colon = value.indexOf(':')
    const key = colon < 0 ? value : value.slice(0, colon)
    let valuePart = colon < 0 ? '' : value.slice(colon + 1)
    if (valuePart.startsWith(' ')) valuePart = valuePart.slice(1)
    if (key === 'event') event = valuePart
    if (key === 'id' && !valuePart.includes('\0')) id = valuePart
    if (key === 'data') {
      data.push(valuePart); size += valuePart.length
      if (size > 4 * 1024 * 1024) throw new Error('流式响应过大，请恢复会话任务')
    }
  }
  return {
    feed(text, end = false) {
      buffer += text
      let index
      while ((index = buffer.search(/[\r\n]/)) >= 0) {
        if (buffer[index] === '\r' && index === buffer.length - 1 && !end) break
        const width = buffer[index] === '\r' && buffer[index + 1] === '\n' ? 2 : 1
        const value = buffer.slice(0, index)
        buffer = buffer.slice(index + width)
        line(value)
      }
      if (buffer.length > 4 * 1024 * 1024) throw new Error('流式响应过大，请恢复会话任务')
      // An unterminated event is deliberately not dispatched at EOF.
    }
  }
}

export async function readChatEvents(response, onEvent, signal) {
  if (!response.ok) {
    const body = await response.text()
    let detail = body
    try { detail = JSON.parse(body).detail || body } catch {}
    throw new Error(`${response.status} ${detail}`)
  }
  if (!response.headers.get('content-type')?.toLowerCase().includes('text/event-stream') || !response.body) {
    throw new Error('服务未返回 SSE，请检查后端版本；请求不会自动重发')
  }
  const reader = response.body.getReader()
  const decoder = new TextDecoder('utf-8', { fatal: true })
  let result, lastId = -1
  const interrupted = () => new Error('已停止接收，服务端任务可能仍在执行；请恢复会话任务查看结果')
  const abort = () => { reader.cancel().catch(() => {}) }
  signal?.addEventListener('abort', abort, { once: true })
  const parser = createSSEParser(({ event, id, data }) => {
    if (result !== undefined) return
    if (signal?.aborted) throw interrupted()
    if (!/^\d+$/.test(id)) throw new Error('流式事件缺少有效序号')
    const sequence = Number(id)
    if (!Number.isSafeInteger(sequence)) throw new Error('流式事件序号无效')
    if (sequence <= lastId) return
    lastId = sequence
    const payload = JSON.parse(data)
    if (event === 'error') {
      const error = new Error(payload.message || '对话服务暂不可用')
      error.status = payload.status
      throw error
    }
    if (event === 'result') {
      if (!payload || typeof payload.response !== 'string' || typeof payload.conversation_id !== 'string') {
        throw new Error('最终响应格式无效，请恢复会话任务查看结果')
      }
      result = payload
    }
    onEvent?.({ event, id, data: payload })
  })
  try {
    if (signal?.aborted) throw interrupted()
    while (result === undefined) {
      const { value, done } = await reader.read()
      if (signal?.aborted) throw interrupted()
      parser.feed(done ? decoder.decode() : decoder.decode(value, { stream: true }), done)
      if (done) break
    }
    if (result === undefined) throw new Error('连接已中断，请恢复会话任务查看结果；请求不会自动重发')
    return result
  } finally {
    signal?.removeEventListener('abort', abort)
    await reader.cancel().catch(() => {})
    reader.releaseLock()
  }
}

export async function postChatStream(url, payload, { signal, onEvent, timeoutMs = 90000 } = {}) {
  const controller = new AbortController()
  const abort = () => controller.abort()
  signal?.addEventListener('abort', abort, { once: true })
  if (signal?.aborted) abort()
  const timer = setTimeout(abort, timeoutMs)
  try {
    const response = await fetch(url, { method: 'POST', signal: controller.signal,
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' }, body: JSON.stringify(payload) })
    return await readChatEvents(response, onEvent, controller.signal)
  } catch (error) {
    if (controller.signal.aborted) throw new Error(signal?.aborted
      ? '已停止接收，服务端任务可能仍在执行；请恢复会话任务查看结果'
      : '接收超时，请恢复会话任务查看结果；请求不会自动重发')
    throw error
  } finally {
    clearTimeout(timer)
    signal?.removeEventListener('abort', abort)
  }
}
