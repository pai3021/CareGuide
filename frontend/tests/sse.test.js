import test from 'node:test'
import assert from 'node:assert/strict'
import { createSSEParser, readChatEvents, postChatStream, applyAnswerEvent } from '../src/lib/sse.js'

const frame = (id, event, data) => `id: ${id}\nevent: ${event}\ndata: ${JSON.stringify(data)}\n\n`
const final = { response: '已核验：药品资料', conversation_id: 'c' }

test('answer deltas append Unicode text; reset retracts drafts and final replaces them', async () => {
  const message = { content: '' }
  const text = frame(0, 'delta', { answer_id: 'a', offset: 0, text: '查询😀' }) +
    frame(1, 'delta', { answer_id: 'a', offset: 3, text: '结果' }) +
    frame(2, 'answer_reset', { answer_id: 'a' }) + frame(3, 'result', final)
  const seen = []
  const result = await readChatEvents(response(text, 1), ({ event, data }) => {
    applyAnswerEvent(message, event, data)
    seen.push(message.content)
    if (event === 'result') message.content = data.response
  })
  assert.deepEqual(seen.slice(0, 3), ['查询😀', '查询😀结果', ''])
  assert.equal(message.content, result.response)
})

test('answer gaps, foreign streams and malformed chunks cannot corrupt the draft', () => {
  const message = { content: '' }
  assert.throws(() => applyAnswerEvent(message, 'delta', { answer_id: 'a', offset: 2, text: 'x' }), /不完整/)
  applyAnswerEvent(message, 'delta', { answer_id: 'a', offset: 0, text: '医生' })
  assert.throws(() => applyAnswerEvent(message, 'delta', { answer_id: 'a', offset: 1, text: 'x' }), /顺序/)
  assert.throws(() => applyAnswerEvent(message, 'delta', { answer_id: 'b', offset: 2, text: 'x' }), /顺序/)
  applyAnswerEvent(message, 'answer_reset', { answer_id: 'b' })
  assert.equal(message.content, '医生')
  assert.throws(() => applyAnswerEvent(message, 'delta', { text: 1 }), /格式/)
})
function response(text, byteSize = 3) {
  const bytes = new TextEncoder().encode(text)
  return new Response(new ReadableStream({ start(controller) {
    for (let i = 0; i < bytes.length; i += byteSize) controller.enqueue(bytes.slice(i, i + byteSize))
    controller.close()
  } }), { headers: { 'Content-Type': 'text/event-stream; charset=utf-8' } })
}

test('SSE handles split CRLF, comments, multiline data and UTF-8 byte boundaries', async () => {
  const events = []
  const text = ': heartbeat\r\nid: 0\r\nevent: session\r\ndata: {"conversation_id":\r\ndata: "中文会话"}\r\n\r\n' + frame(1, 'result', final)
  assert.deepEqual(await readChatEvents(response(text, 1), event => events.push(event)), final)
  assert.equal(events[0].data.conversation_id, '中文会话')
  assert.deepEqual(events.map(e => e.event), ['session', 'result'])
})

test('duplicate and out-of-order sequence ids cannot restore older progress', async () => {
  const events = []
  const text = frame(0, 'session', {}) + frame(2, 'stage', { stage: 'executing' }) +
    frame(1, 'stage', { stage: 'planning' }) + frame(2, 'stage', { stage: 'planning' }) + frame(3, 'result', final)
  await readChatEvents(response(text), event => events.push(event))
  assert.deepEqual(events.map(e => e.id), ['0', '2', '3'])
})

test('midstream errors expose status and never masquerade as completed results', async () => {
  await assert.rejects(readChatEvents(response(frame(0, 'error', { status: 409, message: '任务版本已更新' }))),
    error => error.status === 409 && error.message === '任务版本已更新')
})

test('unexpected EOF, truncated event and invalid final data require explicit recovery', async () => {
  await assert.rejects(readChatEvents(response(frame(0, 'stage', {}))), /连接已中断/)
  await assert.rejects(readChatEvents(response(frame(0, 'result', final).slice(0, -1))), /连接已中断/)
  await assert.rejects(readChatEvents(response(frame(0, 'result', { status: 'completed' }))), /最终响应格式无效/)
})

test('abort cancels a pending reader and does not wait for another server chunk', async () => {
  let cancelled = false
  const controller = new AbortController()
  const pending = new Response(new ReadableStream({ cancel() { cancelled = true } }),
    { headers: { 'Content-Type': 'text/event-stream' } })
  const reading = readChatEvents(pending, null, controller.signal)
  controller.abort()
  await assert.rejects(reading, /停止接收/)
  assert.equal(cancelled, true)
})

test('non-SSE response is rejected without resubmitting the POST', async () => {
  const original = globalThis.fetch
  let calls = 0
  globalThis.fetch = async () => { calls++; return new Response('{}', { headers: { 'Content-Type': 'application/json' } }) }
  try {
    await assert.rejects(postChatStream('/chat/stream', { message: '查询' }), /不会自动重发/)
    assert.equal(calls, 1)
  } finally { globalThis.fetch = original }
})

test('HTTP validation error and event size limits are explicit', async () => {
  await assert.rejects(readChatEvents(new Response('{"detail":"缺少任务版本"}', { status: 422 })), /422 缺少任务版本/)
  const parser = createSSEParser(() => {})
  assert.throws(() => parser.feed('data: ' + 'x'.repeat(4 * 1024 * 1024 + 1)), /过大/)
})
