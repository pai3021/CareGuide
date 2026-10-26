// Real Chrome + built Vue, with a scripted HTTP API. No browser package needed.
import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import fs from 'node:fs/promises'
import http from 'node:http'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const browserPath = process.env.CAREGUIDE_TEST_BROWSER || 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'
const profile = await fs.mkdtemp(path.join(os.tmpdir(), 'careguide-browser-'))
const requests = []
const failures = []
let releaseFirstStream
let version = 1, revision = 1
const tasks = [
  { id: 'drug', revision: 1, label: '查询药品说明书', status: 'waiting_input', missing_fields: ['specification'], options: [] },
  { id: 'slots', revision: 1, label: '查询医生号源', status: 'waiting_input', missing_fields: ['practitioner_id'],
    options: [{ field: 'practitioner_id', values: [{ id: 'PRA-1', name: '合成医生', department: '消化内科' }] }] }
]
let actions = []
const workflow = () => ({ id: 'browser-workflow', version, revision, tasks: structuredClone(tasks) })
const response = () => ({ conversation_id: 'browser-conversation', request_id: 'browser-request-'+revision,
  answer: { body: '## 当前查询\n\n可继续当前任务。[S1]',
    sources: [{ id: 'S1', task_id: 'drug', title: '浏览器测试资料', excerpt: '仅用于验证引用定位。' }] },
  response: '可继续当前任务', workflow_status: actions.some(item => item.status === 'pending_confirmation') ? 'pending_confirmation' : 'waiting_user',
  workflow: workflow(), action_requests: actions, evidence: [], agent_types: ['appointment', 'medication'], simulation: true })
const server = http.createServer(async (req, res) => {
  try {
    const url = new URL(req.url, 'http://localhost')
    if (url.pathname === '/api/test/release-stream') {
      releaseFirstStream?.(); res.writeHead(204); res.end(); return
    }
    if (url.pathname === '/favicon.ico') { res.writeHead(204); res.end(); return }
    if (url.pathname.startsWith('/api/')) {
      const chunks = []
      for await (const chunk of req) chunks.push(chunk)
      const body = chunks.length ? JSON.parse(Buffer.concat(chunks).toString()) : null
      requests.push({ method: req.method, path: url.pathname, body })
      let result = {}
      if (url.pathname === '/api/chat/stream') {
        const event = body.workflow_event
        if (event?.kind === 'answer') {
          assert.equal(event.workflow_id, 'browser-workflow')
          assert.equal(event.plan_version, 1)
          assert.equal(event.task_revision, 1)
          assert.ok(event.event_id)
          const task = tasks.find(item => item.id === event.task_id)
          if (task.id === 'drug') assert.deepEqual(event.values, { specification: '0.2g' })
          if (task.id === 'slots') assert.deepEqual(event.values, { practitioner_id: 'PRA-1' })
          if (task.id === 'booking') assert.deepEqual(event.values, { slot_id: 'SLOT-1' })
          task.status = 'succeeded'; task.missing_fields = []
          if (task.id === 'slots') tasks.push({ id: 'booking', revision: 1, label: '准备模拟预约', status: 'waiting_input',
            missing_fields: ['slot_id'], options: [{ field: 'slot_id', values: [{ id: 'SLOT-1', start: '2030-01-02T10:00:00+08:00' }] }] })
          if (task.id === 'booking') {
            task.status = 'waiting_confirmation'; task.action_id = 'ACT-1'
            actions = [{ id: 'ACT-1', type: 'appointment_create', status: 'pending_confirmation', arguments: { slot_id: 'SLOT-1' },
              workflow_binding: { workflow_id: 'browser-workflow', plan_version: 1, task_id: 'booking', task_revision: 1 } }]
          }
        } else if (event?.kind === 'change') {
          assert.equal(event.plan_version, 1)
          assert.match(body.message, /改为预约 SLOT-2/)
          version = 2
          const task = tasks.find(item => item.id === 'booking')
          task.revision = 2; task.action_id = 'ACT-2'
          actions = [{ id: 'ACT-2', type: 'appointment_create', status: 'pending_confirmation', arguments: { slot_id: 'SLOT-2' },
            workflow_binding: { workflow_id: 'browser-workflow', plan_version: 2, task_id: 'booking', task_revision: 2 } }]
        }
        revision += 1
        result = response()
        const emit = (id, event, data) => res.write(`id: ${id}\nevent: ${event}\ndata: ${JSON.stringify(data)}\n\n`)
        res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no' })
        emit(0, 'session', { conversation_id: 'browser-conversation', stream_id: 'browser-stream' })
        emit(1, 'stage', { stage: 'planning', message: '正在生成任务计划' })
        emit(2, 'workflow', { workflow: { ...workflow(), tasks: tasks.map(task => ({ ...task, status: 'running' })) } })
        emit(3, 'delta', { answer_id: 'browser-answer', offset: 0, text: '正在为您', provisional: true })
        emit(4, 'delta', { answer_id: 'browser-answer', offset: 4, text: '整理查询结果', provisional: true })
        // Keep the final response gated so the browser must render actual
        // streamed progress before it can see the interactive result cards.
        if (!body.workflow_event) await new Promise(resolve => { releaseFirstStream = resolve })
        if (res.destroyed) return
        emit(5, 'result', result)
        res.end(); return
      } else if (url.pathname === '/api/actions/ACT-2/confirm') {
        assert.equal(body.conv_id, 'browser-conversation')
        actions[0].status = 'completed'; tasks.find(item => item.id === 'booking').status = 'succeeded'
        revision += 1
        result = { ...actions[0], workflow: workflow() }
      } else if (url.pathname === '/api/workflows/latest') result = response()
      else if (url.pathname === '/api/auth/me') result = { mode: 'local_demo', user_id: null, role: 'demo' }
      else if (url.pathname === '/api/health') result = { status: 'ok' }
      else if (url.pathname === '/api/appointments') result = { appointments: [] }
      else if (url.pathname === '/api/knowledge/stats') result = { total_chunks: 1 }
      res.writeHead(200, { 'Content-Type': 'application/json' }); res.end(JSON.stringify(result)); return
    }
    const target = path.join(root, 'dist', url.pathname === '/' ? 'index.html' : url.pathname)
    assert.ok(target.startsWith(path.join(root, 'dist')))
    const content = await fs.readFile(target)
    res.writeHead(200, { 'Content-Type': ({ '.js': 'application/javascript', '.css': 'text/css', '.html': 'text/html' })[path.extname(target)] || 'application/octet-stream' })
    res.end(content)
  } catch (error) {
    failures.push(error.stack)
    res.writeHead(500); res.end('scripted test request failed')
  }
})
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve))
const origin = `http://127.0.0.1:${server.address().port}`
const browser = spawn(browserPath, ['--headless=new', '--no-first-run', '--no-default-browser-check',
  '--disable-background-networking', '--disable-component-update', '--disable-sync', '--disable-extensions',
  '--remote-debugging-port=0', '--remote-allow-origins=*', `--user-data-dir=${profile}`, 'about:blank'], { stdio: 'ignore' })
browser.on('error', error => failures.push(error.message))
const pause = ms => new Promise(resolve => setTimeout(resolve, ms))
let socket, session
const pending = new Map()
let sequence = 0
async function command(method, params = {}, target = session) {
  const id = ++sequence
  const result = new Promise((resolve, reject) => pending.set(id, { resolve, reject }))
  socket.send(JSON.stringify({ id, method, params, ...(target ? { sessionId: target } : {}) }))
  return await result
}
async function evaluate(expression) {
  const value = await command('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true })
  assert.equal(value.exceptionDetails, undefined, JSON.stringify(value.exceptionDetails))
  return value.result.value
}
async function until(expression) {
  const end = Date.now()+8000
  while (Date.now() < end) {
    if (await evaluate(expression)) return
    await pause(50)
  }
  throw new Error('UI condition not satisfied: '+expression)
}
async function send(text) {
  await until(`document.querySelector('.composer button[type="submit"]') !== null`)
  await evaluate(`(() => { const input=document.querySelector('.composer textarea'); input.value=${JSON.stringify(text)}; input.dispatchEvent(new Event('input',{bubbles:true})); document.querySelector('.composer').requestSubmit(); })()`)
}
try {
  let ports
  for (let attempt = 0; attempt < 100; attempt++) {
    try { ports = (await fs.readFile(path.join(profile, 'DevToolsActivePort'), 'utf8')).trim().split('\n'); break } catch { await pause(100) }
  }
  assert.ok(ports, 'Chrome did not start with a fresh isolated profile')
  socket = new WebSocket(`ws://127.0.0.1:${ports[0]}${ports[1]}`)
  await new Promise((resolve, reject) => { socket.addEventListener('open', resolve, { once: true }); socket.addEventListener('error', reject, { once: true }) })
  socket.addEventListener('message', event => {
    const message = JSON.parse(event.data)
    const item = pending.get(message.id)
    if (item) { pending.delete(message.id); message.error ? item.reject(new Error(JSON.stringify(message.error))) : item.resolve(message.result) }
  })
  const target = await command('Target.createTarget', { url: 'about:blank' }, null)
  session = (await command('Target.attachToTarget', { targetId: target.targetId, flatten: true }, null)).sessionId
  await command('Page.enable')
  await command('Page.navigate', { url: origin })
  await until(`document.querySelector('.composer textarea') !== null`)
  await send('查医生，同时查询药品说明书')
  await until(`document.querySelector('[data-testid="stream-progress"]')?.textContent.includes('执行中')`)
  await until(`document.body.textContent.includes('正在为您整理查询结果')`)
  assert.equal(await evaluate(`document.querySelector('.action-card form input') === null`), true)
  await evaluate(`fetch('/api/test/release-stream').then(response => response.status)`)
  await until(`document.querySelector('.action-card form input:not(:disabled)') !== null`)
  assert.equal(await evaluate(`document.querySelector('.answer-markdown h2')?.textContent`), '当前查询')
  await evaluate(`document.querySelector('.source-reference').click()`)
  await until(`document.querySelector('.answer-sources article.selected')?.textContent.includes('引用定位')`)
  await evaluate(`(() => { const input=document.querySelector('.action-card form input:not(:disabled)'); input.value='0.2g'; input.dispatchEvent(new Event('input',{bubbles:true})); input.closest('form').requestSubmit(); })()`)
  await until(`document.querySelector('.action-card form select:not(:disabled)') !== null`)
  for (const value of ['PRA-1', 'SLOT-1']) {
    await until(`document.querySelector('.action-card form select:not(:disabled) option[value="${value}"]') !== null`)
    await evaluate(`(() => { const select=document.querySelector('.action-card form select:not(:disabled)'); select.value='${value}'; select.dispatchEvent(new Event('change',{bubbles:true})); select.closest('form').requestSubmit(); })()`)
  }
  await until(`Array.from(document.querySelectorAll('.action-buttons button')).some(button => !button.disabled)`)
  await send('改为预约 SLOT-2')
  await until(`document.body.textContent.includes('SLOT-2') && document.querySelectorAll('.action-buttons').length === 2`)
  assert.equal(await evaluate(`document.querySelectorAll('.action-buttons')[0].querySelector('button').disabled`), true)
  await evaluate(`document.querySelectorAll('.action-buttons')[0].querySelector('button').click()`)
  assert.equal(requests.filter(item => item.path.includes('/confirm')).length, 0)
  await evaluate(`document.querySelectorAll('.action-buttons')[1].querySelector('button').click()`)
  await until(`document.body.textContent.includes('模拟操作已完成')`)
  await command('Page.reload', { ignoreCache: true })
  await until(`Array.from(document.querySelectorAll('button')).some(button => button.textContent.includes('恢复会话任务') && !button.disabled)`)
  await evaluate(`Array.from(document.querySelectorAll('button')).find(button => button.textContent.includes('恢复会话任务')).click()`)
  await until(`document.querySelectorAll('.action-card').length >= 3`)
  assert.equal(await evaluate(`Array.from(document.querySelectorAll('.action-buttons button')).some(button => !button.disabled)`), false)
  assert.deepEqual(requests.filter(item => item.path.includes('/confirm')).map(item => item.path), ['/api/actions/ACT-2/confirm'])
  const beforeStop = requests.filter(item => item.path === '/api/chat/stream').length
  await send('查医生，测试停止接收')
  await until(`document.querySelector('[data-testid="stream-progress"]') !== null`)
  await evaluate(`Array.from(document.querySelectorAll('.composer button')).find(button => button.textContent.includes('停止接收')).click()`)
  await until(`document.body.textContent.includes('已停止接收')`)
  await evaluate(`fetch('/api/test/release-stream').then(response => response.status)`)
  assert.equal(requests.filter(item => item.path === '/api/chat/stream').length, beforeStop + 1)
  assert.equal(await evaluate(`JSON.parse(localStorage.getItem('careguide.frontend.settings')).conversationId`), 'browser-conversation')
  assert.equal(failures.length, 0, failures.join('\n'))
  console.log(JSON.stringify({ kind: 'real_chrome_built_vue_scripted_api', checks: [
    'task_missing_field_submission', 'doctor_selection', 'slot_selection', 'versioned_change',
    'old_confirmation_disabled', 'one_current_confirmation', 'reload_and_restore', 'sse_progress_before_final', 'answer_deltas_before_final', 'markdown_and_citation_navigation',
    'sse_stop_keeps_session_without_resubmit'],
    browser: (await command('Browser.getVersion', {}, null)).product,
    chat_events: requests.filter(item => item.path === '/api/chat/stream').map(item => item.body.workflow_event || { kind: 'initial' }),
    confirmation_requests: 1, paid_calls: 0 }, null, 2))
} finally {
  if (socket?.readyState === WebSocket.OPEN) { try { await Promise.race([command('Browser.close', {}, null), pause(1000)]) } catch {} }
  browser.kill()
  socket?.close()
  server.closeAllConnections()
  await new Promise(resolve => server.close(resolve))
}
