import test from 'node:test'
import assert from 'node:assert/strict'
import { renderMarkdown, safeSourceUrl } from '../src/lib/markdown.js'
import { accessToken, setAccessToken, authHeaders } from '../src/lib/auth.js'

test('markdown renders headings, lists, tables and known citation anchors', () => {
  const html = renderMarkdown('## 查询结果\n\n- **医生甲** [S1]\n\n|科室|医生|\n|---|---|\n|内科|甲|', [{ id: 'S1' }])
  assert.match(html, /<h2>查询结果<\/h2>/)
  assert.match(html, /<strong>医生甲<\/strong>/)
  assert.match(html, /data-source-id="S1"/)
  assert.match(html, /<table>/)
  assert.doesNotMatch(renderMarkdown('未知 [S999]', [{ id: 'S1' }]), /data-source-id/)
})

test('raw HTML, script URLs, remote images and malicious source IDs remain inert', () => {
  const html = renderMarkdown('<img src=x onerror=alert(1)>\n\n[点我](javascript:alert(1))\n\n![追踪](https://tracker.example/a.png)\n\n[S1]', [{ id: 'S1" onclick="alert(1)' }])
  assert.doesNotMatch(html, /<img|<script|href="javascript|onclick=|data-source-id/)
  assert.match(html, /&lt;img/)
  assert.equal(safeSourceUrl('javascript:alert(1)'), undefined)
  assert.equal(safeSourceUrl('https://example.com'), 'https://example.com')
})

test('external links are protected and access tokens stay scoped to a base endpoint', () => {
  assert.match(renderMarkdown('[来源](https://example.com)'), /rel="noopener noreferrer"/)
  setAccessToken('/api', 'one-token')
  assert.deepEqual(authHeaders('/api'), { Authorization: 'Bearer one-token' })
  assert.deepEqual(authHeaders('https://other.example/api'), {})
  assert.equal(accessToken('/api/'), 'one-token')
  setAccessToken('/api', '')
  assert.deepEqual(authHeaders('/api'), {})
})
