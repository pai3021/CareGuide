<template>
  <main :class="['app-shell', `app-shell-${activeView}`]">
    <header class="topbar">
      <a class="brand" href="#" aria-label="CareGuide 首页" @click.prevent="activeView = 'chat'">
        <span class="brand-mark">C</span>
        <span class="brand-name">CareGuide</span>
      </a>

      <nav class="view-nav" aria-label="页面导航">
        <button :class="{ active: activeView === 'chat' }" @click="activeView = 'chat'">对话</button>
        <button :class="{ active: activeView === 'knowledge' }" @click="activeView = 'knowledge'">知识库</button>
        <button :class="{ active: activeView === 'operations' }" @click="activeView = 'operations'; loadOperations()">咨询记录与评测</button>
        <button :class="{ active: activeView === 'appointments' }" @click="activeView = 'appointments'; loadAppointments()">模拟预约</button>
      </nav>

      <div class="topbar-tools">
        <span class="environment-pill">
          <i :class="healthOk ? 'online' : 'offline'"></i>
          {{ healthOk ? '服务在线' : '服务未连接' }}
        </span>
        <a class="docs-link" :href="docsUrl" target="_blank" rel="noreferrer">API 文档</a>
        <button
          class="theme-toggle"
          type="button"
          :aria-label="settings.theme === 'dark' ? '切换到浅色主题' : '切换到深色主题'"
          :title="settings.theme === 'dark' ? '切换到浅色主题' : '切换到深色主题'"
          @click="toggleTheme"
        >
          <span aria-hidden="true">{{ settings.theme === 'dark' ? '☀' : '☾' }}</span>
          <span>{{ settings.theme === 'dark' ? '浅色' : '深色' }}</span>
        </button>
        <button class="avatar-button" title="当前用户">{{ userInitial }}</button>
      </div>
    </header>

    <div v-if="toast" class="toast" role="status">{{ toast }}</div>

    <section v-if="activeView === 'chat'" class="page page-chat">
      <div class="page-heading">
        <div class="heading-copy">
          <span class="kicker">个人学习项目</span>
          <h1>CareGuide 对话</h1>
          <p>查询药品信息、科室资料或挂号流程，查看来源和执行记录。</p>
        </div>
        <div class="heading-actions">
          <span class="session-label">{{ settings.conversationId || '新会话' }}</span>
          <button class="quiet-button" title="清空当前消息并创建新会话" @click="clearConversation">新会话</button>
        </div>
      </div>

      <div class="chat-layout">
        <section class="chat-stage">
          <div class="stage-bar">
            <div class="stage-context">
              <span class="context-dot"></span>
              <span>{{ currentBackend.baseUrl }}</span>
            </div>
            <span>{{ messages.length }} 条消息</span>
          </div>

          <div class="messages" ref="messageList">
            <article v-for="item in messages" :key="item.id" :class="['message', item.role]">
              <div class="message-meta">
                <span>{{ item.role === 'user' ? '你' : currentBackend.label + ' Agent' }}</span>
                <small v-if="item.meta">{{ item.meta }}</small>
              </div>
              <p>{{ item.content }}</p>
              <div v-if="item.simulation" class="simulation-banner">CareGuide 合成医院 · 模拟数据 · 不可用于真实就诊</div>
              <div v-if="item.evidence?.length" class="evidence-list">
                <div v-for="source in item.evidence" :key="source.record_id" class="evidence-source">
                  <a :href="source.citation?.url || source.source_url" target="_blank" rel="noreferrer">
                    {{ source.title || source.record_id }}
                  </a>
                  <small v-if="source.citation?.review_status === 'open_dataset_unreviewed'">
                    开放研究数据 · 未经临床审核
                  </small>
                </div>
              </div>
              <div v-for="action in item.actionRequests" :key="action.id" class="action-card">
                <strong>{{ actionLabel(action.type) }}</strong>
                <span>{{ action.status === 'pending_confirmation' && !currentAction(activeWorkflow, action) ? '已失效，请查看当前操作' : action.status }}</span>
                <pre>{{ formatJson(action.arguments) }}</pre>
                <p>{{ action.message }}</p>
                <div v-if="action.status === 'pending_confirmation'" class="action-buttons">
                  <button :disabled="busy || !currentAction(activeWorkflow, action)" @click="confirmPendingAction(action)">确认执行模拟操作</button>
                  <button :disabled="busy || !currentAction(activeWorkflow, action)" class="quiet-button" @click="rejectPendingAction(action)">拒绝</button>
                </div>
                <button v-if="['reconciling', 'executing'].includes(action.status)" :disabled="busy" @click="confirmPendingAction(action)">核对操作结果</button>
              </div>
              <div v-for="task in item.workflow?.tasks || []" :key="`${task.id}-${task.revision}`" class="action-card">
                <strong>{{ task.label }}</strong><span>{{ currentTask(activeWorkflow, item.workflow, task) ? task.status : '已有更新，请查看最新任务' }}</span>
                <small v-for="scope in task.current_scope || []" :key="scope.label">{{ scope.label }}：{{ scope.value }}</small>
                <form v-if="task.status === 'waiting_input'" @submit.prevent="answerTask(item.workflow, task)">
                  <label v-for="field in task.missing_fields" :key="field">
                    {{ workflowFieldLabel(field) }}
                    <select v-if="task.options.some(option => option.field === field)"
                      v-model="workflowAnswers[`${item.workflow.id}:${task.id}:${task.revision}:${field}`]"
                      :disabled="busy || !currentTask(activeWorkflow, item.workflow, task)">
                      <option disabled value="">请选择</option>
                      <option v-for="option in task.options.find(option => option.field === field).values" :key="option.id" :value="option.id">
                        {{ option.name || option.start || option.id }} · {{ option.department || option.id }}
                      </option>
                    </select>
                    <select v-else-if="field === 'time_period'"
                      v-model="workflowAnswers[`${item.workflow.id}:${task.id}:${task.revision}:${field}`]"
                      :disabled="busy || !currentTask(activeWorkflow, item.workflow, task)">
                      <option disabled value="">请选择时段</option>
                      <option value="morning">上午</option><option value="afternoon">下午</option><option value="evening">晚上</option>
                    </select>
                    <input v-else v-model="workflowAnswers[`${item.workflow.id}:${task.id}:${task.revision}:${field}`]"
                      :disabled="busy || !currentTask(activeWorkflow, item.workflow, task)" />
                  </label>
                  <button :disabled="busy || !currentTask(activeWorkflow, item.workflow, task)">提交并继续</button>
                </form>
                <button v-if="['waiting_input', 'waiting_confirmation', 'waiting_dependencies', 'running', 'ready', 'blocked'].includes(task.status)"
                  :disabled="busy || !currentTask(activeWorkflow, item.workflow, task)" @click="cancelTask(item.workflow, task)">取消该任务</button>
              </div>
              <div v-if="item.trace" class="message-trace">
                <div class="trace-head">
                  <span>工具调用</span>
                  <small v-if="item.trace.requestId">#{{ item.trace.requestId }}</small>
                </div>
                <div v-if="item.trace.toolCalls?.length" class="trace-calls">
                  <details v-for="(call, index) in item.trace.toolCalls" :key="`${item.id}-${index}`" open>
                    <summary>
                      <strong>{{ call.tool_name || 'unknown_tool' }}</strong>
                      <span>{{ call.success ? '成功' : '失败' }}</span>
                    </summary>
                    <pre>{{ formatJson(call.input || {}) }}</pre>
                  </details>
                </div>
                <div v-else class="trace-empty-block">
                  <p>本次请求已生成 trace，但没有可展示的工具输入。</p>
                  <p v-if="item.trace.toolsUsed?.length" class="trace-note">已调用：{{ item.trace.toolsUsed.join(' · ') }}</p>
                </div>
              </div>
            </article>

            <div v-if="messages.length === 0" class="empty-state">
              <div class="empty-symbol">✦</div>
              <h2>从一个就医咨询开始</h2>
              <p>选择一个资料查询，或输入症状咨询、药品及挂号问题。</p>
              <div class="starter-prompts">
                <button @click="usePrompt('头痛就诊前要记录哪些信息？')">就诊准备</button>
                <button @click="usePrompt('请查询精氨酸布洛芬片0.2g的说明书注意事项')">药品说明书</button>
                <button @click="usePrompt('查询CareGuide合成医院消化内科的模拟医生')">模拟医生</button>
                <button @click="usePrompt('查询CareGuide合成医院消化内科2030年1月2日的模拟号源')">模拟号源</button>
              </div>
            </div>
          </div>

          <form class="composer" @submit.prevent="sendMessage">
            <textarea
              v-model="draft"
              rows="3"
              placeholder="输入消息..."
              @keydown.meta.enter.prevent="sendMessage"
              @keydown.ctrl.enter.prevent="sendMessage"
            ></textarea>
            <div class="composer-bottom">
              <span>{{ busy ? '正在请求模型服务；可随时取消。' : '⌘ / Ctrl + Enter 发送' }}</span>
              <button v-if="busy" class="quiet-button" type="button" @click="cancelRequest">取消请求</button>
              <button v-else type="submit" :disabled="!draft.trim()">发送</button>
              <button class="quiet-button" type="button" :disabled="!settings.conversationId" @click="restoreWorkflow">恢复会话任务</button>
            </div>
          </form>
        </section>

        <aside class="chat-sidebar" ref="sidebarRef">
          <div class="chat-sidebar-scroll">
            <section class="side-card session-card">
              <div class="card-heading">
                <div>
                  <span class="kicker">Session</span>
                  <h2>会话信息</h2>
                </div>
                <span class="status-copy muted">{{ settings.conversationId ? '已启用' : '新会话' }}</span>
              </div>
              <div class="session-grid">
                <div>
                  <span>会话 ID</span>
                  <strong>{{ settings.conversationId || '自动生成' }}</strong>
                </div>
                <div>
                  <span>用户 ID</span>
                  <strong>{{ settings.userId || 'anonymous' }}</strong>
                </div>
              </div>
            </section>

            <section class="side-card connection-card">
              <div class="card-heading">
                <div>
                  <span class="kicker">Connection</span>
                  <h2>连接配置</h2>
                </div>
                <span class="status-copy" :class="healthOk ? 'success' : 'muted'">{{ healthLabel }}</span>
              </div>

              <label>
                <span>用户 ID</span>
                <input v-model="settings.userId" @change="persist" placeholder="u001" />
              </label>
              <label>
                <span>会话 ID</span>
                <input v-model="settings.conversationId" @change="persist" placeholder="自动生成" />
              </label>
              <div class="side-actions">
                <button @click="checkHealth">检查连接</button>
                <button class="quiet-button" @click="refreshConsole">刷新</button>
              </div>
            </section>

            <section class="side-card trace-card">
              <div class="card-heading">
                <div>
                  <span class="kicker">Last trace</span>
                  <h2>最近一次请求</h2>
                </div>
                <span class="trace-status" :class="lastResponse ? 'has-data' : ''"></span>
              </div>

              <div v-if="lastResponse" class="trace-body">
                <div class="latency">
                  <span>响应耗时</span>
                  <strong>{{ lastResponse.latencyMs || '-' }}<small> ms</small></strong>
                </div>
                <dl class="detail-list">
                  <div><dt>主 Agent</dt><dd>{{ lastResponse.primaryAgent || lastResponse.agentType || '-' }}</dd></div>
                  <div><dt>意图</dt><dd>{{ lastResponse.intent || '-' }}</dd></div>
                  <div><dt>置信度</dt><dd>{{ formatPercent(lastResponse.routingConfidence) }}</dd></div>
                  <div><dt>知识库</dt><dd :class="lastResponse.knowledgeUsed ? 'success' : 'muted'">{{ lastResponse.knowledgeUsed ? '已使用' : '未使用' }}</dd></div>
                  <div><dt>检索后端</dt><dd>{{ lastResponse.retrievalBackend || '-' }}</dd></div>
                  <div><dt>模拟操作</dt><dd :class="lastResponse.simulation ? 'danger' : 'muted'">{{ lastResponse.simulation ? '是' : '否' }}</dd></div>
                  <div><dt>转人工</dt><dd :class="lastResponse.escalated ? 'danger' : 'muted'">{{ lastResponse.escalated ? '是' : '否' }}</dd></div>
                </dl>
                <p v-if="lastResponse.routingReason" class="routing-reason">{{ lastResponse.routingReason }}</p>
                <div v-if="lastTrace?.trace" class="trace-call-list">
                  <div class="trace-call-title">工具调用</div>
                  <div v-for="(call, index) in lastTrace.trace.toolCalls" :key="`${call.tool_use_id || index}`" class="trace-call-item">
                    <div class="trace-call-meta">
                      <strong>{{ call.tool_name || 'unknown_tool' }}</strong>
                      <span>{{ call.latency_ms || 0 }} ms</span>
                    </div>
                    <pre>{{ formatJson(call.input || {}) }}</pre>
                  </div>
                  <div v-if="!lastTrace.trace.toolCalls?.length" class="trace-empty-block">
                    <p>这次 trace 没有记录到工具输入。</p>
                    <p v-if="lastTrace.trace.toolsUsed?.length" class="trace-note">已调用：{{ lastTrace.trace.toolsUsed.join(' · ') }}</p>
                  </div>
                </div>
              </div>
              <p v-else class="side-empty">发送消息后，这里会显示 Agent 路由、意图和耗时。</p>
            </section>

            <section class="side-card monitor-card">
              <div class="card-heading">
                <div>
                  <span class="kicker">Runtime</span>
                  <h2>运行状态</h2>
                </div>
                <button class="link-button" @click="loadMonitor">刷新</button>
              </div>
              <div class="mini-stats">
                <div><strong>{{ totalRequests }}</strong><span>请求</span></div>
                <div><strong>{{ agentCount }}</strong><span>Agent</span></div>
                <div><strong>{{ activeAlerts.length }}</strong><span>告警</span></div>
              </div>
              <div v-if="activeAlerts.length" class="alert-note">{{ activeAlerts[0].message || '检测到运行告警' }}</div>
              <p v-else class="healthy-note">当前没有活跃告警。</p>
            </section>
          </div>
        </aside>
      </div>
    </section>

    <section v-else-if="activeView === 'knowledge'" class="page page-knowledge">
      <div class="page-heading">
        <div class="heading-copy">
          <span class="kicker">Knowledge operations</span>
          <h1>知识库</h1>
          <p>检索带来源的医疗资料；新增资料需先核验来源与适用范围。</p>
        </div>
        <div class="count-display"><strong>{{ knowledgeCount }}</strong><span>chunks</span></div>
      </div>

      <div class="knowledge-layout">
        <section class="workspace-card search-workspace">
          <div class="card-heading">
            <div><span class="kicker">Retrieval</span><h2>检索知识</h2></div>
            <code>POST /search</code>
          </div>
          <div class="search-line">
            <input v-model="searchQuery" placeholder="例如：头痛就诊前记录哪些信息" @keydown.enter="searchKnowledge" />
            <button @click="searchKnowledge" :disabled="busy || !searchQuery.trim()">搜索</button>
          </div>
          <label><input type="checkbox" v-model="rewriteSearch" /> 查询改写</label>
          <label><input type="checkbox" v-model="rerankSearch" /> 模型重排</label>
          <div v-if="searchResults.length" class="result-list">
            <article v-for="(item, index) in searchResults" :key="item.id || item.title || index" class="result-item">
              <span class="result-number">{{ String(index + 1).padStart(2, '0') }}</span>
              <div>
                <div class="result-title"><strong>{{ item.title || '未命名文档' }}</strong><small>score {{ item.score ?? '-' }}</small></div>
                <p>{{ item.content }}</p>
              </div>
            </article>
          </div>
          <div v-else class="workspace-empty">输入就医问题开始搜索。</div>
        </section>

        <section class="workspace-card import-workspace">
          <div class="card-heading"><h2>资料来源</h2></div>
          <p>查询结果保留来源和适用范围。药品需确认名称、剂型与规格；挂号指南需确认医院和院区。</p>
          <p>资料未覆盖时会说明无法核实。当前不提供诊断、处方、实时号源或实际预约。</p>
          <label>上传带来源的 JSON 资料包<input type="file" accept="application/json,.json" @change="importBundle" /></label>
          <button @click="loadOperations">刷新待审核资料</button>
          <article v-for="item in knowledgeSubmissions" :key="item.id">
            <p>{{ item.id }} · {{ item.status }} · {{ item.bundle.records.length }} 条资料</p>
            <button v-if="item.status === 'pending'" @click="reviewBundle(item, true)">审核通过</button>
            <button v-if="item.status === 'pending'" @click="reviewBundle(item, false)">驳回</button>
            <button v-if="item.status === 'reviewed'" @click="publishBundle(item)">发布</button>
          </article>
          <input v-model="reviewNote" placeholder="填写来源核对与适用范围审核意见" />
        </section>
      </div>

      <section class="workspace-card skills-workspace">
        <div class="card-heading">
          <div><span class="kicker">Loaded skills</span><h2>已加载能力</h2></div>
          <button class="link-button" @click="reloadSkillSet">重新加载</button>
        </div>
        <div class="skill-table">
          <div v-for="skill in skillsData.skills" :key="skill.skillId" class="skill-item">
            <span class="skill-dot"></span><strong>{{ skill.name || skill.skillId }}</strong><span>{{ skill.description || skill.scopeLabel }}</span><small>{{ skill.versionLabel }}</small>
          </div>
          <div v-if="!skillsData.skills.length" class="workspace-empty">暂无已加载 Skill。</div>
        </div>
      </section>
    </section>

    <section v-else-if="activeView === 'operations'" class="page">
      <div class="page-heading"><h1>咨询记录与开发评测</h1><button @click="loadOperations">刷新</button></div>
      <section class="workspace-card">
        <h2>我的咨询交接记录</h2>
        <p>对话中输入“转人工协助”即可创建本地记录。该队列尚未连接医院或医护。</p>
        <article v-for="item in handoffs" :key="item.id">
          <p><strong>{{ item.id }}</strong> · {{ item.status }} · {{ item.urgency }}</p>
          <p>{{ item.summary }}</p>
          <button v-if="item.status === 'open'" @click="updateHandoff(item, 'in_progress')">开始处理</button>
          <button v-if="item.status === 'in_progress'" @click="updateHandoff(item, 'resolved')">标记处理完成</button>
          <button v-if="['open', 'in_progress'].includes(item.status)" @click="updateHandoff(item, 'cancelled')">取消记录</button>
        </article>
        <p v-if="!handoffs.length">暂无咨询记录。</p>
      </section>
      <section class="workspace-card">
        <h2>开发集评测</h2>
        <p>运行意图识别评测；可附带对话和证据一致性 Judge。不会运行保留集或自动更新基线。</p>
        <label>开发样本数<input type="number" min="1" max="240" v-model.number="evalLimit" /></label>
        <label><input type="checkbox" v-model="evalJudge" /> 启用 Judge</label>
        <textarea v-model="evalDialog" placeholder="可选：填写一段测试对话，每行一轮" />
        <button :disabled="evalBusy" @click="runDevEvaluation">{{ evalBusy ? '评测中…' : '运行开发评测' }}</button>
        <pre v-if="evalReport">{{ formatJson(evalReport) }}</pre>
        <button v-if="evalReport?.id" @click="promoteBaseline">将当前报告设为开发基线</button>
      </section>
    </section>

    <section v-else-if="activeView === 'appointments'" class="page">
      <div class="page-heading">
        <div class="heading-copy"><span class="kicker">Synthetic FHIR</span><h1>模拟预约</h1>
          <p>这里只显示 CareGuide 合成医院数据，不连接中大医院或任何真实医院。</p></div>
        <button @click="loadAppointments">刷新</button>
      </div>
      <div class="simulation-banner">所有医生、号源和预约均为模拟数据，不可用于真实就诊。</div>
      <section class="workspace-card">
        <article v-for="item in appointments" :key="item.id" class="appointment-item">
          <div class="appointment-title">
            <strong>{{ item.department || '科室待确认' }}</strong>
            <span>{{ formatAppointmentStatus(item.status) }}</span>
          </div>
          <p>{{ item.practitioner_name || '医生待确认' }} · {{ formatAppointmentTime(item.start) }}</p>
          <small>预约编号 {{ item.id }}<template v-if="appointmentSlotId(item)"> · 号源 {{ appointmentSlotId(item) }}</template></small>
        </article>
        <p v-if="!appointments.length">当前没有模拟预约。</p>
      </section>
    </section>
  </main>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import {
  backendMeta,
  createInitialSettings,
  reloadSkills,
  requestChat,
  requestHealth,
  requestKnowledgeStats,
  requestMonitor,
  requestSearch,
  requestToolTrace,
  requestSkills,
  requestAction,
  requestAppointments,
  saveSettings
} from './lib/backends'
import { requestOperations, uploadKnowledge } from './lib/backends'
import { acceptWorkflow, currentTask, currentAction, answerEvent } from './lib/workflow'

const settings = reactive(createInitialSettings())
const activeView = ref('chat')
const messages = ref([])
const draft = ref('')
const busy = ref(false)
const activeWorkflow = ref(null)
const workflowAnswers = reactive({})

function workflowFieldLabel(field) {
  return ({ specification: '药品规格', drug_name: '药品全名', dosage_form: '剂型',
    time_period: '时段',
    practitioner_id: '医生', slot_id: '号源', date: '日期', department: '科室',
    appointment_id: '预约编号', hospital_id: '医院', campus: '院区' })[field] || field
}

async function answerTask(workflow, task) {
  if (!currentTask(activeWorkflow.value, workflow, task)) return
  try {
    const values = Object.fromEntries(task.missing_fields.map(field => [field,
      workflowAnswers[`${workflow.id}:${task.id}:${task.revision}:${field}`]]).filter(([, value]) => value?.trim()))
    await sendMessage(answerEvent(workflow, task, values, createMessageId()))
  } catch (error) { toast.value = error.message }
}

async function cancelTask(workflow, task) {
  if (!currentTask(activeWorkflow.value, workflow, task)) return
  await sendMessage({ kind: 'cancel', workflow_id: workflow.id, plan_version: workflow.version,
    task_id: task.id, task_revision: task.revision })
}

async function restoreWorkflow() {
  if (!settings.conversationId) return
  try {
    const query = new URLSearchParams({ user_id: settings.userId, conv_id: settings.conversationId })
    const result = await requestOperations(settings.backend, settings, `/workflows/latest?${query}`)
    if (!result.workflow) { showToast('当前会话没有已保存任务'); return }
    activeWorkflow.value = acceptWorkflow(activeWorkflow.value, result.workflow)
    messages.value.push({ id: createMessageId(), role: 'assistant', content: result.response,
      workflow: result.workflow, actionRequests: result.action_requests, evidence: result.evidence })
  } catch (error) { toast.value = error.message }
}
const healthOk = ref(false)
const healthLabel = ref('未检查')
const statusText = ref('')
const knowledgeCount = ref('-')
const searchQuery = ref('头痛就诊前记录哪些信息')
const searchResults = ref([])
const rewriteSearch = ref(false)
const rerankSearch = ref(false)
const knowledgeSubmissions = ref([])
const handoffs = ref([])
const appointments = ref([])
const reviewNote = ref('')
const evalLimit = ref(12)
const evalJudge = ref(false)
const evalDialog = ref('')
const evalBusy = ref(false)
const evalReport = ref(null)

async function loadOperations() {
  try {
    const results = await Promise.all([
      requestOperations(settings.backend, settings, `/handoffs?user_id=${encodeURIComponent(settings.userId || 'demo-user')}`),
      requestOperations(settings.backend, settings, '/knowledge/submissions')
    ])
    handoffs.value = results[0].records
    knowledgeSubmissions.value = results[1].submissions
  } catch (error) { toast.value = error.message }
}
async function loadAppointments() {
  try {
    const data = await requestAppointments(settings.backend, settings)
    appointments.value = data.appointments || []
  } catch (error) { toast.value = error.message }
}

async function confirmPendingAction(action) {
  try {
    const result = await requestAction(settings.backend, settings, action.id, 'confirm')
    action.status = result.status
    action.result = result.result
    if (result.workflow) activeWorkflow.value = acceptWorkflow(activeWorkflow.value, result.workflow)
    showToast(result.status === 'completed' ? '模拟操作已完成' : `操作状态：${result.status}`)
    await loadAppointments()
  } catch (error) { toast.value = error.message }
}

async function rejectPendingAction(action) {
  try {
    const result = await requestAction(settings.backend, settings, action.id, 'reject')
    action.status = result.status
    if (result.workflow) activeWorkflow.value = acceptWorkflow(activeWorkflow.value, result.workflow)
    showToast('已拒绝模拟操作')
  } catch (error) { toast.value = error.message }
}

function actionLabel(type) {
  return ({ appointment_create: '创建模拟预约', appointment_reschedule: '改期模拟预约',
    appointment_cancel: '取消模拟预约' })[type] || type
}
function formatAppointmentStatus(status) {
  return ({ booked: '已预约', cancelled: '已取消', pending: '待处理', proposed: '待确认' })[status] || status || '未知状态'
}
function formatAppointmentTime(value) {
  if (!value) return '时间待确认'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return new Intl.DateTimeFormat('zh-CN', {
    year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', hour12: false
  }).format(date)
}
function appointmentSlotId(item) {
  return item.slot_id || item.slot_ids?.[0] || ''
}
async function updateHandoff(item, status) {
  try {
    await requestOperations(settings.backend, settings, `/handoffs/${item.id}/status`, { user_id: settings.userId || 'demo-user', status })
    await loadOperations()
  } catch (error) { toast.value = error.message }
}
async function importBundle(event) {
  try {
    const file = event.target.files[0]
    if (file) await uploadKnowledge(settings.backend, settings, file)
    await loadOperations()
  } catch (error) { toast.value = error.message }
}
async function reviewBundle(item, approve) {
  try {
    await requestOperations(settings.backend, settings, `/knowledge/${item.id}/review`, {
      reviewer: settings.userId || 'local-developer', approve, note: reviewNote.value })
    await loadOperations()
  } catch (error) { toast.value = error.message }
}
async function publishBundle(item) {
  try {
    await requestOperations(settings.backend, settings, `/knowledge/${item.id}/publish`, {})
    await loadOperations(); await loadStats()
  } catch (error) { toast.value = error.message }
}
async function runDevEvaluation() {
  evalBusy.value = true
  try {
    evalReport.value = await requestOperations(settings.backend, settings, '/eval/run', {
      limit: evalLimit.value, judge: evalJudge.value,
      dialogs: evalDialog.value.trim() ? [evalDialog.value.trim().split('\n')] : [] })
  } catch (error) { toast.value = error.message }
  finally { evalBusy.value = false }
}
async function promoteBaseline() {
  try {
    await requestOperations(settings.backend, settings, `/eval/baseline/${evalReport.value.id}`, {})
    toast.value = '开发基线已更新'
  } catch (error) { toast.value = error.message }
}
const messageList = ref(null)
const sidebarRef = ref(null)
const monitorData = ref(emptyMonitorData())
const skillsData = ref({ count: 0, skills: [], errors: [] })
const lastResponse = ref(null)
const lastTrace = ref(null)
const toast = ref('')
let toastTimer
let messageSequence = 0
let sidebarObserver
let chatAbortController

const currentBackend = computed(() => backendMeta(settings.backend, settings))
const docsUrl = computed(() => `${currentBackend.value.baseUrl}/docs`)
const userInitial = computed(() => (settings.userId || 'U').slice(0, 1).toUpperCase())
const activeAlerts = computed(() => monitorData.value.activeAlerts)
const agentCount = computed(() => Object.keys(monitorData.value.agentStats).length)
const totalRequests = computed(() => Object.values(monitorData.value.agentStats).reduce((sum, item) => sum + Number(item.total || 0), 0))

watch(() => settings.conversationId, persist)
watch(() => settings.theme, (theme) => {
  applyTheme(theme)
  persist()
}, { immediate: true })
onMounted(() => {
  refreshConsole()
  updateSidebarHeight()
  if (typeof ResizeObserver !== 'undefined') {
    sidebarObserver = new ResizeObserver(updateSidebarHeight)
    if (sidebarRef.value) sidebarObserver.observe(sidebarRef.value)
  }
  window.addEventListener('resize', updateSidebarHeight)
})

onBeforeUnmount(() => {
  chatAbortController?.abort()
  sidebarObserver?.disconnect?.()
  window.removeEventListener('resize', updateSidebarHeight)
})

function persist() { saveSettings(settings) }

function emptyMonitorData() {
  return { agentStats: {}, toolStats: {}, activeAlerts: [], suggestions: [] }
}

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme
}

function toggleTheme() {
  settings.theme = settings.theme === 'dark' ? 'light' : 'dark'
}

function updateSidebarHeight() {
  const sidebar = sidebarRef.value
  if (!sidebar) return
  const rect = sidebar.getBoundingClientRect()
  const height = Math.max(320, Math.floor(rect.height))
  sidebar.style.setProperty('--sidebar-height', `${height}px`)
}

async function refreshConsole() {
  await Promise.allSettled([checkHealth(), loadStats(), loadMonitor(), loadSkills()])
}

async function checkHealth() {
  try {
    const data = await requestHealth(settings.backend, settings)
    healthOk.value = data.status === 'ok'
    healthLabel.value = data.status || 'ok'
    statusText.value = JSON.stringify(data, null, 2)
  } catch (error) {
    healthOk.value = false
    healthLabel.value = '不可用'
    statusText.value = error.message

  }
}

async function loadStats() {
  try {
    const data = await requestKnowledgeStats(settings.backend, settings)
    knowledgeCount.value = data.total_chunks ?? data.totalChunks ?? '-'
  } catch {
    knowledgeCount.value = '-'
  }
}

async function loadMonitor() {
  try {
    monitorData.value = await requestMonitor(settings.backend, settings)
  } catch {
    monitorData.value = emptyMonitorData()
  }
}

async function loadSkills() {
  try {
    skillsData.value = await requestSkills(settings.backend, settings)
  } catch {
    skillsData.value = { count: 0, skills: [], errors: [] }
  }
}

async function reloadSkillSet() {
  busy.value = true
  try {
    skillsData.value = await reloadSkills(settings.backend, settings)
    showToast('Skills 已重新加载')
  } catch (error) {
    statusText.value = error.message
    showToast('Skills 加载失败')
  } finally { busy.value = false }
}

async function sendMessage(workflowEvent) {
  if (!['answer', 'run', 'change', 'cancel'].includes(workflowEvent?.kind)) workflowEvent = undefined
  const content = workflowEvent?.kind === 'cancel' ? '取消当前任务' : workflowEvent ? Object.entries(workflowEvent.values || {}).map(([field, value]) =>
    `${workflowFieldLabel(field)}：${value}`).join('，') || '继续任务' : draft.value.trim()
  if (!content || busy.value) return
  if (!workflowEvent && activeWorkflow.value && /改成|改为|改查|换成|换号源|更换|取消|不用|另外|增加|再加|不查/.test(content)) {
    workflowEvent = { kind: 'change', workflow_id: activeWorkflow.value.id,
      plan_version: activeWorkflow.value.version }
  }
  messages.value.push({ id: createMessageId(), role: 'user', content })
  draft.value = ''
  busy.value = true
  chatAbortController = new AbortController()
  try {
    const response = await requestChat(settings.backend, settings, content, chatAbortController.signal, workflowEvent)
    if (response.conversationId && !settings.conversationId) {
      settings.conversationId = response.conversationId
      persist()
    }
    lastResponse.value = response
    activeWorkflow.value = acceptWorkflow(activeWorkflow.value, response.workflow)
    lastTrace.value = await loadToolTrace(response.requestId)
    const meta = [response.intent, response.primaryAgent || response.agentType, response.knowledgeUsed ? 'RAG' : '', response.escalated ? '转人工' : ''].filter(Boolean).join(' · ')
    messages.value.push({ id: createMessageId(), role: 'assistant', content: response.response, meta,
      trace: lastTrace.value?.trace || null, simulation: response.simulation,
      actionRequests: response.actionRequests, evidence: response.evidence, workflow: response.workflow })
    await loadMonitor()
  } catch (error) {
    messages.value.push({ id: createMessageId(), role: 'assistant', content: error.message, meta: '请求失败' })
  } finally {
    chatAbortController = null
    busy.value = false
    await nextTick()
    messageList.value?.scrollTo({ top: messageList.value.scrollHeight, behavior: 'smooth' })
  }
}

function cancelRequest() {
  chatAbortController?.abort()
}

function usePrompt(prompt) { draft.value = prompt }

function clearConversation() {
  activeWorkflow.value = null
  messages.value = []
  lastResponse.value = null
  lastTrace.value = null
  settings.conversationId = ''
  persist()
}

async function searchKnowledge() {
  busy.value = true
  try {
    const data = await requestSearch(settings.backend, settings, searchQuery.value, 5, rewriteSearch.value, rerankSearch.value)
    searchResults.value = data.results || []
    showToast(`检索完成，返回 ${searchResults.value.length} 条结果`)
  } catch (error) {
    statusText.value = error.message
    showToast('检索失败，请检查连接')
  } finally { busy.value = false }
}

async function loadToolTrace(requestId) {
  try {
    return await requestToolTrace(settings.backend, settings, requestId)
  } catch {
    return null
  }
}

function formatPercent(value) {
  const number = Number(value || 0)
  return `${(number <= 1 ? number * 100 : number).toFixed(1)}%`
}

function formatJson(value) {
  try {
    return JSON.stringify(value ?? {}, null, 2)
  } catch {
    return String(value ?? '')
  }
}

function createMessageId() {
  messageSequence += 1
  return `message-${Date.now()}-${messageSequence}`
}

function showToast(message) {
  toast.value = message
  clearTimeout(toastTimer)
  toastTimer = setTimeout(() => { toast.value = '' }, 2600)
}
</script>
