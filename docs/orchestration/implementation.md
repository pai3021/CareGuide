# 编排架构与运行说明

CareGuide 使用 LangGraph 管理任务规划、执行、检查、局部重规划、等待和结束。任务 DAG 的依赖、参数绑定和结果失效规则由业务代码实现。验收场景见 [验收矩阵](acceptance.md)，结果摘要见 [verification.json](verification.json)。

## 请求与执行路径

`POST /chat` → `get_chat_runtime()` → `OrchestrationRuntime.process()`。

急症边界先行。完整原文和保守的简单查询白名单决定分流，分类器单标签不能屏蔽复合请求。简单完整请求直接路由；简单缺参、时效性号源及预约准备可进入无需 Planner 的单任务持久流程。不确定请求交给 Planner。

- `planner.py`：读取原文、候选意图、已有任务、参数 schema、动作目录及消息接收时间，生成计划或变更提案；最多一次格式修正，关闭 SDK 隐式重试。
- `contracts.py`、`bindings.py`：校验角色权限、参数、原文片段、输入类型、环和规模；每版最多 12 项。候选集合不等于用户选择，查询所有医生号源需要对应范围的授权。
- `runtime.py`：构建 LangGraph 外层流程，处理急症优先、请求分流、中断与恢复。
- `scheduler.py`、`worker.py`：最多 4 个任务并发、叶调用最多 4 个；下游只等待自身依赖。复用已有 Agent、工具和医院服务，并隔离任务参数、结果和证据。
- `store.py`：局部变更只失效受影响任务及其下游，保留无关成功结果和有效确认。任务版本和派发标识拒绝受影响的迟到结果。相对日期绑定原消息的上海日历。
- `errors.py`、`replies.py`：分类错误、有限修正并输出部分结果；模型失败不能抹掉成功资料，最终回复无需额外润色模型调用。

等待中的旧任务不占用整个会话。明确的新查询独立处理，旧任务及确认状态保留；明确的补参、选择、变更或取消继续对应任务。含糊的输入会提示用户说明归属，不自动当作参数或确认。带指代的短句需要明确对象。恢复历史任务时使用其 workflow_id 和版本，不能依赖“最新任务”接口返回任意旧流程。

## 确认与持久恢复

预约沿用 AppointmentService 和 `/actions/{id}/confirm`。ActionStore 原子核对用户、会话、参数及任务版本，领取后才写。已领取、已完成或待核对的操作不能被改计划抹掉；改期和取消需要新动作与确认。未知结果只读核对操作标识、患者和业务事实，重复确认不重复写；HTTP 200 不代替事务及预约结果核验。

业务任务、结果、动作和预算共用 `CAREGUIDE_BUSINESS_DB`。图检查点使用同目录独立 SQLite 文件，例如 `careguide.sqlite3` 与 `careguide.checkpoints.sqlite3`。两库不是同一事务，恢复以业务状态、派发标识及真实写事实为准。检查点与业务库分开，避免异步检查点提交和同步业务事务竞争同一文件的写锁。当前验证范围为单机 SQLite。

| 对象 | 有效期与恢复规则 |
| --- | --- |
| Redis 会话 | 24 小时；基础 session 模式为进程内记忆 |
| 工作流 | 30 分钟；过期后保留历史，重新发起请求 |
| 补参等待 | 最长 300 秒，受工作流与上游结果期限约束 |
| 医生结果／号源 | 300 秒／60 秒；恢复与派发前重验，过期刷新并重新选择 |
| action 确认 | 最长 15 分钟，同时受任务、工作流和输入有效期约束 |
| 长期偏好 | 字段 180 天，读取时逻辑清理并增加版本 |

等待通过 interrupt 结束当前 HTTP 请求，持久化缺参、引用、结果和预算。恢复核对身份、会话、计划／任务版本、候选成员和有效期。过期不删除业务事实，也不授权复用旧确认。

API 的 answer 事件绑定 workflow_id、plan_version、task_id、task_revision、values、event_id；change 提交新要求，cancel 停止未执行分支，run 恢复。GET `/workflows/latest?user_id=…&conv_id=…` 获取最新任务，GET `/workflows/{id}?user_id=…&conv_id=…` 获取指定任务。身份不符返回 404、版本冲突 409、过期 410、字段错误 422。

## 预算和长期记忆

默认累计额度为模型 20、工具 30、重试 3、重规划 3 次，累计调用执行 120 秒、单请求墙钟 45 秒，80% 起收尾。并发先原子预留，失败不退次数；取消或超时无法证明远端已停止时保守结算预留时间。累计执行是并发调用耗时之和，用户等待不计。恢复、改版、确认及 direct 回退沿用旧账本；缺失账本拒绝执行。usage 只记录提供方实际返回值，不推算缺失 token 或货币费用。

仅 `CAREGUIDE_MEMORY_MODE=redis_chroma` 启用持久记忆。用户持续偏好经廉价筛选后才调用提取模型。白名单为时段、医院、科室、院区、语言，每项绑定用户消息 ID、原话、时间、范围和版本。

SQLite 为权威画像，Chroma 为投影。来源与类型校验后去重、合并、更正或删除，CAS 检测并发冲突；投影失败保留旧权威画像。当前仅自动复用验证后的长期时段到只读号源查询，本次明确条件优先，其他偏好作为背景。记忆读取最多 2 秒、写入最多 8 秒，失败不抹掉业务回复。逐用户／逐日独立预算为模型 10、存储 10 次、累计 60 秒，每次最长 3 秒。

## 运行与复现

使用 Python 3.12，按仓库根目录 [README](../../README.md) 安装依赖和启动服务。配置模板为 `.env.example`；`CAREGUIDE_ORCHESTRATION_MODE=langgraph` 默认启用。复合规划需要兼容 Chat Completions 的模型配置，无配置时明确返回规划不可用。基础简单查询和以下离线示例无需模型密钥。

LangGraph、checkpoint 和 aiosqlite 版本固定在 `requirements.txt`。标准 `Dockerfile` 支持独立构建；`Dockerfile.dev-update` 仅适用于已有兼容镜像的增量更新。模型使用只读挂载，业务库和检查点需持久化到业务卷。

```bash
python -m pytest -q -p no:cacheprovider
node --test frontend/tests/workflow.test.js
npm --prefix frontend run build
node frontend/tests/browser-workflow.mjs
python scripts/reproduce_orchestration.py --scenario dependency
python scripts/reproduce_orchestration.py --scenario replan
python scripts/reproduce_orchestration.py --scenario partial
python scripts/reproduce_orchestration.py --scenario memory
python scripts/reproduce_orchestration.py --scenario wait --phase start --state-dir .study/repro-wait-new
# 上一进程退出后，在有效期内运行；新 start 使用新目录。
python scripts/reproduce_orchestration.py --scenario wait --phase resume --state-dir .study/repro-wait-new
```

示例注入可控模型响应，执行真实 Planner、worker 与状态路径。测试进程不自动读取 `.env`；运行全量测试时使用独立业务数据库并避免继承部署环境的 CAREGUIDE_* 设置。浏览器测试需已安装 Chrome，可通过 CAREGUIDE_TEST_BROWSER 指定浏览器路径。

`scripts/verify_existing_services.py` 默认只读。`--write-isolated --state-dir <新目录>` 创建独立 UUID 的合成资源验证 FHIR、Redis、Chroma 链路；QA 资源按脚本规则保留，测试向量只验证存储，不代表检索质量。

索引复用脚本 `scripts/reuse_verified_index.py` 校验可信原语料导出的逐文档哈希、模型、维度、来源与有限向量，目标存在时拒绝覆盖。先写临时 collection，完整校验后发布；不会通过修改旧索引指纹绕过校验。它用于已验证真实 embedding 的复用，不能替代新查询的检索质量评测。

## 验证范围与回退

自动化验收覆盖权限、参数绑定、依赖并发、部分结果、局部变更、预算、中断恢复、重复确认和长期偏好。具体结果与时间见 verification.json。浏览器使用真实 Chrome 与构建后的 Vue，API 响应为脚本；生产 API 测试使用实际 FastAPI 工厂与可控模型。历史服务集成验证与本次代码回归分开记录。

真实模型规划质量、线上费用和临床有效性尚未评测。user_id 为演示身份，尚无生产认证；未验证多主分布式部署。医院限合成服务，资料未经临床审核。过期历史目前逻辑隔离，不自动物理删除。

需要切回直接路由时设置 `CAREGUIDE_ORCHESTRATION_MODE=direct` 并重建应用配置。确认仍检查已保存任务绑定和预算，图状态保留；重新启用后仅恢复有效流程。direct 模式不提供图卡片恢复。回退到更旧的二进制前，须盘点待确认及待核对操作并制定迁移或失效方案，避免旧代码绕过新绑定与预算检查。保留数据卷和业务库。
