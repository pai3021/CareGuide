# CareGuide 编排与运行可靠性实施记录

实现验收：2026-10-21；Git 交付记录更新：2026-10-23。依据：[最新版 TODO](careguide-orchestration-todo.md)；逐项验收：[acceptance.md](acceptance.md)；逐例结果及服务快照：[verification.json](verification.json)。

## 实施起点

用户选择“完全恢复 main，仅取回 TODO 后重新实施”，并要求不新建分支。沿用 `public-main`；基线 `origin/main` 为 `22a91b84bea5838f973c83d680a5b189eb6575dc`，实施起点 ahead/behind 0/0、索引及工作区无差异。2026-10-23 用户授权提交推送，交付使用已有 `codex/orchestration-reliability` 分支；`public-main`／远端 `main` 保留基线，不合并本轮实现。

TODO 单独取自 `origin/codex/orchestration-reliability` 的 `0d8bb04`，原 SHA-256 为 `b7c6f3c2a0e5dfbfe9962b57df8f290fa1d313e41d372d15dcc24488bd2bc87c`。旧 69 项修改已逐文件备份校验到 `.study/backups/before-main-sync-20261020T163518Z/`，另保留 stash；没有重新套用旧实现。保留 `.env`、模型及 Docker 数据卷，冻结语料、标注和评测结果未改。

先固定基线与错误契约，再一起接入 LangGraph、预算、等待和局部变更，最后完善长期记忆与集成，保证首次规划至恢复均沿用同一业务约束。

## 生产调用链及关键文件

`POST /chat` → `get_chat_runtime()` → `OrchestrationRuntime.process()`。独立急症边界先行；复合原文检测及保守简单查询白名单决定分流，分类器单标签不能屏蔽原文。简单完整请求直接路由；简单缺参、时效性号源及预约准备可进入不调用 Planner 的单任务持久流程。不确定请求交给 Planner。

- `orchestration/planner.py`：真实兼容 Chat Completions 接口读取完整原文、候选意图、已有任务、参数 schema、动作目录及消息接收时间，生成 PlanProposal／PlanChange；仅一次格式修正，SDK 隐式重试关闭。
- `contracts.py`、`bindings.py`：程序校验角色／动作权限、参数、原文、输入类型、环和规模；每版最多 12 项。候选集合不是用户选择，查所有医生号源需有针对该范围的授权。
- `runtime.py`：LangGraph 外层 plan／execute／check／replan／wait／finish 循环。任务 DAG 是业务状态数据，框架不自动解释模型生成的任意依赖。
- `scheduler.py`、`worker.py`：FIRST_COMPLETED 调度，最多 4 个任务并发、叶调用最多 4 个；下游只等自身依赖。复用现有 Agent、工具和医院服务，隔离参数、原文片段、结果和证据。
- `store.py`：局部变更仅失效受影响任务及下游，保留无关成功结果、有效确认和仍在执行的任务。任务版本及派发标识拒绝受影响的迟到结果；旧结果仅归档。修改日期保留未改变的已确认条件；相对日期绑定原消息的上海日历。
- `errors.py`、`replies.py`：安全错误分类、有限修正和确定性部分结果出口。最终模型失败不能抹掉成功资料；无必需的额外润色模型调用。

预约沿用 AppointmentService 和 `/actions/{id}/confirm`。ActionStore 原子核对用户、会话、参数及任务版本，领取后才写。已领取、已完成或待核对操作不能被改计划抹掉；改期／取消需新动作和确认。未知结果只读核对 FHIR 操作标识、患者和业务事实，重复确认不重复写；HTTP 200 也不能代替事务及预约结果核验。

## 持久化与恢复边界

业务任务、结果、动作和预算共用 `CAREGUIDE_BUSINESS_DB`。图检查点使用同目录独立 SQLite 文件，例如 `careguide.sqlite3` 与 `careguide.checkpoints.sqlite3`，均留在原业务卷。分文件修复了已复现的写锁问题：同步业务事务会阻塞事件循环，不能同时等同文件的异步 checkpoint 提交。两库不是同一事务；恢复以业务状态、派发标识及真实写事实为准，图节点重入不能绕过这些约束。当前验证单机 SQLite，未验证多主分布式部署。

| 对象 | 有效期和恢复规则 |
| --- | --- |
| Redis 会话 | 24 小时；基础 session 模式保留 |
| 工作流 | 30 分钟；过期后保留历史，重新发起新请求 |
| 补参等待 | 最长 300 秒，取工作流及上游结果期限的更早值 |
| 医生等结果／号源 | 300 秒／60 秒；恢复、派发前重验，过期刷新并重新选择 |
| action 确认 | 最长 15 分钟，同时受工作流、任务和输入有效期约束 |
| 长期偏好 | 字段 180 天，读取时逻辑清理并增加版本 |

等待通过 interrupt 结束当前 HTTP 请求；持久化缺参、引用、已完成结果及预算。恢复核对身份、会话、计划／任务版本、候选成员和有效期。多个问题无法唯一绑定自由文本时先澄清。过期不自动删业务事实，不授权复用旧确认。暂不自动物理清空历史表及过期投影。

默认预算为模型 20、工具 30、重试 3、重规划 3 次，累计调用执行 120 秒、单请求墙钟 45 秒，80% 起收尾。并发先原子预留，失败不退次数；取消／超时无法证明远程已停止时保守结算预留时间。累计执行是并发调用耗时之和，等待不计；墙钟和有效期分别处理。恢复、改版、确认或回退 direct 都不清零。缺失旧账本拒绝执行。收尾停止新增分支及探索，交付已有证据；没有自动增加旧流程额度的接口。usage 仅记录提供方实际返回值，缺失单独计数，不伪造 token／货币成本。

## 长期记忆

仅 `CAREGUIDE_MEMORY_MODE=redis_chroma` 启用持久路径。assistant 写入后先廉价筛选用户持续偏好；无候选及仅临时条件不调用提取模型。白名单为时段、医院、科室、院区、语言，每项必须有服务端用户消息 ID、原话、时间、范围和版本；不推断诊断、剂量，不存助手建议或他人情况。

SQLite 为权威画像，Chroma 为投影。来源和类型校验后去重、合并、更正、删除，CAS 检测并发冲突；投影 upsert 失败保留旧权威画像，过晚投影不发布旧提取。无来源旧画像不自动信任。消息来源元数据在发给模型前剥离非标准字段。

当前只自动复用验证后的长期时段到只读号源查询，本次明确条件优先；其他偏好仅作背景，不能自动替代关键参数。Redis 身份分隔符转义，普通旧 key 不变，有歧义的特殊字符旧 key 不自动读，按原 TTL 保留。记忆读取最多 2 秒、写入最多 8 秒，错误或卡住不抹掉业务回复。独立逐用户／逐日预算为模型 10、存储 10 次、累计 60 秒，每次最长 3 秒。

## 本机运行

工作区 `/mnt/d/mycode/CareGuide`；复用 Windows `D:\miniconda3\envs\nexusops-dev\python.exe`（3.12.14）、Node 24.16.0、Docker Desktop。未创建环境。新增 17 个依赖，host 原 133 个包、开发镜像原 100 个解析后的包版本不变，pip check 通过。固定 LangGraph 1.0.5、SQLite checkpoint 3.0.1、aiosqlite 0.21.0；0.22.1 的接口不兼容由实际恢复测试发现，已排除。其余见 requirements.txt。

| 服务 | 当前端口 |
| --- | --- |
| 后端／前端 | localhost:8010／5175 |
| FHIR／Redis／Chroma | 8080／6380／8002 |
| Prometheus／Postgres | 19091／仅容器网络 5432 |

沿用模型 `D:\mycode\models\CareGuide\careguide-intent-roberta-v1` 的只读挂载。原业务、检索、trace、Redis、Chroma、Postgres 卷保留。PowerShell 在项目根目录：

```powershell
docker build --pull=false --build-arg CAREGUIDE_DEV_BASE=careguide-careguide:before-orchestration-20261021 -f Dockerfile.dev-update -t careguide-careguide:latest .
docker build --pull=false -t careguide-frontend:latest frontend
docker compose --env-file .env -f docker-compose.yml -f docker-compose.full.yml -f docker-compose.integrations.yml -f docker-compose.transformer.yml --profile memory --profile integrations up -d --no-deps --no-build careguide frontend
```

发布时发现 main 语料指纹 `6e47b0ad0159…` 与旧镜像／索引 `1ac145b7c53a…` 不同。先显式用 `docker-compose.local-bm25.yml` 临时恢复，随后核对原镜像导出的全部 10,015 条真实 embedding 输入哈希，确认 ID 和检索文本完全一致。`scripts/reuse_verified_index.py` 进一步校验原指纹、模型、1024 维度、文档、来源和有限向量，禁网复用已付费生成的真实向量到新名称，未改冻结数据、旧索引或虚构向量。当前已恢复原 Hybrid 配置；BM25 层仅保留为显式备选。

迁移先写临时 collection，完整校验后改为正式名称，查询不能发现未发布半成品。Chroma 0.5 不允许更新中再次传入距离字段，初次未发布索引改名保留，不删除。回读全部 10,015 条向量最大差为 `1.4901161193847656e-08`（Chroma cosine 存储再次归一化的浮点差，不是逐位一致）；已用对应文档自身的真实缓存向量验证 Hybrid 查询路径，未作任意新查询的质量评测，embedding 调用为 0。

迁移只允许对可信原语料导出的绑定证明执行，目标存在即拒绝覆盖。可在现有开发镜像禁网运行脚本，指定 `--binding <原语料逐文档哈希JSON> --source <旧collection> --index-path <已有Chroma目录>`；本次实际命令、源／目标指纹及回读证据同步 verification.json。不能通过直接改旧索引指纹或使用不匹配缓存来绕过校验。

两个 profiles 用于解析已有依赖定义，`--no-deps` 仅更新应用。Dockerfile.dev-update 复用本机已配置 CPU Transformer 镜像，严格断言原包不变；标准 Dockerfile 仍是独立构建入口。Docker 忽略 .env、.study、本地凭据和模型。其他机器须改用其实际已有环境和挂载。

`CAREGUIDE_ORCHESTRATION_MODE=langgraph` 默认启用。模式和预算配置见 .env.example。复合请求需配置兼容模型；无配置明确返回规划不可用。健康及脚本验收不调用真实模型；实际交互若沿用现有提供方配置可能产生费用。本次未发出这类请求，未重建索引或下载模型。

## 复现

PowerShell 复用现有环境；示例注入可控模型、运行真实 Planner／worker／状态路径，无外部模型费用：

```powershell
$py = 'D:\miniconda3\envs\nexusops-dev\python.exe'
& $py -m pytest -q -p no:cacheprovider
node --test frontend/tests/workflow.test.js
Push-Location frontend; node node_modules/vite/bin/vite.js build; Pop-Location
node frontend/tests/browser-workflow.mjs
& $py scripts/reproduce_orchestration.py --scenario dependency
& $py scripts/reproduce_orchestration.py --scenario replan
& $py scripts/reproduce_orchestration.py --scenario partial
& $py scripts/reproduce_orchestration.py --scenario memory
& $py scripts/reproduce_orchestration.py --scenario wait --phase start --state-dir .study/repro-wait-new
# 上一进程退出后，在有效期内运行；新 start 必须用新目录。
& $py scripts/reproduce_orchestration.py --scenario wait --phase resume --state-dir .study/repro-wait-new
& $py scripts/verify_existing_services.py
```

运行全量 pytest 前移除 shell 继承的 CAREGUIDE_* 设置，将业务数据库指向临时路径；项目不自动读 .env，不修改其内容。正式 wrapper、JUnit、构建、浏览器报告位于 `.study/runs/20261021-main-baseline/`，逐例结果同步 verification.json。

服务脚本默认只读；`--write-isolated --state-dir <新目录>` 可复现 FHIR／Redis／Chroma 写入验证。仅创建本次 UUID 的合成医生、患者、号源、最终取消的预约及两个 QA collection；Redis QA key 120 秒过期，FHIR／Chroma QA 资源保留，不自动删除既有数据。测试 embedding 是本地确定性向量，仅验证存储链路。

API 的 answer 绑定 workflow_id、plan_version、task_id、task_revision、values、event_id；change 为明确新要求，cancel 无需模型停止未执行分支，run 恢复。GET `/workflows/latest?user_id=…&conv_id=…` 用于刷新恢复。身份不符 404、版本冲突 409、过期 410、字段错误 422。原确认入口保留，结果不明时同入口只核对。

## 验证及实际限制

main 基线 165 项：163 通过、2 失败，25.06 秒；原正向医院测试未固定日期，资料已过期。保留原断言和冻结有效期，给正向范围测试指定 2026-09-21，并增加过期时钟拒绝测试。最终全量 **332 通过、0 失败、0 跳过，51.63 秒**；Node 3 项、Vite 构建、Chrome 7 项交互通过。覆盖增加导致总测试耗时增加，这不是业务性能比较。2026-10-23 在交付分支提交前重新运行同一套 332 项测试，全部通过（pytest 56.09 秒）；Node 3 项再次通过。源码哈希与原验收镜像一致，新记录见 verification.json 的 pre_push_regression。

离线路由对比为相同 main 请求、固定单标签、oracle JSON、确定性 worker，每例 10 样本；5 个小样例任务 5/8 → 8/8、完整样例 2/5 → 5/5。当前简单医生／药品查询中位数分别为 17.505／20.952 ms，main 为 0.649／3.327 ms；三种复合夹具为 129.896–144.950 ms，每项新增一次脚本 Planner 调用。持久预算和状态有实际延迟开销，原始样本及工具次数见 verification.json。不能推广为真实模型成功率、分类器整体收益、临床有效性或线上吞吐。真实模型质量／费用评测未运行，付费调用 0。

已有 Docker 集成覆盖 FHIR 丢写响应 GET 对账、重复确认零写、改期／取消、真实 Redis／Chroma 偏好更新。浏览器是真 Chrome＋构建 Vue＋脚本 API；生产 API 为实际 FastAPI 工厂＋可控模型。分开报告，不冒充真实提供方端到端评测。保留 LangChain 序列化默认值弃用提示和 Chroma 原有 telemetry 兼容日志，未扩大反序列化权限或改动无关依赖。

user_id 沿用演示身份，尚无生产认证。SQLite 当前为单机方案；医院限合成服务，医疗资料边界不变。无自动物理删除历史、无限预算续跑或未知写操作的自动重提。

## 回退

首选在当前代码设置 `CAREGUIDE_ORCHESTRATION_MODE=direct`，按原 Compose 层重建应用配置。确认仍检查已保存任务绑定和预算；图状态保留，重新启用后仅恢复未过期流程。direct 模式不提供图卡片恢复，也不延长有效期。不要使用 down -v。

旧镜像保留为 `careguide-careguide:before-orchestration-20261021` 及前端对应标签。不能直接用旧后端接管含新待确认 action 的数据库：旧代码缺少本次绑定／预算保护。二进制回退须先只读盘点待确认和待核对操作，再确定迁移／失效方案；不能删除数据库或盲目覆盖现有事实。此次未执行该回退。
