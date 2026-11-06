# 编排架构与运行说明

CareGuide 默认采用 Orchestrator–Workers：统一轻量意图理解模块提供明确目标或候选线索，主编排器采用确定性计划，或通过内部 `UnderstandingPlanner` 为未解决请求理解并规划，程序控制派发与结果回收。LangGraph 管理执行、检查、局部重规划、等待和结束；DAG 的依赖、参数绑定和结果失效规则由业务代码实现。验收场景见 [验收矩阵](acceptance.md)，结果摘要见 [verification.json](verification.json)。

## 请求与执行路径

`POST /chat` → `get_chat_runtime()` → `UnifiedRuntime.process()`。

急症和受限请求先检查边界。结构化 answer/cancel/run 事件及单个明确补参绕过理解模型；确认继续使用原动作入口。其余自由文本先交给 `LocalUnderstanding`，输出 `IntentUnderstanding`，包含目标列表、各目标参数、缺参、来源、限制、关系、候选和未理解片段。规则、实体提取、本地分类与语义检索属于该模块内部能力，按需调用；分类和检索不会把候选示例变成用户事实或执行授权。

| 模块 | 实际职责 | 输出与处理条件 |
| --- | --- | --- |
| 轻量意图理解 `LocalUnderstanding` | 规则与实体解析明确请求；分类和检索候选通过只读准入后也可生成单任务 | 输出确定目标及参数，或候选、限制与未理解部分；不能丢掉剩余要求 |
| 主编排器 `UnifiedRuntime` | 采用组合器／模板生成的本地计划，或调用内部 `UnderstandingPlanner` 判断新建／补参／修改／取消／继续／澄清 | 计划统一校验；未知含义、关系、指代或变更由模型决策，最多一次格式修正 |
| Workers 与中央执行器 | 按角色、动作和参数合约执行任务，返回结果、缺参、失败及确认状态 | 参数必须有来源；缺参可恢复，业务准备操作仍需确认 |

`understanding.py` 的组合器将明确目标转换为任务节点；医生＋指定药品说明书可形成两个独立任务，`doctor-slots-v1` 模板组织医生→用户选择→号源。用户明确要求预约才增加号源选择→准备→确认，“先别预约”不生成准备动作。

分类器在严格规则不能解决时按需调用，沿用现有本地权重与标签顺序，惰性加载、只读、禁止下载。其输出包含 scores、margin、input_tokens 和 truncated；模型或依赖缺失、超时、标签不匹配均留下诊断并降级。Top-K 是竞争候选，不能直接生成多个业务目标。softmax、字符及稠密向量相似度均未校准，分数只参与准入筛选，不能代替动作、参数与文本覆盖检查。本模式不增加分类 LLM 调用，未知请求的语义检索通过显式预算路径调用 embedding API。

### 分层理解与候选获取

`query_parser.py` 是轻量理解模块的确定性解析路径，统一解析已支持的医生、号源和确切药品说明书表达，包括“想找一位看消化内科的医生”“我想看看明天下午还有哪些号源”“我想看看精氨酸布洛芬片的禁忌”。完整理解的请求直接生成单任务或已有组合，不先调用分类／embedding 再重复匹配相同句式。

- 参数从当前原文提取，相对日期绑定本次消息的上海日历，非法日期不能被静默忽略。任何未理解分句或限制都保留到 Planner；已经支持的明确独立组合和“先别预约”模板继续本地处理。
- 已知药品缺规格时建立可恢复任务，复用原追问和补参逻辑。新查询保留旧等待任务；本地可确定的新查询使用独立账本，经过候选／Planner 才识别的新查询保留已消耗额度并沿用原账本。
- 默认 `CAREGUIDE_INTENT_CANDIDATE_ROUTING=read_only` 开放候选到模板的有界路径：仅 `search_doctors`、`list_available_slots`、`lookup_drug_label`。分类或稠密检索提出动作，`candidate_routing.py` 独立提取当前实体并检查单一对象、日期合法性、动作词和全部剩余文本。支持有限查询词汇的不同顺序和组合，不要求再次命中 `query_parser.py` 的完整句式。词汇之外的条件、否定、指代、多对象、多日期、未解析子目标及写操作均拒绝，保留原文和候选交给 Planner。
- 旧 CareGuide-13 权重不变。显式适配器把 `appointment_action` 映射为查医生／号源候选族，把“注意事项”映射为说明书候选，再由各动作的槽位与文本覆盖合约消歧。医生与号源准入必须包含一个明确的已知科室；药品准入必须包含一个明确的已知药名，缺规格可进入已有等待任务。候选里的参数不被复制；药品剂型沿用明确药名对应的目录解析。
- 分类筛选要求完整、有限且归一化的 13 类分布、非截断、最高分至少 0.90、与次高分差至少 0.20。语义筛选要求前两条不同参考均对应同一新查询动作、两条分数均至少 0.85，且第二条比不同动作／操作的候选至少高 0.10；参考 ID、原文、动作、操作和版本必须匹配本地可信库。字符近邻不参与准入。以上是版本化工程默认值，尚未完成独立校准；单一高分不足以生成计划。
- `CAREGUIDE_INTENT_SEMANTIC_POLICY=parallel` 为默认：规则未解决时并行获取分类和语义候选，然后统一准入，强候选冲突时拒绝准入。可选 `on_demand`：分类准入成功则跳过远程检索和 Planner；分类未准入再继续业务检索，仍未准入才交 Planner。并行让两种候选共同参与判断；按需可能减少检索请求，也可能增加串行等待，质量及延迟需要另行测量。
- 设置 `CAREGUIDE_INTENT_CANDIDATE_ROUTING=disabled` 可恢复只向 Planner 提供线索的方式；该模式下按需检索沿用分类严格多数启发。结构化 change 事件始终禁用新任务准入。后获取候选才确定的新请求保留已消耗预算并沿用原账本，不覆盖旧等待任务。
- trace 的 `local_understanding.candidate_admission` 记录准入版本、接受／拒绝、原因、来源和工程阈值；`semantic_retrieval` 继续记录检索策略与实际请求数。按需模式分类准入后检索状态为 `skipped_candidate_plan`，新计划的 `plan_source=candidate_template`。检索器的 `direct_acceptance=False` 表示近邻本身只是参考，实际准入由独立合约完成。

规则命中跳过分类、候选检索和理解／规划调用；候选准入通过后也会跳过 Planner，后续回答生成仍可能使用模型。候选准入的受控回归见 `tests/test_candidate_routing.py`，验证新增表达、分类与语义独立准入、拒绝边界、补参、旧任务和预算；工程控制验证不等于模型质量评测。早期只测候选获取的 20 例 [控制流程对照](../evaluation/layered-understanding-control-20261105.json) 保留其原源码哈希，未覆盖本次新增准入。

当前版本完整离线回归为 **605 passed**。使用真实已有 RoBERTa 和 12 条固定本地开发探针检查，得到 1 个原规则计划、0 个新增分类计划、11 个未解决请求，无意外本地动作。多条改写被旧模型判为“其他”，号源表达的分类分数也未达到准入默认值；这说明分支已通过受控测试，但旧模型尚未展示新增覆盖。没有根据这些结果降低阈值或重训权重。当前语义准入只完成受控向量候选验收，尚未运行新版本的真实 embedding／Planner 对照。结果和源码哈希见 [本地权重检查](../evaluation/candidate-routing-native-20261105.json)。可使用以下无网络命令复现同一组探针：

```bash
python scripts/evaluate_candidate_routing.py --model-path /path/to/careguide-intent-roberta-v1 --output tmp/candidate-routing-native.json
```

随后补充了八组共 64 条业务动作标注并在首次真实推理前冻结。无网络消融中，规则与规则＋旧分类器均生成 9 个本地计划、动作集合均匹配 9 条，新增分类准入为 0；其中两条时间参数需 Worker 后续绑定，不能把原始计划字段留空当成执行失败。仅语义与联合策略未运行，默认策略未修改。数据、方法和逐项结果见 [业务动作评测](../evaluation/business-intent-eval-v1.md)。该版本完整离线回归为 618 passed。当前不宣称总体效果或成本收益，所有修改均未部署到已有服务。

上述评测之后，按项目的架构表达选择，将源码、配置模板和 Compose 的默认策略统一改为 `parallel`，保留 `on_demand` 显式覆盖。该默认值调整未追加评测或外部请求，历史报告仍以各自源码哈希为准；已有 `.env` 或进程中显式设置的策略优先，不会自动被新默认覆盖。

2026-11-05 的前一版本（候选仅供 Planner）使用真实本地 RoBERTa、MiMo Planner 和 Qwen embedding，完成六条固定开发样例的真实服务对照。只生成并校验计划，共 32 次 chat 和 8 次 embedding 请求，无医院工具调用；该快照离线回归为 546 passed。以下历史结果不代表本次候选准入，各配置结果如下，括号为 chat／embedding 请求数：

| 候选配置 | 按需检索运行 | 并行检索运行 |
| --- | --- | --- |
| 规则＋Planner | 6/6（5／0） | 5/6（6／0） |
| 规则＋分类候选＋Planner | 6/6（6／0） | 6/6（5／0） |
| 规则＋分类及语义候选＋Planner | 6/6（5／2） | 6/6（5／6） |

完整候选配置下，按需策略在该组样例中减少了四次 embedding 请求，两种策略均通过六例。每次运行均含一条零模型请求的本地控制，首次语义检索包含一批示例库向量化。无候选基线不受检索策略影响，其第二次运行的历史指代样例发生一次 `PlanningFailure`，不应把两次基线差异归因于策略。测试仅保留该失败的异常类型，未确认底层错误码；未追加外部请求。样例是开发探针，固定顺序、每组一次，不能据此确认分类／检索提升总体准确率或线上速度，也不能将调用数直接换算为货币节省。源文件哈希、逐例结果、预算和提供方 usage 见 [本次真实服务对照](../evaluation/intent-live-comparison-20261105.json)。这些修改尚未部署到已有服务。

主模型接收完整当前原文、最近最多六条有界历史、用户来源和当前任务状态；历史助手／工具内容不能成为授权来源。历史用户原话必须显式引用真实 message_id/quote。缺少原始时间的历史相对日期只供理解，不能当作新的日期依据；已有来源的相对日期按原消息的上海日历解析。信息不足返回澄清，不复制全历史给子 Agent。

- `understanding.py`、`decisions.py`、`understanding_planner.py`：轻量理解、本地组合／模板和统一 RequestDecision；读取原文、候选、已有任务、schema 及 worker 职责目录，关闭 SDK 隐式重试。`planner.py` 保留兼容入口及共享预算辅助。
- `query_parser.py`：共享单查询解析；`understanding.py` 统一处理组合与模板，按策略获取候选；`candidate_routing.py` 对单目标只读候选做槽位和覆盖准入。
- `contracts.py`、`bindings.py`：校验角色权限、参数、原文片段、输入类型、环和规模；每版最多 12 项。候选集合不等于用户选择，查询所有医生号源需要对应范围的授权。
- `unified.py`：应用统一决策并记录 `plan_source=single/compose/template/candidate_template/llm`、识别来源与模板版本；`runtime.py` 共享 LangGraph 执行、急症中止、中断与恢复。
- `scheduler.py`、`worker.py`：最多 4 个任务并发、叶调用最多 4 个；下游只等待自身依赖。复用已有 Agent、工具和医院服务，并隔离任务参数、结果和证据。
- `store.py`：局部变更只失效受影响任务及其下游，保留无关成功结果和有效确认。任务版本和派发标识拒绝受影响的迟到结果。相对日期绑定原消息的上海日历。
- `errors.py`、`replies.py`：分类错误、有限修正并固定展示业务状态。`answering.py` 在完成或部分完成且有可用查询资料时，用一次有预算的模型调用生成自然语言正文；单任务与多任务共用，不再额外调用摘录汇总。`synthesis.py` 保留为旧摘录方案的兼容实现。

动作目录同时提供各动作的 completion 合约：医院查询返回 `items`，准备操作返回 `prepared`，资料检索返回 `source_evidence`。目录和程序校验共用同一映射，模型不能自行猜测完成条件；格式修正最多一次，反馈包含有界字段位置／类型或合约检查提示，不回传原始输入或私有异常。

单任务与组合／模板都生成同一 `PlanProposal`，统一校验和持久执行。主 Agent 只提出决策；中央程序执行器负责 DAG 和权限，子 Agent 不彼此发送消息。急症角色使用既有优先出口，不等同于普通可规划的医疗诊断任务。

## 业务意图语义检索

`semantic_intent.py` 复用 `OpenAICompatibleEmbeddingFunction` 和 `CAREGUIDE_EMBEDDING_*` 的模型、维度、地址及密钥配置，默认对应硅基流动 `Qwen/Qwen3-Embedding-0.6B`／1024 维。它创建独立的业务检索实例，使用“召回相同业务意图和会话操作示例”的 query instruction。医疗 RAG 继续使用原检索指令；已有知识索引和 CareGuide-13 类别中心不被修改或直接当作业务动作标签。

`intent_examples.json` 包含 36 条人工编写的合成业务表达，覆盖实际动作目录、补参／变更、指代、否定及范围外请求。它是检索参考库，不是临床资料、用户授权或自动执行计划。返回最多 4 条近邻，包含动作候选、会话操作、注释、示例 ID 和版本；不把向量传给主模型。主模型仍须按当前原文的目标、否定和参数来源决定任务，示例中的预约措辞不授权当前操作。

| 设置 | 行为 |
| --- | --- |
| `CAREGUIDE_INTENT_SEMANTIC_MODE=auto` | 默认；有有效 embedding 密钥时配置远程候选，未知请求且有主模型时才实际调用；无配置保留字符近邻 |
| `remote` | 明确要求配置 embedding 密钥，否则启动报错；运行中的服务故障有界降级 |
| `local_ngram` | 意图候选保持本地；不调用 embedding API |
| `CAREGUIDE_INTENT_SEMANTIC_POLICY=parallel` | 默认并行获取分类和业务检索候选，经同一准入模块判断；未解决才进 Planner |
| `CAREGUIDE_INTENT_CANDIDATE_ROUTING=read_only` | 默认开放三种有界单目标只读模板；`disabled` 恢复候选仅供 Planner |
| `CAREGUIDE_INTENT_CLASSIFIER_MODE=auto` | unified 默认保留旧本地分类源；`disabled` 可关闭分类源以对照仅语义，不改变旧 direct/langgraph 的分类配置 |
| `on_demand`（POLICY） | 可选先分类，分类准入未通过再检索，保留作同输入的质量／时延／调用数对照 |
| `CAREGUIDE_INTENT_SEMANTIC_TIMEOUT=3` | 向量候选总超时，允许 0～10 秒之间的正数；实际 HTTP 超时进一步受请求剩余额度约束，SDK 重试为 0 |
| `CAREGUIDE_INTENT_SEMANTIC_INDEX` | 可选预构建业务索引路径；为空时首次未知请求向量化示例库一次。显式文件缺失或版本不符时降级，不偷偷重建 |

每个 runtime 在内存保留公共示例矩阵。索引指纹覆盖模型、维度、服务地址、query instruction 和完整示例版本，加载时核对指纹、长度、有限数及非零范数；不需要新增向量数据库。查询候选缓存最多 256 项、有效期 300 秒，键包含配置指纹、用户、会话和当前文本的哈希；缓存不保留原始查询，返回结果为独立副本。并发首次请求只生成一次公共索引；服务故障后 30 秒暂缓再次请求。HTTP 超时不代表远端未执行，失败请求不退次数；超时保守结算预留时间。

规则／结构化事件早退出，不产生向量调用；没有主模型时也不为未知请求增加无用的远程费用。候选在预算建立后按 `on_demand/parallel` 策略获取，之后一次主 Agent 决策。每次真实 embedding SDK 请求计入共享 `model` 额度和执行时间；提供方 usage 计入 token 统计，未返回 usage 时不编造。冷启动无索引通常需要示例批次和查询共 2 次请求，预构建或已热索引仅查询 1 次，缓存命中 0 次；它们不是生成式 Planner 调用数。查询得到独立新流程时，已消耗的向量／Planner 额度继续随原账本保留。

下列默认评测命令仅运行本地字符基线，不联网：

```bash
python scripts/evaluate_intent_semantic.py --mode local_char_ngram
```

`intent_semantic_cases.json` 是单独人工编写的 26 条合成样例，与参考库无逐字重复，覆盖改写、多目标、指代、变更、否定及域外。指标是 top-k 示例候选的目标动作覆盖率：22 条有动作目标的样例逐条计算覆盖比例再取平均；域外／受限样例单独列出，不伪装成已校准的拒绝准确率。字符基线用原 6 示例／top-3，稠密检索用业务库／top-4，两者包含示例库与检索器的共同差异，不是控制变量的 embedding 优势比较。受控向量／SDK 测试只验证协议和执行边界，不计入真实语义质量结果。

以下命令明确调用已配置的 embedding 服务，可能产生费用；不会自动读取 `.env`，本轮离线验收未运行这些远程命令：

```bash
python scripts/build_intent_semantic_index.py --output .study/business-intents-v1.json
python scripts/evaluate_intent_semantic.py --mode remote
```

预构建命令拒绝覆盖已有目标；成功后可将索引放到运行进程可读的位置并设置索引路径。容器配置必须使用容器内可读路径。真实语义召回、阈值校准、线上延迟与费用收益尚未评测，语义直接接受保持关闭。

## 子任务上下文与成果

worker 默认接收独立 `TaskBrief`，包括动作目标、用途、带来源的已知参数、允许动作、约束、输出格式与完成条件，模型消息历史为空。上游数据经过类型／版本／选择校验后绑定成最小参数。没有公共对话 memory 供 worker 任意通信，没有 A2A，也不 Fork 主 Agent 全历史。缺参、候选选择和失败经结构化结果返回中央执行器。

`WorkerResult` 包含状态、摘要、数据、证据引用、缺失输入及失败。数据 JSON 超过 32 KiB 时，worker 返回同业务 SQLite 内的成果引用；读取核对用户、会话、工作流、任务版本、期限、大小和 SHA256，拒绝任意文件路径。权威业务结果和完整工具轨迹仍保存在服务端。主模型只收到有界任务摘要或证据摘录，不接收整个工具历史；成果引用由服务端读取，不授予模型文件系统权限。

回答 Agent 读取有界当前问题、任务状态及成功任务的资料／结构化结果，为每条资料分配 `[S1]` 等来源标识，生成可自然改写的中文回答。系统提示要求只解释来源中的一般资料，不作个人诊断、处方或剂量决策；医院数据明确为合成信息。模型没有工具或确认能力，程序单独追加业务状态，完整候选、证据和确认卡片仍来自权威任务结果。`medical_answering_enabled=False` 继续表示未开放临床决策能力，不表示未使用模型组织资料。

公开配置、Compose 与生产工厂默认 `mimo-v2.6-flash`／`https://api.xiaomimimo.com/v1`，其他兼容服务可显式覆盖。MiMo 请求关闭 thinking；只转发 `delta.content`，不转发 reasoning、工具参数或 Planner JSON。完整回答阶段最多 15 秒，并受共享模型额度和请求剩余时间约束；SDK 不重试。usage 从流末提供方报告读取，缺失则记录缺失，不推算 token。单任务也可能增加一次回答调用；未配置模型、等待选择／补参／确认、无可用资料、额度不足时保留程序回复。

程序检查正常结束标志、长度、来源标识、链接及部分不允许的操作成功措辞。它不证明每句话都由来源蕴含，也不验证医学正确性；生成式改写需要后续真实效果评测。生成途中显示的正文带“尚未完成”状态，收到 `result` 才替换为最终回答；失败发送 `answer_reset` 撤回草稿并回退，不重跑工具。回答期间任务版本／revision 发生变化也撤回旧回答，以当前任务状态返回。记忆和 trace 保存同一个最终回复。

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

## SSE 聊天与进度

网页默认使用 `POST /chat/stream`，请求体与 `/chat` 相同，也支持原有 `workflow_event`。保留 `POST /chat` 的一次性 JSON 接口；两者共用会话锁、记忆读写、编排器与确认保护，不分别执行同一请求。前端使用 fetch 消费 POST 响应，不使用只能自动发起 GET 的 EventSource，也不自动重试 POST 或退回 JSON 重发。

| SSE event | 内容与用途 |
| --- | --- |
| `session` | 流开始返回 stream_id、conversation_id，前端立即保存会话，便于断线后恢复 |
| `stage` | 理解、动态规划、重规划、整理结果、保存会话等阶段，不包含隐藏推理或模型原始 JSON |
| `workflow` | 已落库任务的有界进度快照：工作流版本／revision，任务标签、状态、角色和依赖；只读展示，不用于确认或补参授权 |
| `delta` | 模型正文的真实内容增量：answer_id、Unicode 码点 offset、text、provisional；前端按序追加，展示为未完成草稿 |
| `answer_reset` | 撤回指定 answer_id 的草稿，等待程序回退或当前版本结果 |
| `result` | 原 JSON 接口的完整最终结果；刷新证据、可交互任务卡片与确认入口，然后结束连接 |
| `error` | 开流后的结构化错误，含 status 与用户可读 message；不会把中途失败当作完成 |

每个流的 id 从 0 递增；每 10 秒无事件时发送注释心跳。前端解析 UTF-8 分片、CRLF、多行 data，忽略重复／倒序序号；任务快照仍按工作流版本和 revision 防止旧进度回退。最终结果通过既有完整契约展示，只有最终卡片可发起补参／确认。

这是实时的**进度与模型正文 SSE**。后端开启真实 `stream=True`，从提供方流中逐段转发正文，不把完整回答事后切片。Planner 的未校验提案、工具参数和隐藏推理不会作为正文事件转发。流式生成的草稿尚未通过最终引用检查，最终 `result` 是权威展示结果。

请求体或恢复事件缺字段可在开流前返回常规 HTTP 错误；开流后 HTTP 状态已经确定，运行中的 404／409／410 等通过 `error.status` 传递。只有收到 `result` 才算完成，意外 EOF、协议错误或接收超时提示显式恢复，不自动重复提交。客户端最长接收 90 秒，后台执行仍受原请求与累计预算限制。

“停止接收”或网络断开不代表业务取消。服务端在本进程内继续有界执行并保存任务／会话结果；客户端保留会话 ID，通过“恢复会话任务”查看权威状态，取消业务分支仍须发送原结构化 cancel 事件。事件队列最多 64 项，慢客户端可以丢中间进度快照；正文 delta、撤回事件和最终 result/error 在保持连接时不会丢弃，正文采用背压。断线释放等待发送的正文，后台仍受原预算约束。进度不是持久事件日志，不支持 Last-Event-ID 重放或跨进程自动续流；进程重启后使用既有工作流恢复接口。

响应设置 `Cache-Control: no-cache, no-transform` 与 `X-Accel-Buffering: no`；前端 Nginx `/api/` 关闭缓冲、缓存和 gzip，上游读取超时 120 秒。Vite 继续使用原代理。配置改动需实际重新构建／部署才对已有服务生效。

## 预算和长期记忆

默认累计额度为模型 20、工具 30、重试 3、重规划 3 次，累计调用执行 120 秒、单请求墙钟 45 秒，80% 起收尾。并发先原子预留，失败不退次数；取消或超时无法证明远端已停止时保守结算预留时间。累计执行是并发调用耗时之和，用户等待不计。恢复、改版、确认及 direct 回退沿用旧账本；缺失账本拒绝执行。未知跟进经模型判定为独立新请求时，新流程与已付出判断成本的旧流程共享账本，不通过新建 ID 清零；严格本地可识别的新查询使用独立账本。结构化取消不要求剩余模型额度。usage 只记录提供方实际返回值，不推算缺失 token 或货币费用。

仅 `CAREGUIDE_MEMORY_MODE=redis_chroma` 启用持久记忆。用户持续偏好经廉价筛选后才调用提取模型。白名单为时段、医院、科室、院区、语言，每项绑定用户消息 ID、原话、时间、范围和版本。

SQLite 为权威画像，Chroma 为投影。来源与类型校验后去重、合并、更正或删除，CAS 检测并发冲突；投影失败保留旧权威画像。当前仅自动复用验证后的长期时段到只读号源查询，本次明确条件优先，其他偏好作为背景。记忆读取最多 2 秒、写入最多 8 秒，失败不抹掉业务回复。逐用户／逐日独立预算为模型 10、存储 10 次、累计 60 秒，每次最长 3 秒。

## 运行与复现

使用 Python 3.12，按仓库根目录 [README](../../README.md) 安装依赖和启动服务。配置模板为 `.env.example`；代码、配置模板和 Compose 默认 `CAREGUIDE_ORCHESTRATION_MODE=unified`。已有进程／配置文件显式设置优先，修改默认值不会自动部署或覆盖它们。未知请求的动态规划需要兼容 Chat Completions 的模型配置，无配置时明确澄清并说明无法可靠规划；已知单任务、医生＋药品组合及医生→号源模板无需模型密钥。

本地分类候选复用 `CAREGUIDE_QIC_MODEL_PATH` 和 `CAREGUIDE_QIC_MODEL_DEVICE`，依赖已安装的 torch/transformers；不随服务启动下载权重。空路径时尝试仓库同级 `models/CareGuide/careguide-intent-roberta-v1`。缺依赖或缺权重按候选不可用降级，仍可运行本地规则与配置好的主模型。本模式使用独立理解层，不执行旧 QIC Fusion 的分类 LLM 分支；旧 `CAREGUIDE_INTENT_TAXONOMY` 等分流配置用于 langgraph/direct。

LangGraph、checkpoint 和 aiosqlite 版本固定在 `requirements.txt`。标准 `Dockerfile` 支持独立构建；`Dockerfile.dev-update` 仅适用于已有兼容镜像的增量更新。模型使用只读挂载，业务库和检查点需持久化到业务卷。

```bash
python -m pytest -q -p no:cacheprovider
node --test frontend/tests/workflow.test.js frontend/tests/sse.test.js
npm --prefix frontend run build
node frontend/tests/browser-workflow.mjs
python scripts/reproduce_unified.py --scenario compose
python scripts/reproduce_unified.py --scenario template
python scripts/reproduce_unified.py --scenario wait --phase start --state-dir .study/unified-wait-new
python scripts/reproduce_unified.py --scenario wait --phase resume --state-dir .study/unified-wait-new
python scripts/reproduce_orchestration.py --scenario dependency
python scripts/reproduce_orchestration.py --scenario replan
python scripts/reproduce_orchestration.py --scenario partial
python scripts/reproduce_orchestration.py --scenario memory
python scripts/reproduce_orchestration.py --scenario wait --phase start --state-dir .study/repro-wait-new
# 上一进程退出后，在有效期内运行；新 start 使用新目录。
python scripts/reproduce_orchestration.py --scenario wait --phase resume --state-dir .study/repro-wait-new
```

`reproduce_unified.py` 使用本地组合／模板、真实 worker 与状态路径，模型调用为 0；template 示例显式选择合成医生与号源并确认两次，验证只创建一次。wait 的 start/resume 分别在独立进程执行，累计工具从 1 到 2，成功分支只执行一次。兼容脚本 `reproduce_orchestration.py` 注入可控模型响应，执行真实 Planner、worker 与状态路径。测试进程不自动读取 `.env`；运行全量测试时使用独立业务数据库并避免继承部署环境的 CAREGUIDE_* 设置。浏览器测试需已安装 Chrome，可通过 CAREGUIDE_TEST_BROWSER 指定浏览器路径。

`scripts/verify_existing_services.py` 默认只读。`--write-isolated --state-dir <新目录>` 创建独立 UUID 的合成资源验证 FHIR、Redis、Chroma 链路；QA 资源按脚本规则保留，测试向量只验证存储，不代表检索质量。

索引复用脚本 `scripts/reuse_verified_index.py` 校验可信原语料导出的逐文档哈希、模型、维度、来源与有限向量，目标存在时拒绝覆盖。先写临时 collection，完整校验后发布；不会通过修改旧索引指纹绕过校验。它用于已验证真实 embedding 的复用，不能替代新查询的检索质量评测。

## 验证范围与回退

自动化验收覆盖权限、参数绑定、依赖并发、部分结果、局部变更、预算、中断恢复、重复确认和长期偏好。具体结果与时间见 verification.json。浏览器使用真实 Chrome 与构建后的 Vue，API 响应为脚本；生产 API 测试使用实际 FastAPI 工厂与可控模型。历史服务集成验证与本次代码回归分开记录。

真实模型规划仅完成上述六例开发对照，尚无代表性质量评估；线上费用收益和临床有效性尚未评测。user_id 为演示身份，尚无生产认证；未验证多主分布式部署。医院限合成服务，资料未经临床审核。过期历史目前逻辑隔离，不自动物理删除。

需要使用旧 Planner 入口时设置 `CAREGUIDE_ORCHESTRATION_MODE=langgraph`；使用旧直接路由可设置 `direct`，两者都需重新构建应用配置。direct 模式不提供图卡片恢复。当前共享代码能解析新增可选字段，旧入口和直接确认同样遵守 budget_id；旧入口仍不具备统一理解／混合决策语义，切换前应完成或隔离这些未完成请求。已有旧流程在 unified 中仍受原版本、选择和确认检查。回退到更旧二进制前盘点待确认及待核对操作并制定迁移或失效方案，保留数据卷和业务库。
