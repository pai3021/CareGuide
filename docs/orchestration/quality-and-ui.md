# 回答展示、质量评测与访问边界

本轮保留原 Worker 调用链，仅增加明确的工具执行进度。模型内部推理和工具参数仍不作为正文展示。

## 展示与计时

回答正文使用禁用原始 HTML、远程图片和危险链接协议的 Markdown 渲染。模型生成中的正文仍是可撤回草稿；最终结果通过 `answer.body`、`answer.sources` 提供正文与来源映射，`response` 保留兼容的完整纯文本。点击 `[S1]` 定位并展开对应来源。任务状态和确认卡片继续来自程序，不由 Markdown 或模型文字授权。

用户向上滚动后暂停跟随，回到底部恢复跟随。阶段提示为理解、规划、工具执行、生成回答和保存会话，不展示隐藏思考过程。

`timings` 包含聊天处理墙钟、首段正文及分阶段耗时，不包括网络传输和浏览器渲染。模型／工具调用与规划／回答阶段可以嵌套或并行，不得相加当作总耗时。首段正文可能是随后撤回的草稿。终端响应包含会话写入时间；trace 在业务回复生成时保存，因此其 timing 快照早于会话写入。

## 回答忠实度

在线保留引用成员、任务覆盖、链接和操作措辞检查，新增有限的量值检查：比较明确的质量、体积和百分比，支持 `0.2g = 200mg`。证据包括工具返回的规格元数据，不把用户问题生成的显示标题当作证据。来源含数字范围时主动弃权；它不理解复杂否定、适用条件、隐含推论或所有数字表达，也可能保守地拒绝合法改写。失败继续撤回草稿并交付原查询结果。

四个预先编写的合成资料用例分别检查规格、否定、条件和来源对应关系，包含正确／错误对照：

```bash
python scripts/evaluate_answer_faithfulness.py --mode offline
python scripts/evaluate_answer_faithfulness.py --mode live --output answer-report.json
```

`live` 明确调用已配置的模型：生成回答、评审生成回答，并评审已知错误对照。Judge 的回答引文和证据引文必须逐字存在。当前使用同模型族评审，有相关性偏差，只作为暂定诊断；不把其结论当独立医学或语义准确率。脚本不自动加载 `.env`，不调用医院写操作，已有输出文件不会覆盖。

## 意图层对比

```bash
python scripts/evaluate_intent_ablation.py --mode offline
python scripts/evaluate_intent_ablation.py --mode live --output intent-report.json
```

三组为规则＋主模型、增加原 RoBERTa、再增加业务 embedding。相同六个固定输入、相同模型与结构校验，统一去除字符近邻以隔离新增候选的作用。计划只生成和校验，不执行工具。评估动作集合、参数、禁止动作、错误、耗时和调用数；不等于端到端业务成功率。

真实分类器或 embedding 不可用时显式标记，不能用降级结果冒充该模块效果。分类器预热单独记录，向量冷启动费用计入该组。一例明确医生查询为本地控制项；其余覆盖改写、多目标、否定、范围及历史指代。每组仅一次运行，顺序和服务负载可能影响时延，不能宣称性能优势。

2026-10-26 的小规模结果：

| 配置 | 目标与参数完整 | 模型请求数（含 embedding） |
| --- | ---: | ---: |
| 规则＋主模型 | 5/6 | 6 |
| ＋RoBERTa | 6/6 | 5 |
| ＋业务 embedding | 6/6 | 11 |

这批样例没有显示 embedding 带来额外完成率收益，也不足以证明分类器的普遍增益。当前仍保持候选辅助模式，不启用未经校准的阈值直达。

回答首轮四例中两例生成成功、两例回退。根据过程性冗余表述精简通用提示后，同一开发集四例均生成成功，Judge 将已列出的生成断言标为支持，并将四组已知错误对照标为不支持。它是重复使用的开发集复测，不是独立测试集或完整断言覆盖证明。原始记录保存在本地忽略目录，公开摘要见 `quality-verification.json`。

## 可选身份认证

默认保持本地演示模式，`/auth/me` 明确返回 `local_demo`。启用后，身份来自服务端令牌映射；请求体或查询参数不能指定其他用户。工作流、动作和 trace 保持用户隔离，管理接口另要求 `admin`。这是可撤销的随机 Bearer token 接入，不提供密码登录、OIDC 或 SSO。

在不纳入 Git 的位置创建哈希凭据文件；命令只显示一次原始令牌：

```bash
python scripts/create_access_token.py --file .study/access.json --user alice --role user
python scripts/create_access_token.py --file .study/access.json --user maintainer --role admin
```

运行进程设置 `CAREGUIDE_REQUIRE_AUTH=true` 和 `CAREGUIDE_AUTH_FILE`。容器必须把文件只读挂载，并使用容器内路径，例如：

```yaml
services:
  careguide:
    environment:
      CAREGUIDE_REQUIRE_AUTH: "true"
      CAREGUIDE_AUTH_FILE: /run/secrets/careguide-access.json
    volumes:
      - ./.study/access.json:/run/secrets/careguide-access.json:ro
```

令牌在网页“访问设置”中输入，仅按 API 地址保存在当前标签页的 sessionStorage，不写入长期设置。令牌哈希从文件读取；删除对应记录即可撤销访问，配置错误拒绝访问。对外访问还需 HTTPS 和适当的网关限流；此功能不替代完整账户系统。本轮不自动开启本地服务认证，也不生成或提交任何真实令牌。

## 工程检查

模型调用失败统一输出稳定错误码和中文提示：`model_timeout`（程序或接口超时）、`model_auth_failed`（401）、`model_quota_exceeded`（402 或明确的余额/额度耗尽信号）、`model_rate_limited`（普通 429）、`model_connection_failed`（网络连接失败）、`model_access_denied`（其他 403）。未知错误与服务端故障保留 `model_unavailable`。不展示供应商原始报错或凭据；不自动重试鉴权或额度错误。规划与 Worker 使用相同分类；回答生成失败会保留已完成工具结果，并在回复中附上对应提示与结构化诊断。回答校验失败仍使用既有校验诊断，不归为供应商故障。

此增量通过 106 项离线回归（模型错误分类、两种 Planner、Worker 失败、统一理解、回答流及聊天 SSE），包括真实 OpenAI SDK 的模拟 HTTP/网络错误和回答失败保留成功工具结果；未为此增量调用真实模型。

后续修复计划执行：已分配动作的 Worker 首轮指定工具；供应商忽略工具选择且未执行时，使用计划中已校验的动作和参数经过原有 dispatcher 执行，不把空调用误记为资料未覆盖。一般就医检索使用绑定的 query，避免把整份任务简报当检索词。简报保留原始任务目标，输出格式不作为额外任务；材料查询缺医院/院区时澄清。相关离线回归 118 项通过。一次真实 MiMo 原句复测（“帮我整理一下就诊前需要准备哪些材料，用列表回答。”）返回 waiting_user、无错误，耗时约 6.6 秒；这是澄清流程验证，不代表已实现通用材料清单问答或一般延迟指标。

CI 已运行前端协议与渲染测试。Markdown 依赖固定版本，现有 Vue/Vite 及其依赖仅在既有主版本范围内更新补丁，锁文件一并提交。原单机 SQLite、非持久 SSE 事件和兼容运行入口继续保留。
