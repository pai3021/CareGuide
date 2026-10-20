# 多任务编排与运行可靠性实施记录

## 授权与边界

2026-10-20 用户解除本地 TODO 的实施暂停，授权 LangGraph、结构化规划、局部重规划、任务等待恢复、累计预算、异常出口和长期记忆保护。医院仅使用合成数据；预约写入继续通过原有准备／确认服务。未经确认不创建主要环境、切换操作系统、引入外部服务或付费调用。

核心依据：[多任务编排与运行可靠性 TODO](careguide-orchestration-todo.md)。清单已从被 Git 忽略的 `.study/` 迁至本目录，与本文及后续验收记录一起纳入版本控制。

## 初始核查

- 基线提交：`22a91b84bea5838f973c83d680a5b189eb6575dc`。
- 起始分支 `main`，工作区干净；已创建 `codex/orchestration-reliability`，未推送或合并。
- 只读联网核查远程 `main`：同为上述提交。沙箱内网络失败，获自动审批后重试成功。
- 项目及父路径未发现适用 `AGENTS.md`。已读取 README、CI、依赖、原目录完整 TODO。
- Windows 路径 `CareGuide` 与当前挂载路径 `careguide` 指向同一仓库。
- 已将基线提交完整导出到 `/tmp/careguide-orchestration-baseline`，用于隔离运行改造前回归，不覆盖现有文件。
- 当前 WSL：`/usr/bin/python3` 3.10.12，无 pytest；另有 Python 3.11.15。README 和 CI 要求 Python 3.12。
- 只读发现既有 Windows 环境 `D:\miniconda3\envs\agent-dev`，元数据为 Python 3.12.13、pytest 9.1.1、LangGraph 1.2.9。与 requirements 的 FastAPI/Pydantic/OpenAI 版本有差异，尚未执行该解释器；已向用户请求跨操作系统调用许可。

## 源码核对

- `AgentOrchestrator.process()` 先识别再路由；普通角色 gather，并无 Planner、任务数据依赖或检查点。
- `extract_tool_calls()` 将非法 JSON 归一成 `{}`；`_dispatch()` 将权限以外的大部分异常归为缺参数。
- `_run_agent()` 还按工具名复用结果，需要改为任务内规范化参数去重。
- `ActionStore` 已有 SQLite 事务、TTL、参数哈希与幂等记录；确认执行结果未知时的对账仍需加强。
- `MemoryManager.update_profile()` 每轮调用模型，缺少来源校验，并先删除旧画像再添加，存在失败丢失风险。

## 实施顺序

1. 固定基线、输入／结果契约；异常分类、解析修正和确定性回复出口。
2. 核定依赖兼容后，将 Planner、DAG 调度、持久化等待及预算放入 LangGraph 外层流程；保留直接路由。
3. 局部重规划、版本冲突与失效传播、预约确认核对及 API／前端交互。
4. 长期记忆门槛、来源与原子更新保护。
5. 故障注入、TODO 第 4 节验收、回归、构建及可复现示例。

## 当前验证状态

用户在环境确认时指示“暂停先”，目标已按要求暂停。未获 Windows 环境执行许可，未执行该解释器；尚未运行 Python 基线、安装依赖或修改运行时代码。未把环境阻塞当作测试失败，未勾选任何能力完成项。未发生付费模型或外部医院调用。恢复需用户指示，跨操作系统执行仍需明确许可。
