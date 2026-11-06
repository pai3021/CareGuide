# 业务动作标签与本地分流评测

本轮将业务动作与会话操作分开标注，并测量旧分类器是否给现有规则增加分流覆盖。数据、模型权重、分流阈值均未根据结果修改。新增的分类器关闭开关用于后续对照；当次分类器默认 `auto`，检索调度默认 `on_demand`。之后项目默认调度改为 `parallel`，本页结果仍属于原报告快照。

## 数据与标签

[business_intent_cases.json](../../careguide/evaluation/business_intent_cases.json) 含 64 条人工编写的合成表达，分为医生、号源、说明书、其他只读查询、交易、组合任务、会话跟进、边界／直接回答八组，每组八条。它与参考库、原六例和本地十二例探针无逐字重合，未用于训练或阈值选择。数据在首次真实推理前固定，哈希见 [冻结记录](business-intent-eval-v1.lock.json)。

该集合由开发者标注，未经独立复核，不是真实用户数据或总体准确率基准，也不能因为没有逐字重合就声称具备语义独立性。

标签包含四类信息：

| 字段 | 含义 |
| --- | --- |
| `operation` | new、change、answer、cancel、clarify、direct、boundary；与动作类别分开 |
| `actions` | 对齐程序能力目录，支持多目标；不是沿用 appointment_action 等粗类别 |
| `arguments` | 各动作需要匹配的参数子集，相对日期绑定固定消息时间 |
| `local_eligible`／`forbid_local_plan`／`forbidden_actions` | 简单只读目标、必须交出轻量模块的条件，以及不得生成的动作 |

分类权重和原 CareGuide-13 标签均未改动；业务动作标注是新评测目标，不代表已有模型已经训练成业务动作分类器。未来训练需要另建训练／验证集，不能把本集合用于拟合后再作为独立评测。

## 方法与实际结果

真实本地 RoBERTa 在 CPU 上对 64 条原文各预测一次。推理只接收文本，标准答案仅用于评分；之后将这批真实预测等同复用到各分流配置中。无语义候选夹具、无远程客户端、无 Planner 或医院工具。模型和分词器哈希、逐例预测及结果见 [JSON 报告](business-intent-local-20261105.json)。

| 指标 | 仅规则 | 规则＋旧分类器 |
| --- | --- | --- |
| 本地生成计划 | 9 | 9 |
| 动作集合匹配 | 9 | 9 |
| 原始计划参数子集也匹配 | 7 | 7 |
| 14 条简单只读目标中，动作及原始参数匹配 | 5 | 5 |
| 新增候选准入计划 | 0 | 0 |
| 不应本地处理却生成计划 | 0 | 0 |

两条原始参数不匹配来自医生→号源模板：时间参数在 Worker 的 `prepare()` 阶段按原文和来源时间绑定，而原始 `TaskSpec.arguments` 留空。本轮只评分原始计划，不执行该后续阶段；这两条不能解释成实际工具查询错误或执行失败。

14 条简单只读目标中，旧分类最高标签经现有动作族映射后覆盖 4 条，达到当前强候选筛选条件的覆盖为 0 条。这是粗类别动作族的覆盖统计，不是细粒度动作准确率。全量最高标签分布为：其他 33、appointment_action 25、注意事项 3、emergency 2、就医建议 1。

本轮没有观察到旧分类器带来的新增正确分流。该结果只适用于此人工集合与既定准入策略，不能推广成分类器毫无价值，也没有测量线上延迟或费用。重放耗时不作为原生推理时延报告。

## 可切换项与未运行项

`CAREGUIDE_INTENT_CLASSIFIER_MODE=disabled` 在 unified 轻量模块中关闭旧分类源，可与已配置的语义候选单独对照；`auto` 保持现有权重路径和降级行为。该开关不改动旧 direct/langgraph 路由的分类配置，也不会自行部署。

`evaluate_intent_ablation.py` 增加了 `rules_embedding_llm`，可比较规则＋Planner、加入分类、仅加入语义、同时加入分类与语义四组。该脚本的真实服务模式会产生外部请求，本轮未执行。

当前报告中仅语义、分类＋语义按需、分类＋语义并行三项明确标为 `not_run_embedding_unavailable`。此前授权的 40 次请求已用完，本轮没有追加。不能由本轮推断语义独用更好，或宣布按需／并行哪一种最优。当次评测没有修改默认策略；随后按架构表达选择的 `parallel` 默认值不属于本次效果结论。

## 无网络复现

```bash
# 无权重时只运行规则，模型项标为未运行。
python scripts/evaluate_business_intents.py --output tmp/business-intent-rules.json
# 使用已存在的本地权重，禁止自动下载；生成新文件，拒绝覆盖。
python scripts/evaluate_business_intents.py --model-path /path/to/careguide-intent-roberta-v1 --output tmp/business-intent-local.json
```

本轮完整离线回归为 618 passed；将指标名称明确为“原始计划注释匹配”后，又通过 13 项评测专项测试。下一阶段先独立复核标签、建立单独训练／验证数据，再测仅语义与联合候选的真实质量、升级率和调用代价，之后决定是否重训分类器或调整调度默认值。
