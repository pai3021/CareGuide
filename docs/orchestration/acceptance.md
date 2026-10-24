# 编排与可靠性验收矩阵

2026-10-21 验收场景。以下结果来自可控模型和故障注入；真实提供方规划质量尚未评测。最终结果与离线对比样本见 [verification.json](verification.json)，运行说明见 [implementation.md](implementation.md)。

## 任务执行与恢复

| 场景 | 实际结果 | 主要证据 |
| --- | --- | --- |
| 等待补参或确认时发起新查询 | 执行新查询，旧任务及确认状态不变；仍可恢复旧任务 | test_orchestration_api.py::test_new_query_during_pending_workflow_preserves_old_task_and_can_resume_it |
| 等待中收到含糊输入 | 明确澄清归属，不改旧任务、不消耗旧任务预算 | test_orchestration_api.py::test_ambiguous_pending_reply_explicitly_clarifies_without_changing_task_or_budget |
| 无号源后改后天 | 模型擅改日期被拒；用户明确同意后只更新号源，药品证据保留 | test_orchestration_acceptance.py::test_no_slots_cannot_silently_change_date_but_explicit_user_change_preserves_drug_result |
| 药品缺规格、号源可执行 | 号源完成；补规格后只执行药品任务 | 同文件::test_missing_drug_spec_does_not_block_ready_slots_or_repeat_them_on_resume |
| 两个药品的禁忌 | 实际 Planner 接口生成两项，同一工具分对象执行，证据章节互不混用 | 同文件::test_two_drug_contraindication_tasks_execute_real_planner_and_keep_sections_isolated |
| 消化内科改心内科 | 失效医生及下游，保留药品结果；旧回复被拒 | test_workflow_store.py::test_local_replanning_invalidates_only_affected_results_and_descendants；test_orchestration_api.py::test_http_local_replan_preserves_unaffected_wait_and_rejects_old_answer |
| 准备后换号源、点旧确认 | 后端拒绝旧操作，无写入；新 action 重新确认 | test_confirmation_reliability.py::test_changed_slot_expires_old_confirmation_before_backend_write；Chrome versioned_change／old_confirmation_disabled |
| 旧查询未返回就修改 | 无法取消的旧结果只归档，不覆盖新任务；无关执行中分支保留且不重查 | test_orchestration_acceptance.py::test_http_change_during_inflight_query_discards_late_old_result；test_local_change_keeps_unaffected_inflight_task_and_does_not_repeat_it |
| 号源服务持续故障 | 单分支有限重试，继续交付成功药品资料 | test_dag_scheduler.py::test_transient_failure_retries_only_failed_branch_and_delivers_other_results；partial 示例 |
| 模型连续相同计划 | 一次无进展变更即停止，不追加工具派发 | test_orchestration_acceptance.py::test_repeated_identical_replan_stops_after_one_without_extra_tool_dispatch |
| 已预约后恢复或重复确认 | 复用已完成记录，不重复创建；完成写不能被取消计划抹掉 | test_orchestration_api.py::test_http_dependency_selection_prepare_confirm_and_resume_never_duplicate_booking；test_confirmation_reliability.py::test_unrelated_plan_change_keeps_confirmation_and_repeat_is_idempotent |
| 取消分支／新增查询 | 取消沿依赖传播；新增独立任务，其他结果保留；无模型额度仍能取消 | test_orchestration_acceptance.py::test_user_cancel_and_add_are_local_and_preserve_completed_doctor_query；test_explicit_cancel_works_with_exhausted_model_budget_and_no_planner_call |
| 后续出现急症 | 普通未执行分支停止，进入升级，不再派发工具／规划 | 同文件::test_emergency_followup_preempts_waiting_workflow_before_any_ordinary_call；原直接急症回归 |
| 进程重启后补充或确认 | 独立 OS 进程从 SQLite 恢复，成功分支不重查、预算不清零；新服务实例对账未知写结果 | test_orchestration_examples.py；test_langgraph_checkpoint.py；test_confirmation_reliability.py::test_unknown_outcome_is_persisted_across_service_restart_and_never_reexecuted；Docker wait start/resume 与 FHIR GET 对账 |

## 异常与预算

| 场景 | 实际结果 | 主要证据 |
| --- | --- | --- |
| 条件完整、模型非法 JSON | 明确 format_error，有限修正；耗尽停止，零猜参执行 | test_runtime_failures.py::test_model_format_error_is_repaired_without_asking_user_for_known_fields；test_format_repairs_are_bounded_and_do_not_execute_a_tool；test_model_planner.py |
| 确实缺药品规格 | 独立追问，已完成分支保留；单任务缺参也可无 Planner 持久恢复 | test_dag_scheduler.py::test_wait_then_reopen_and_resume_does_not_repeat_independent_completed_query；test_orchestration_api.py::test_simple_missing_parameter_promotes_direct_route_without_planner_and_survives_reopen |
| 一成功一超时 | 统一出口为部分完成，来源和失败项均保留 | test_runtime_failures.py::test_composite_partial_result_keeps_successful_source_and_reports_failed_branch；test_timeout_classification_reaches_dispatcher |
| 不回复、过期后返回 | 请求已结束；旧答案 410，已完成查询保留，旧确认不可复用 | test_orchestration_acceptance.py::test_expired_wait_rejects_late_answer_and_preserves_completed_query；test_orchestration_api.py::test_http_wait_restore_budget_and_reject_stale_or_other_owner_event |
| 多分支预算将尽、恢复继续 | 原子预留不超额；恢复沿用旧额度、保留部分结果；收尾禁止新增 | test_workflow_budget.py；test_dag_scheduler.py::test_parallel_branches_share_hard_tool_limit_and_keep_completed_result；test_orchestration_acceptance.py::test_resume_does_not_reset_exhausted_shared_tool_budget；test_closing_budget_rejects_added_branch_and_keeps_previous_result |
| 预约超时、可能已完成 | 结果待核对；重启后 GET 查操作标识，只核对不重发 | test_confirmation_reliability.py::test_committed_write_with_lost_response_is_reconciled_without_resubmit；真实 FHIR 集成检查 recovery_methods=[GET]、重复确认网络请求 0 |
| 以后上午、本次下午 | 长期有来源；本次过滤下午号源，不改长期；明确更正才更新 | test_profile_memory.py::test_durable_morning_survives_temporary_afternoon_and_same_message_is_not_reextracted；test_preference_query.py::test_persistent_morning_is_reusable_but_current_afternoon_query_takes_priority；真实 Redis／Chroma 集成 |
| 无候选、非法提取、存储失败、并发 | 无候选零提取；非法不落库；失败保旧，CAS 冲突显式报告；业务回复不被记忆阻塞 | test_profile_memory.py 的准入／来源／更正／删除／过期／冲突测试；test_preference_query.py::test_stalled_memory_cannot_hide_completed_business_reply |

## 其他关键契约

- `test_model_planner.py`：实际 AsyncOpenAI SDK＋httpx.MockTransport 验证请求 schema、usage、HTTP 故障及隐式重试关闭；没有真实网络费用。
- `test_orchestration_api.py`：实际生产工厂及 FastAPI 路径，完整选择／准备／确认、归属隔离、版本冲突、直接路径和缺参恢复。
- `test_task_bindings.py`、`test_orchestration_contracts.py`、`test_workflow_store.py`：环、权限、引用版本、单候选不自动选择、参数来源、不授权扩大范围、不抹掉真实写事实。
- `test_dag_scheduler.py::test_downstream_starts_after_its_own_parent_without_waiting_for_slow_independent_task`：事件同步证明独立并行、下游无需等慢的无关分支；不是仅断言结果列表。
- `test_source_expiring_after_answer_never_dispatches_old_selected_doctor[0/1]`：用户已选但派发前上游过期，预算充足刷新后重新选；无额度阻塞退出，不死循环、不发旧参数。
- `test_preference_query.py`：上海时区 UTC 号源过滤、午夜跨轮相对日期、当次否定偏好追问、记忆卡住后的业务交付。
- `test_confirmation_reliability.py`：并发确认只一次写、变更与确认竞态、身份／会话隔离、HTTP 200 缺事务证据不报成功。
- 前端 Node 3 项＋Chrome 7 项：字段补参、医生／号源选择、版本变更、旧确认禁用、只确认当前 action、刷新恢复。浏览器使用脚本 API，与实际服务集成分别报告。

## 对比、恢复与维护取舍

原直接路由 main 和新实现使用相同确定性角色、固定单标签，模型返回 oracle JSON，5 个小样例各 10 次；逐次时延、证据范围和工具／Planner 次数在 verification.json。此对比只表明代码能保留被单标签遗漏的任务，并量化该夹具下的额外开销，不能说明模型真实拆解能力。

LangGraph 提供外层条件循环、interrupt 和持久 checkpoint；任意 DAG 的就绪判断、输入绑定和局部失效仍由项目代码显式实现。相比只用自研调度器，增加 checkpoint 依赖、独立 SQLite 文件和图恢复维护；收益是无需另造中断／恢复协议。没有测量开发工时或宣称维护成本百分比下降，直接模式保留作业务回退。

重规划与未知写结果记录真实次数；相同计划停止、不支持隐式扩大日期或医生范围。五类示例为依赖、局部重规划、跨进程等待恢复、部分失败、记忆更新，均可通过脚本重现。验证区分离线示例、浏览器脚本 API 与真实服务集成。

离线异常统计按固定案例单独记录：相同计划案例消耗 1 次无效重规划、总工具 2 次且无额外派发；partial 示例为号源 2 次尝试、1 次重试、1 次无进展重规划，药品只执行 1 次；跨进程 wait 的累计工具从 1 到 2、累计模型始终 1，恢复进程新增模型调用 0；预算耗尽恢复案例工具累计始终 1。以上是各夹具实际计数，不是生产发生率。

本机发布检查发现 main 与旧索引 manifest 指纹不同，但 10,015 条真实检索输入哈希一致。经过严格来源和向量校验，禁网复用真实缓存到新名称并恢复原 Hybrid。原索引和未发布半成品均保留，未修改冻结资料。新增 test_index_reuse.py 7 项覆盖禁止覆盖、内容／模型／维度不匹配、非有限数、半成品拒绝。真实 Chroma 全量回读与已知文档 Hybrid 烟测另外记录；不冒充新问题的真实 embedding 质量评测。

2026-10-24 回归：pytest **335 项全部通过**，58.03 秒；前端 Node **3 项通过**，Vite 生产构建通过。新增场景覆盖旧流程等待时的新查询与旧任务恢复。真实模型质量／费用评测未运行，历史服务集成和浏览器结果不作为本次重新执行的结果。详细计数见 verification.json 的 release_regression。
