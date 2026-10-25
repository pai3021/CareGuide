# 编排与可靠性验收矩阵

默认 unified 入口及共享执行器的验收场景。以下结果来自离线真实 worker、可控模型和故障注入；真实提供方规划质量尚未评测。最终结果与历史离线对比样本见 [verification.json](verification.json)，运行说明见 [implementation.md](implementation.md)。

MiMo v2.6 回答流式改造后，完整后端回归 **416 项通过**，其中回答流式新增 **16 项**；前端 Node **12 项通过**，生产构建与真实 Chrome 脚本 API 工作流验证通过。提供方响应由 MockTransport 模拟，未调用真实付费服务。

## SSE 前后端

| 场景 | 实际结果 | 主要证据 |
| --- | --- | --- |
| JSON／SSE 共用执行链 | 相同真实 worker 和预算；SSE 的最终结果与 JSON 契约一致，只保存一次会话 | test_chat_stream.py::test_stream_and_json_use_same_workers_budgets_and_final_contract |
| 实时任务进度 | worker 被同步屏障阻塞时，已收到 running 事件，尚无 result；浏览器此时显示进度且不能操作最终卡片 | test_chat_stream.py::test_real_progress_precedes_worker_completion_and_disconnect_preserves_execution；browser-workflow.mjs 的 sse_progress_before_final |
| 断线与停止接收 | ASGI disconnect 结束传输，后台仍按预算执行并落库，任务尝试／工具不重复；浏览器保留会话，不自动重发 | test_chat_stream.py::test_asgi_disconnect_stops_delivery_without_cancelling_persisted_task；Chrome sse_stop_keeps_session_without_resubmit |
| 并发用户 | 每个请求独立事件接收器，工作流／消息不串用户 | test_chat_stream.py::test_concurrent_users_receive_only_their_own_progress |
| 心跳和背压 | 执行阻塞时可发送注释心跳；队列有界，慢接收可丢中间进度但不丢连接中的最终结果 | test_chat_stream.py 的 heartbeat／backpressure 测试 |
| 开流前／中途错误 | 缺字段仍 HTTP 422；中途 404／409／503 发送脱敏 error 帧，无私有异常内容 | test_chat_stream.py::test_error_frames_and_pre_header_validation_are_sanitized |
| 补参、身份和确认 | 通过 SSE 补参；跨用户 404、旧版本 409；重复显式确认只创建一次 | test_chat_stream.py::test_stream_wait_resume_versions_and_confirmation_remain_authoritative |
| 前端协议 | UTF-8 任意分片、CRLF、多行 data、重复／倒序 id、EOF、abort、大小限制及非 SSE 响应均有明确处理 | frontend/tests/sse.test.js；原 workflow.test.js |
| 模型正文增量 | 真实 OpenAI SDK 消费 MockTransport SSE；模型尚未结束时，前端已收到正文 delta，最终回复与记忆／trace 一致 | test_answer_streaming.py::test_real_sse_delta_precedes_model_finish_final_memory_and_trace_agree；Chrome answer_deltas_before_final |
| 回答失败回退 | 引用无效、无结束标志、截断、工具调用、超时或业务版本变化时撤回草稿，不丢已完成查询结果 | test_answer_streaming.py 的 bad_answers／deadline／concurrent_workflow_change 测试 |
| 正文背压和预算 | 正文增量按序传送；断线释放等待，撤回可替换积压草稿；一次完整回答计一次模型调用，缺失 usage 不编造 | test_answer_streaming.py 的 backpressure／reset／usage／budget 测试 |

当前支持阶段／任务进度与真实模型正文增量，最终回复做引用和结构检查，不能据此证明语义忠实或医学正确。浏览器验证使用实际 Chrome 和生产构建 Vue，后端是脚本 API；真实 SDK／FastAPI／编排器另由上述后端测试覆盖。没有据此声称线上网络可靠性、模型质量或费用收益。

## 业务意图 embedding

| 场景 | 实际结果 | 主要证据 |
| --- | --- | --- |
| 默认工厂实际接线 | 复用硅基流动模型／地址配置，使用独立业务 query instruction；真实 OpenAI SDK＋MockTransport 的近邻进入主 Planner，一次生成式决策 | test_intent_semantic.py::test_production_factory_passes_real_embedding_candidates_to_one_planner |
| 规则早退出／无主模型 | 明确医生查询 0 次模型调用；无主模型的未知请求不增加远程 embedding 费用 | 同文件::test_no_chat_planner_means_no_remote_embedding_cost；生产工厂测试 |
| usage 与共享预算 | 示例批次、查询各按真实 SDK 请求计数；提供方 token 统计进入预算；模型发现独立新流程时不重置既有 embedding 成本 | 同文件::test_real_sdk_business_bank_usage_cache_scope_and_no_authority；test_embedding_cost_survives_model_discovered_new_workflow |
| 缓存与并发 | 同用户／会话命中查询缓存；不同用户重新查询；返回独立副本；并发只生成一次公共示例矩阵 | 同文件的 cache_scope／concurrent_requests 测试 |
| 索引来源与故障 | 预构建索引核对完整配置／示例指纹；零向量、维度错误和非有限数拒绝；失败后有限冷却，不伪造语义向量 | 同文件的 prebuilt_index／invalid_vectors 测试 |
| 超时和预算不足 | 缺账本或额度耗尽不派发 SDK；HTTP 超时受实际请求剩余预算收紧；超时不自动重试、不抹掉费用，错误诊断不含私有响应文本 | 同文件的 exhausted_or_missing_budget／timeout／http_timeout 测试 |
| 高相似度与否定 | 即使“准备预约”示例相似度为 1，当前“暂时不要预约”仍不能由候选生成执行计划 | 同文件::test_high_similarity_is_only_a_reference_and_cannot_authorize_negated_booking |
| 独立评测入口 | 26 条另行编写样例，无参考库逐字重复；本轮只测原字符基线，真实稠密语义召回未运行 | scripts/evaluate_intent_semantic.py；intent_semantic_cases.json；verification.json |

SDK 受控向量仅用于验证传输、计费和安全契约，不用于计算真实 embedding 准确率。字符基线候选动作覆盖与语义路由准确率不是同一指标；尚未启用相似度直接执行。

## 统一理解与 Orchestrator–Workers

| 场景 | 实际结果 | 主要证据 |
| --- | --- | --- |
| 默认生产工厂/API | 实际建立 UnifiedRuntime 并复用本地分类适配器，不调用旧识别器的模型分支 | test_unified_runtime.py::test_default_factory_wires_existing_classifier_and_does_not_call_legacy_recognizer |
| 明确单任务／医生＋药品组合 | 生成统一 PlanProposal；本地初始规划 0 次；参数与证据分别绑定 | test_unified_understanding.py；test_unified_runtime.py::test_local_composition_is_durable_and_brief_isolated；test_unified_examples.py |
| 医生→号源→预约 | 版本化模板；显式选择医生、号源后准备；重复确认只创建一次 | test_unified_runtime.py::test_template_selection_confirmation_and_resume_do_not_need_planner；template 命令 |
| 未知剩余目标、关系与否定 | 整句交主 Agent／澄清，不只执行规则覆盖的半句；先别预约不生成准备 | test_unified_understanding.py；test_unified_synthesis_and_sources.py |
| 原 RoBERTa 与语义候选 | 标签／分数／截断保留为线索；高分不授权，多候选不拆成多目标；故障与标签不匹配显式降级 | test_unified_understanding.py；真实 CPU 适配器烟测摘要 |
| 补规格同时新增医生 | 一个 RequestDecision；成功则同时应用，非法新增导致整包回滚 | test_unified_runtime.py::test_dynamic_mixed_answer_and_addition_with_real_worker_and_safe_synthesis_fallback；test_input_and_change_package_rolls_back_completely_on_invalid_addition |
| 历史指代与来源 | 只接受精确用户原话；助手不能授权；历史预约不能覆盖当前取消；相对日期用原始时间，无时间不重新锚定 | test_unified_synthesis_and_sources.py |
| worker 上下文与长成果 | 独立简报、空历史，无 sibling 泄漏；长成果使用校验归属／版本／完整性的引用 | test_unified_runtime.py；test_unified_synthesis_and_sources.py::test_large_worker_result_uses_owned_versioned_reference_and_preserves_trace |
| 跨任务综合 | 只选对应证据原文；非法引用退回确定性结果，保留状态与成功资料；共用模型预算 | test_unified_synthesis_and_sources.py::test_synthesis_only_selects_verified_excerpts_and_budget_is_shared；混合变更测试 |
| 单次理解／规划协议 | 实际 AsyncOpenAI SDK＋MockTransport 检查 schema、worker 职责、usage；JSON 修正有限 | test_unified_model_protocol.py |
| 预算与模式回退 | 模型发现独立新请求后共享已计费账本；旧入口和直接确认同样读取预算别名；耗尽后结构化取消无需理解调用 | test_unified_runtime.py 的 budget／cancel／mode_fallback 测试 |
| 新查询、急症与进程恢复 | 旧等待不被新查询修改；急症先于理解中止分支；start/resume 两个进程累计工具 1→2、模型 0 | test_unified_runtime.py；test_unified_examples.py::test_wait_command_restores_in_another_process_without_repeating_success |

以下矩阵覆盖共享执行器及旧 langgraph 入口；旧 API 夹具显式选择 langgraph，保留原有行为断言。新增 unified 测试另外覆盖默认入口与新理解／委派链路，不用旧模式结果冒充新理解模型的正确率。

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

2026-10-25 业务意图向量接入回归：pytest **400 项全部通过**，新增语义检索场景 15 项；前端 Node **10 项通过**，Vite 生产构建、真实 Chrome 工作流／SSE 交互和公开配置模板 Compose 检查均通过。26 条合成样例的原字符基线已运行；受控 SDK／向量用于契约验证，未当作真实语义效果。没有调用真实付费 embedding／生成式模型，也未更改冻结数据、原索引或运行服务。原 RoBERTa 的 CPU 烟测、真实服务集成和 Nginx 检查保留各自先前日期的证据。详细计数见 verification.json。
