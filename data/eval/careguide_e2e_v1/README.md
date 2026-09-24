# CareGuide 端到端工作流评测

数据集共 120 条合成工作流，开发集和固定回归集各 60 条，覆盖知识问答、合成预约和安全边界。

评测分为两层：

- 确定性检查：路由、工具参数、确认前零写入、幂等性以及 Slot/Appointment 状态一致性。
- LLM Judge：使用独立模型评估任务完成、证据支持、流程准确性、安全边界和对话质量。

开发集结果为确定性任务 100%、Judge overall 0.9273；固定回归集结果为确定性任务 88.33%、Judge overall 0.9015。模型、配置和数据哈希见 `freeze_manifest_v4_provider.json`，结果摘要见 `frozen_provider_heldout_result_v4.json`。

用例不包含真实患者数据，也没有经过临床审核；结果只用于衡量项目工作流。
