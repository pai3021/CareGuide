# 医疗框架评测开发契约 V1

这些人工编写的合成用例只验证工程机制：执行 Trace 的路由、Skill、Tool、参数、证据、受控副作用和状态；Skill 的路由后选择；已核验资料记录的检索排序与困难负例。`execution_cases.jsonl` 在无外部 LLM 的离线执行模式下运行，不能代表真实模型的端到端成功率。

`retrieval_cases.jsonl` 的正例仅引用 `data/medical_v1` 中已有记录 ID；负例故意提出资料未覆盖的问题，用于发现误召回。原始医疗语料及 240/120 意图样本不改写。这个开发集不是医生标注、临床安全评测、独立保留集，也不提供诊断或用药结论。

该集合用于验证通用的任务评分、失败归因、Skill 选择和 RAG 指标机制；不包含其他业务领域的标签或历史分数。

运行方式：

```powershell
& 'D:\miniconda3\envs\nexusops-dev\python.exe' -m scripts.run_medical_framework_evaluation --retrieval-mode bm25 --output data/eval/results/medical_framework_dev_bm25_new.json
# 已有私有模型配置及医疗向量索引时可测 Hybrid；输出文件名必须是新路径。
& 'D:\miniconda3\envs\nexusops-dev\python.exe' -m scripts.run_medical_framework_evaluation --config-file .env --retrieval-mode hybrid --output data/eval/results/medical_framework_dev_hybrid_new.json
```

执行契约可用 `--trace-bundle` 对同一用例集的真实执行 Trace 评分；默认 11 条用例使用确定性规则回退。2026-09-19 的 BM25 和 Hybrid 开发报告均为执行 11/11、Skill 6/6、资料定位正例 Recall@1 9/9；两种检索对 2 条未覆盖问题均误召回（困难负例准确率 0/2）。不能把正例 Recall@1 说成完整检索成功率；工具层仍须检查药品、规格、医院和院区的精确范围。
