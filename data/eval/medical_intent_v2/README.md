# 医疗细粒度意图 V2 数据集

18 个输出状态：15 个可路由细类，以及 `urgent_handoff`、`needs_clarification`、`out_of_scope`。共 540 条开发者合成表达，按场景族划分为 360 条开发集和 180 条保留集；同一场景的两种措辞只出现在同一分区。`manifest.json` 保存逐文件 SHA-256 和类别计数。

`dev.jsonl` 可进入 V2 few-shot 和向量模板；`heldout.jsonl` 只供显式评测，不参与提示词、规则和阈值调整。V1 的 240/120 数据及其历史成绩保持独立。样本没有患者来源、第二标注者或临床审核；当前正式集均为单一细类任务，多意图指标只能说明单标签情况，不能证明跨任务覆盖。

运行开发集离线诊断：

```powershell
& 'D:\miniconda3\envs\nexusops-dev\python.exe' -m scripts.run_medical_intent_v2_evaluation --split dev --offline --output data/eval/results/medical_intent_v2_dev_offline.json
```

真实模型评测需显式选择配置文件，例如增加 `--config-file .env` 并去掉 `--offline`。报告写入被 Git 忽略的 `data/eval/results`，拒绝覆盖；保留集应在版本固定后单次评测。
