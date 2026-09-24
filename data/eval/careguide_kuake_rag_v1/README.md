# CareGuide KUAKE RAG V1 评测集

全部查询和gold段落来自KUAKE-IR train；按唯一gold文档固定拆分为8,000 train、1,000 dev和1,000冻结test。三组查询检索同一个10,015篇运行时语料库，官方dev/test不进入主评测。

V1已在`freeze_manifest.json`记录代码、语料、模型和参数哈希，然后于2026-09-23运行冻结test一次。
最终MRR@10、Recall@10和nDCG@10为0.4857、0.668和0.5292，审计记录见
`frozen_test_result.json`。该test不再用于调参或重复报告。
