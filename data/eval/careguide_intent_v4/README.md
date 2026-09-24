# CareGuide-13 意图数据集

KUAKE-QIC 官方 train 的 6931 条样本，加 200 条预约与 200 条急症表达。按类别和场景族拆分为 train、validation、test；同一场景族不跨 split。

精确规模为 train 5132、validation 1102、test 1097。自建数据为开发者整理的合成表达，未经临床审核。官方 QIC dev 不参与本版本切分。冻结 test 在版本定型前不得进入 prompt、centroid、规则、阈值或错误分析。

`careguide-intent-roberta-v1` 已在训练参数和 checkpoint 哈希冻结后完成一次正式 test 评测。
最终 Accuracy 为83.77%、Macro-F1为83.66%；配置、哈希和测试摘要见
`transformer_release.json`。该 release 不再使用 test 调参或重复选择 checkpoint。
