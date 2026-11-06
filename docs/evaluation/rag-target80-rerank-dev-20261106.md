# Top80、检索指令与8B重排器的真实对照

本轮没有达到最终Recall@10超过80%的目标。按预先固定规则选中的4B原指令Top80方案，在完整100条开发样本上从72%降至68%，因此不采用。继续保留0.6B Embedding＋4B Reranker Top40的质量优先配置。

## 固定协议与实际用量

- 沿用同一100条公开开发query、4B/1024维Embedding分数、两路RRF常数60和Dense:BM25=4:1；候选来自ANN，未在等待授权期间切换成精确搜索。
- 先在固定调参50条上比较三组Top80：4B原英文指令、4B检索相关性指令、8B相同检索相关性指令。每组只改变指令或模型，原始query和正文不变。
- 以调参Recall@10优先、MRR@10其次选方案；只将选中的方案请求另外50条内部验证。验证结果不参与选型，不因结果较差再换方案重跑。
- 实际200次请求，全部成功，未使用预留5次失败恢复额度。持久化预算上限205，关闭自动重试；总输入Token为2,925,411。没有新增Embedding、生成式LLM、医院工具请求，也没有运行冻结test。
- 开发数据有历史使用，不称为新的独立测试或临床评测。

## 第一阶段：同一调参50条

| 模型 / 指令 / 候选池 | Recall@10 | Recall@1 | MRR@10 |
| --- | ---: | ---: | ---: |
| 4B / 原指令 / Top40，已有真实基线 | 74% | 38% | 0.4842 |
| 4B / 原指令 / Top80 | 72% | 38% | 0.4812 |
| 4B / 检索相关性 / Top80 | 72% | 38% | 0.4797 |
| 8B / 检索相关性 / Top80 | 68% | **48%** | **0.5430** |

1. 扩池没有改善这50条的最终命中：新增0条，丢失1条。
2. 指令改变新增2条、丢失2条，Recall没有净增。
3. 同一检索指令和Top80下，8B相对4B新增1条、丢失3条；但首位命中提高10个百分点、MRR提高约13.2%。不同指标的取舍需要明确，不能笼统说更大模型全部更好或全部更差。

因此按预定Recall优先规则选择4B原指令，而非根据更高MRR改动目标。

## 第二阶段及完整样本

下表均比较4B原指令，Embedding与RRF相同，仅候选预算由40变为80。

| 样本与指标 | Top40基线 | Top80所选方案 |
| --- | ---: | ---: |
| 内部验证50条：Recall@10 | **70%** | 64% |
| 内部验证50条：MRR@10 | **0.4279** | 0.4196 |
| 完整100条：候选覆盖 | 85% | **88%** |
| 完整100条：最终Recall@10 | **72%** | 68% |
| 完整100条：最终MRR@10 | **0.4560** | 0.4504 |
| 完整100条：重排输入Token | **695,127** | 1,374,677 |
| 完整100条：API平均耗时 | **1.313s** | 2.373s |

完整100条中新增命中0条、丢失4条，绝对下降4个百分点，相对下降约5.6%。2,000次query配对bootstrap的差值95%区间为[-8,-1]个百分点；内部验证50条的差值区间为[-14,0]个百分点。这些区间仅描述复用开发样本。

Top80输入Token约为Top40的1.98倍，API平均耗时约为1.81倍。API调用耗时不等于端到端延迟或生产P95，账单金额也不能只由Token总数推断。

损失分解：原Top40中有10条未进入两路并集、5条被截断、13条入池后未进Top10、72条命中；Top80对应为10、2、20、68。扩池多覆盖3条，却增加了7条入池后的排序损失，不能用88%的候选覆盖宣布达到最终80%。

## 额外的零API查询侧核对

核对到一份历史`mimo-v2.5`改写缓存，包含当前1,000条开发query、1,998条已生成改写，文件哈希与历史报告一致。仅复用缓存做BM25扩展，没有调用任何生成或Embedding服务。

原query始终保留；保守数字一致性检查后可用1,892条改写。比较改写总权重0、0.25、0.5、1，仍只用调参500的Top40覆盖和MRR选择。最后选出权重0，即不扩展，内部验证Top40覆盖仍为83.6%。非零权重可补充部分词法候选，但未形成应采用的Top40方案。数字检查不能证明改写完全语义等价。

同时核对了语料构建代码：当前KUAKE切片选择长度40至512字符的原段落，写入正文时没有二次截断；通用标题与关键词也没有额外问题信息。因此，“补回截断正文”“给重排加通用标题”缺乏当前数据上的优先级依据。

## 取舍与仍未完成的工作

本轮排除了继续单纯扩大候选池、换8B或放宽指令作为已验证收益方案。精确搜索和更强词法带来的候选覆盖收益仍保留为分阶段证据，没有套用旧重排响应生成虚构最终分数。

若继续争取超过80%，下一步应转向训练集上的领域相关性适配、困难负例，以及独立的多相关文档标注审阅。原gold和历史冻结结果继续保留。这里是后续方向，不代表已完成训练或获得新提升。

## 可复算证据

[固定计划](../../data/eval/cache/rag-target80-rerank-plan-dev-20261106.json)、[实际响应](../../data/eval/cache/rag-target80-rerank-responses-dev-20261106.json.gz)、[请求账本](../../data/eval/cache/rag-target80-rerank-requests-dev-20261106.json.gz)、[完整结果](rag-target80-rerank-dev-20261106.json)、[缓存改写对照](rag-cached-rewrites-dev-20261106.json)。均不包含密钥。

以下命令不读取私有配置、不发模型请求：

```powershell
python -m scripts.compare_rag_rerank_scaling replay `
  --plan data/eval/cache/rag-target80-rerank-plan-dev-20261106.json `
  --scores data/eval/cache/rag-recall-scores-qwen4b-dev-20261106.json.gz `
  --responses data/eval/cache/rag-target80-rerank-responses-dev-20261106.json.gz `
  --output tmp/target80-rerank-replay.json

python -m scripts.summarize_rag_rerank_scaling `
  --plan data/eval/cache/rag-target80-rerank-plan-dev-20261106.json `
  --scores data/eval/cache/rag-recall-scores-qwen4b-dev-20261106.json.gz `
  --responses data/eval/cache/rag-target80-rerank-responses-dev-20261106.json.gz `
  --ledger data/eval/cache/rag-target80-rerank-requests-dev-20261106.json.gz `
  --output tmp/target80-summary-replay.json
```

历史改写缓存也已提供[公共副本](../../data/eval/cache/rag-query-rewrites-dev-20260923.jsonl)，与原始文件SHA256一致，可独立复算词法扩展对照：

```powershell
python -m scripts.evaluate_rag_cached_rewrites `
  --scores data/eval/cache/rag-recall-scores-qwen4b-dev-20261106.json.gz `
  --rewrites data/eval/cache/rag-query-rewrites-dev-20260923.jsonl `
  --output tmp/cached-rewrites-replay.json `
  --selected-cache tmp/cached-rewrites-selected-replay.json.gz
```

已实际执行公共缓存复算，质量指标、分阶段变化和用量与发布报告一致。原0.6B索引备份、冻结test及历史结果哈希保持不变，实际服务配置没有切换。

相关检索、精确搜索、查询改写、实验选型及预算协议回归共87项通过。未使用的5次额度仅为本计划失败恢复预留，本轮没有将其用于新增实验。
