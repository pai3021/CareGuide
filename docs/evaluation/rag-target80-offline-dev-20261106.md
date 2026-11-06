# 最终Recall@10超过80%的后续优化：离线阶段

本轮目标仍是改善重排后的最终Top10，而非将候选覆盖率作为80%的达成证据。目前固定100条开发样本的已验证结果仍是72%，目标尚未完成。本轮新增外部模型请求为0，原模型配置、语料、gold与冻结test均保持不变。

本报告记录离线阶段。后续在追加授权后已完成200次真实重排请求，Top80未改善最终命中，结果见[真实对照报告](rag-target80-rerank-dev-20261106.md)；质量优先配置继续保留Top40。

## 失败发生在哪里

沿用同一100条样本，复用真实完整重排响应，逐阶段核对：

| 阶段 | 0.6B Embedding＋4B重排Top40 | 4B Embedding＋4B重排Top40 |
| --- | ---: | ---: |
| gold不在两路各Top100并集 | 13 | 10 |
| 已在并集，但被Top40截断 | 7 | 5 |
| 已入池，未进最终Top10 | 8 | 13 |
| 最终命中 | 72 | 72 |

这些是观测到的损失阶段，不直接等于模型语义能力的原因判断。文本审阅显示，部分query与gold涉及不同问题维度、存在未给出的上下文，也有需要继续检查的实体与症状匹配问题。不删除这些样本，不根据gold改写输入，不把相关性审阅当成临床审核。

## 已完成的零API实验

| 方案 | 选择方式 | 验证观察 | 决定 |
| --- | --- | --- | --- |
| 4B Embedding的融合参数 | 45组预定义配置，仅调参500选Top40方案 | 验证500覆盖83.6%→83.0% | 不采用 |
| 中文BM25分词与长度归一 | 8组：二字基线、单字＋二字、二字＋三字、三者组合，各比较b=0.75/0.3；仅调参500选择 | 选中单字＋二字、b=0.75；BM25 Top100覆盖57.6%→71.6%，两路并集91.0%→92.8%，但RRF Top40覆盖83.6%→83.2% | 记录词法候选收益，暂不替换整条链路 |
| 各路候选保留配额 | 每个固定预算比较原/新词法、Dense/RRF填充、0/5/10/20条词法保留，仅调参500选方案 | Top40验证83.6%→82.6%；Top60验证86.4%→86.6%，仅净多覆盖1条 | 不据此宣称最终收益 |
| MMR去冗余 | 5个预设权重，仅调参50选择，再验证50 | 原Embedding选中不做MMR；新Embedding选中权重0.9，验证最终Recall仍70%，全100仍72% | 不采用 |
| ANN近似索引与精确余弦 | 同一4B文档与query向量，全部1,000条dev精确对照，无参数选择 | 验证500的Dense Top100覆盖89.4%→91.0%，RRF Top40覆盖83.6%→84.4% | 提供可选精确检索，等待最终重排验证 |

词法实验首先在全部1,000条开发query上复现原BM25的Top100文档ID与数值分数，再评估替代分词。基线一致性未通过时脚本拒绝继续。MMR使用已缓存的真实文档向量与完整重排响应，不使用gold构造相似度。

两个需要修正的假设：

1. 当前一万条KUAKE文档的标题为通用标题加编号，关键词统一为`KUAKE-IR`。因此，虽然召回使用标题、关键词和正文而重排只用正文，补上这些字段在此数据上缺乏信息收益依据。
2. 更强词法候选并不自动意味着Top40更好；当前固定融合会丢失一部分候选。新的词法收益和最终收益必须分开报告。

聚合原始数据及来源哈希见[JSON报告](rag-target80-offline-dev-20261106.json)。所有调参和验证数据均已有开发使用，属于探索性开发证据。

## 向量索引的独立损失与可选精确检索

使用已缓存的10,015条真实4B文档向量与1,000条query向量，计算完整余弦相似度；不重编码、不改检索指令。ANN返回的Top100与精确Top100平均重合95.213%，这是搜索集合的近似程度，不是gold覆盖率。

500条内部验证中，精确搜索让Dense Top100的gold覆盖从89.4%到91.0%，两路并集从91.0%到92.2%；RRF Top40从83.6%到84.4%，新增覆盖5条、丢失1条。调参500的Top40也从85.0%到86.2%。说明当前索引近似搜索确有小幅候选损失，但这不能直接解释全部最终失败。

新增可选运行配置：

```dotenv
CAREGUIDE_DENSE_SEARCH=exact
```

默认仍为`ann`。精确模式从身份已校验的现有Chroma集合加载文档向量到内存，查询只编码一次，先按业务范围与有效期筛选文档，再计算余弦；不建库或修改索引。它需要额外内存，计算量随有效候选规模增长。本轮批量纯计算耗时不当作线上延迟或生产P95。

真实4B索引上的3条原生回放与离线精确Top80完全一致，见[回放记录](exact-dense-native-replay-20261106.json)。未向改动后的候选池套用旧重排响应，最终Recall@10仍需真实请求验证。下方已经固定的205次请求计划仍沿用原ANN候选池，未在等待授权期间变更计划。

原始精确分数已保存为[公共缓存](../../data/eval/cache/rag-recall-scores-qwen4b-exact-dev-20261106.json.gz)。已有本地向量检查点时可零API重算：

```powershell
python -m scripts.audit_rag_dense_index `
  --plan data/eval/cache/rag-embedding4b-plan-dev-20261106.json `
  --baseline-scores data/eval/cache/rag-recall-scores-qwen4b-dev-20261106.json.gz `
  --vector-db data/source_cache/qwen-embedding-4b-1024-20261106/vectors.sqlite3 `
  --output tmp/exact-audit-replay.json `
  --exact-cache tmp/exact-audit-scores-replay.json.gz
```

## 已固定并执行的真实对照

固定4B Embedding、原两路RRF和Top80候选，对照三组：

1. 4B Reranker＋原英文直接回答指令：与已有Top40结果比较候选池变化。
2. 4B Reranker＋检索相关性指令：在相同Top80下验证任务指令变化。
3. 8B Reranker＋相同检索相关性指令：只比较模型大小。

新指令允许回答问题的部分段落与相关后续回复，不要求完整独立回答；不加入gold信息，不把与query无关的文档定义为相关。实验动机是检索资料与用户问题的表达形式不完全相同，效果必须实测。

Qwen官方展示了可配置的重排任务指令，硅基流动文档列出4B/8B对`instruction`的支持：[Qwen模型卡](https://huggingface.co/Qwen/Qwen3-Reranker-8B)、[接口文档](https://docs.siliconflow.cn/docs/api/rerank-post)。支持接口不代表本项目中一定有提升。

先在固定调参50条运行3组，共150次，再按调参Recall@10、MRR@10选择方案，平分优先4B及原指令。只将所选方案运行另外50条，正常总计200次；最多205次尝试，预留5次失败后手动恢复，不启用自动重试。每次最多80篇公开候选正文，正常正文约204万字符，含失败恢复的上限约212万字符。仅调用Rerank，不调用Embedding、生成式LLM或医院工具。

Top80在固定100条上的覆盖为88%，调参50为90%，内部验证50为86%；这是本候选池的上限，不是最终成绩。若未达最终80%以上，继续保留原结果并定位损失，不重新抽样、降低难度或变更评分口径。

固定计划见[计划JSON](../../data/eval/cache/rag-target80-rerank-plan-dev-20261106.json)，已在授权后完成。下面的命令只做离线检查，不读取服务密钥或发请求：

```powershell
python -m scripts.compare_rag_rerank_scaling run `
  --plan data/eval/cache/rag-target80-rerank-plan-dev-20261106.json `
  --scores data/eval/cache/rag-recall-scores-qwen4b-dev-20261106.json.gz `
  --responses data/source_cache/rag-target80-rerank-20261106/responses.json `
  --output data/source_cache/rag-target80-rerank-20261106/result.json
```

## 零API复算

词法与候选配额实验直接从公开分数缓存和原始语料复算：

```powershell
python -m scripts.evaluate_rag_lexical_variants `
  --cache data/eval/cache/rag-recall-scores-qwen4b-dev-20261106.json.gz `
  --output tmp/lexical-replay.json `
  --selected-cache tmp/lexical-selected-replay.json.gz

python -m scripts.evaluate_rag_candidate_quotas `
  --baseline-cache data/eval/cache/rag-recall-scores-qwen4b-dev-20261106.json.gz `
  --lexical-cache tmp/lexical-selected-replay.json.gz `
  --output tmp/quotas-replay.json
```

去冗余可用公开的候选相似度缓存复算，不需要本地向量索引或模型服务：

```powershell
python -m scripts.evaluate_rag_diversity `
  --plan data/eval/cache/rag-embedding4b-plan-dev-20261106.json `
  --old-scores data/eval/cache/rag-recall-scores-dev-20261105.json.gz `
  --new-scores data/eval/cache/rag-recall-scores-qwen4b-dev-20261106.json.gz `
  --responses data/eval/cache/rag-embedding4b-rerank-dev-20261106.json.gz `
  --similarity-source data/eval/cache/rag-diversity-cosines-dev-20261106.json.gz `
  --output tmp/diversity-replay.json `
  --cases-output tmp/diversity-cases-replay.jsonl `
  --similarity-cache tmp/diversity-cosines-replay.json.gz
```

离线公共缓存回放已实际运行，结果一致。相关检索、分词、配额、去冗余、实验选择、精确检索和预算协议回归共85项通过，公共Compose模板校验通过。默认策略与实际服务配置未切换，冻结test与旧0.6B索引备份哈希保持不变。
