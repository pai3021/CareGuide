# RAG Embedding 0.6B 与 4B 对照

升级 Embedding 改善了候选覆盖，但本轮固定100条开发样本的最终 Recall@10 仍为72%。因此继续保留 0.6B Embedding＋4B Reranker Top40 的质量优先配置，同时单独保存4B索引；不据此替换运行配置。

## 固定条件

- 使用同一份10,015篇公开语料、1,000条开发query、原检索指令和1024维向量，只将 `Qwen/Qwen3-Embedding-0.6B` 换为 `Qwen/Qwen3-Embedding-4B`。文档与query均用新模型重新编码，不混用两种模型的向量。
- Dense与BM25各取Top100，沿用RRF常数60、Dense:BM25=4:1；BM25分数完全复用，不在本轮重新选权重。
- 重排沿用同一固定100条query、`Qwen/Qwen3-Reranker-4B`、相同英文指令与Top40预算。候选正文不变，具体入池文档允许随Embedding变化；0.6B组复用已有真实响应。
- 保持原500条调参、500条内部验证划分。开发集曾用于历史实验，这些结果不作为新的独立测试；不运行冻结test，不删除困难样本，gold不进入模型请求。

计划、语料及缓存身份均由脚本校验。[聚合结果](rag-embedding4b-dev-20261106.json)记录指标与文件校验和。

## 重排前：500条内部验证

| 指标 | 0.6B Embedding | 4B Embedding | 变化 |
| --- | ---: | ---: | ---: |
| Dense Top100覆盖 | 86.0% | 89.4% | +3.4个百分点 |
| 两路Top100并集覆盖 | 88.4% | 91.0% | +2.6个百分点 |
| RRF Top20覆盖 | 70.2% | 76.8% | +6.6个百分点 |
| RRF Top40覆盖 | 80.0% | 83.6% | +3.6个百分点 |
| RRF Top10命中 | 64.2% | 69.2% | +5.0个百分点 |
| RRF MRR@10 | 0.4123 | 0.4678 | +0.0556 |

Top40新增覆盖32条、丢失14条，净增加18条。按query配对的2,000次bootstrap，覆盖率差值95%区间为[+1.0,+6.4]个百分点。该区间描述复用开发样本的变化。

## 重排后：同一100条试验子集

以下两组均使用4B Reranker、相同英文指令与Top40候选预算。

| 指标 | 0.6B Embedding | 4B Embedding |
| --- | ---: | ---: |
| 入池覆盖 | 80% | 85% |
| 最终Recall@10 | **72%** | **72%** |
| 最终MRR@10 | 0.4656 | 0.4560 |
| 最终Recall@1 | 38% | 36% |
| 最终nDCG@10 | 0.5243 | 0.5174 |
| 重排输入Token | 690,935 | 695,127 |
| 重排API平均耗时 | 1.286s | 1.313s |

4B新增命中4条，同时丢失4条，最终净增0条；配对bootstrap差值95%区间为[-6,+5]个百分点。这不能证明两种模型等效，也不足以支持升级后的最终收益。

相关文档入池却未进入最终Top10的数量，从8条变为13条。说明本轮更好的候选覆盖尚未被后续排序转化为最终净增；不能仅根据这一现象把所有损失归因为重排模型能力。候选组成及服务返回分数均可能影响排名。

## 采用决定与后续方向

维持现有[质量优先配置](../../configs/rag-quality.env.example)：0.6B Embedding＋4B Reranker Top40。4B Embedding保留为独立实验索引。本轮没有修改实际`.env`，也没有重启或切换运行服务。

下一步应先审阅4条新增、4条丢失及入池后未命中的具体案例，再决定是否验证适配新Embedding的融合参数、排序策略或领域训练。以上属于后续方向，本轮没有继续调参或追加API请求。

本轮没有公平测量两种Embedding的单query在线延迟或生产P95，不能从批量建库耗时推算线上代价。表中耗时仅为Rerank API客户端调用耗时。

## 两套索引均保留

- 原 `data/chroma` 未覆盖。
- 当前语料对应的0.6B完整备份：`data/source_cache/qwen-embedding-baselines/06b-1024-chroma-20261106`，目标集合有10,015条记录、1024维；旁边的同名 `.snapshot.json` 保存逐文件SHA256。复制前后与实验结束均核对，备份不作为可写运行目录。
- 旧1,000条query向量备份：`data/source_cache/qwen-embedding-baselines/06b-1024-dev-query-vectors-20261106.json`。
- 新4B索引：`data/source_cache/qwen-embedding-4b-1024-20261106/chroma`，发布的完整集合有10,015条记录、1024维，带独立模型与语料身份。

向量索引与私有服务配置不随Git发布。运行时新增可选 `CAREGUIDE_CHROMA_PATH`，用于显式选择独立索引目录，留空沿用原目录。若复现实验配置，需要同时匹配模型、维度和索引，例如：

```dotenv
CAREGUIDE_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-4B
CAREGUIDE_EMBEDDING_DIMENSIONS=1024
CAREGUIDE_CHROMA_PATH=./data/source_cache/qwen-embedding-4b-1024-20261106/chroma
```

这仅是独立实验配置示例，不是本轮采用的新默认方案。原生 `MedicalKB` 使用缓存向量和真实响应完成3条回放，候选集合与最终排序均匹配，见[原生索引回放记录](embedding-4b-native-replay-20261106.json)；这是接线验证，不新增质量样本。

## 请求与离线复算

实际346次Embedding尝试，其中345次成功、1次服务端失败未返回usage；另有100次Rerank，总计446次，未超过授权的350＋100次上限。关闭SDK自动重试，请求前持久化计数，按已完成批次恢复。成功Embedding请求返回的输入Token合计1,157,313；失败请求用量未知，实际费用以账户账单为准。

公开缓存包含检索分数和真实完整重排响应，不包含密钥。下面只读公共语料与缓存，不调用API，也不需要本地Chroma索引：

```powershell
python -m scripts.compare_rag_embeddings replay `
  --plan data/eval/cache/rag-embedding4b-plan-dev-20261106.json `
  --baseline-scores data/eval/cache/rag-recall-scores-dev-20261105.json.gz `
  --new-scores data/eval/cache/rag-recall-scores-qwen4b-dev-20261106.json.gz `
  --responses data/eval/cache/rag-embedding4b-rerank-dev-20261106.json.gz `
  --output tmp/embedding4b-public-replay.json
```

已实际执行零API复算，重排前、最终Top10和配对变化均与发布报告一致。相关检索、缓存、预算及配置回归56项通过；冻结test、冻结清单和历史结果校验和保持不变。
