# RAG 重排指令、模型与候选池对照

本轮找到一个有实际收益的质量优先配置：保持两路召回、原 RRF 4:1 与 Top40 候选不变，用 Qwen3-Reranker-4B 替换 0.6B。同一固定 100 条开发样本的最终 Recall@10 从 60% 到 72%，MRR@10 从 0.4179 到 0.4656（英文指令控制组）。尚未达到最终 80%；没有把候选覆盖率或调参结果作为最终测试成绩。

## 实验设计

- 沿用前轮按固定哈希选出的 100 条开发验证 query，不按命中情况重新抽样，也不删除困难或标注待审阅案例。
- Qwen Dense、BM25 原始 Top100 分数、语料、原 query 和候选正文保持不变，融合仍是 RRF 常数60、Dense:BM25=4:1。
- 已有中文指令、0.6B、Top40 的100次真实响应作为基线复用，不重复扣请求预算。
- 三组新增对照每次改变一个因素：指令语言（含义相同）、模型大小、候选池大小。请求顺序按query轮换。
- 新请求返回全部40或100项排序，用于计算最终Top10及定位gold位置；gold从不进入请求。
- 本轮追加授权300次，实际300次，无失败、无自动重试：100次0.6B、200次4B，累计输入Token为3,093,477。不调用Embedding、生成式LLM、医院工具或冻结test。
- 开发集已有历史使用，本轮是固定试验子集上的对照，不是新的独立测试或临床评测。

指令实验依据为 [Qwen官方模型卡](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) 的多语言英文指令建议；接口参数依据 [硅基流动官方文档](https://docs.siliconflow.cn/docs/api/rerank-post)。这提供了实验动机，不保证项目中必然有收益。

## 实测结果

以下均为同一100条开发query，最终Top10。

| 模型 / 指令 / 候选池 | 候选覆盖 | Recall@10 | MRR@10 | 输入Token | API平均耗时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.6B / 中文 / Top40（已有缓存） | 80% | 60% | 0.4003 | 710,935 | 0.805s |
| 0.6B / 英文 / Top40 | 80% | 60% | 0.4179 | 690,935 | 0.837s |
| **4B / 英文 / Top40** | **80%** | **72%** | **0.4656** | **690,935** | **1.286s** |
| 4B / 英文 / Top100 | 86% | 69% | 0.4510 | 1,711,607 | 2.446s |

对应的前轮原0.6B中文Top20在相同100条样本上为58%，属于候选池和模型都不同的整套配置对照；不能把58%→72%全归因于模型大小。历史冻结test的66.8%也不能直接与本子集72%比较。

### 观察、解释与决定

1. **仅改英文指令，没有Recall净增。** 新命中2条、丢失2条；MRR有所提高。保留“指令值得验证”的判断，不宣称语言本身提高了召回率。
2. **相同英文指令和Top40预算下，4B新增命中12条、丢失0条。** Recall绝对增加12个百分点，相对增加20%。按query配对的2000次bootstrap，差值95%区间为[+6,+18]个百分点。入池未进Top10的gold由20条降为8条，说明这次改进主要减少了重排阶段损失。
3. **4B Top100没有更好。** 相对4B Top40新增命中3条、丢失6条，净减少3条；差值区间[-9,+3]个百分点。Token约为2.48倍，API平均耗时约为1.90倍。这一配置不采用。
4. **质量优先配置有成本。** 4B Top40相较英文0.6B Top40，输入Token相同，但API平均耗时约增加54%。不同模型的Token单价可能不同，实际费用以账户账单为准，不能因Token相同就说价格相同。

区间仅描述这批复用开发样本的配对变化，不替代更大范围或独立数据的验证。API耗时也不代表包含检索、编排与回答生成的完整服务P95。

## 保留原排序的离线对照

在查看本轮结果前固定五个reranker排名权重：0、0.25、0.5、0.75、1，与原检索排名通过RRF融合。仅使用返回完整排序的三个新增组，不用旧Top10响应补造其余分数。将100条按固定种子再次划成50条调参、50条内部验证；先以调参Recall@10、再MRR选择，平分优先纯重排。

三组均选出权重1，即纯重排。4B Top40在这50条验证样本上Recall@10仍为72%，混合没有形成应采用的新方案。因此不把“保留原排序”写成已实现收益。结果见 [离线排名融合报告](rag-rank-retention-dev-20261106.json)。

## 外部服务一致性诊断

Top40与Top100两组中共享4000个query-document对，返回分数平均绝对差约0.00225、最大约0.03556；只有9条query的共享40项相对排序完全一致。这说明更换候选集合时，也伴随了服务返回分数的变化。

目前没有定位变化来自服务批次、数值精度还是其他因素，不把Top100下降完全归因为“模型被干扰”。各候选池分别实际请求的结果作为主要证据；不能从一次大池打分精确推算另一池的实际API结果。

## 可选运行配置

提供 [质量优先配置模板](../../configs/rag-quality.env.example)：

```dotenv
CAREGUIDE_RETRIEVAL_MODE=hybrid
CAREGUIDE_RERANK_ENABLED=true
CAREGUIDE_RERANK_MODEL=Qwen/Qwen3-Reranker-4B
CAREGUIDE_RERANK_INSTRUCTION=Rank the candidate passages by whether they can directly answer the Chinese medical query.
CAREGUIDE_RERANK_CANDIDATES=40
```

模板还固定原两路融合参数、关闭可选BGE第三路。它不包含密钥，也不会创建索引；使用前需要语料与模型身份匹配的Qwen向量索引。运行时现已读取显式重排指令，留空仍使用原中文默认值。

可以在部署评审后使用两个配置文件合并参数：

```powershell
docker compose --env-file .env --env-file configs/rag-quality.env.example config --quiet
```

本轮只校验公共模板的Compose配置，没有修改实际`.env`或重启服务。公共默认仍是离线BM25；启用原重排时的默认模型/Top20也保留，质量优先方案通过上述模板显式选择。

## 零API复算

[实际响应缓存](../../data/eval/cache/rag-reranker-model-dev-20261106.json.gz)、[固定计划](../../data/eval/cache/rag-reranker-model-plan-dev-20261106.json)、[聚合报告](rag-reranker-models-dev-20261106.json)均已保存，不含密钥或私人数据。

```powershell
python -m scripts.compare_rag_rerank_models run `
  --scores data/eval/cache/rag-recall-scores-dev-20261105.json.gz `
  --plan data/eval/cache/rag-reranker-model-plan-dev-20261106.json `
  --responses data/eval/cache/rag-reranker-model-dev-20261106.json.gz

python -m scripts.evaluate_rag_rank_retention `
  --scores data/eval/cache/rag-recall-scores-dev-20261105.json.gz `
  --plan data/eval/cache/rag-reranker-model-plan-dev-20261106.json `
  --responses data/eval/cache/rag-reranker-model-dev-20261106.json.gz `
  --output tmp/rank-retention-replay.json
```

以上命令默认不加载私有配置、不发模型请求。真实API执行必须另外显式传入远端开关和预算，本轮授权已全部使用，不能复用额度。

相关检索/协议/预算/配置回归46项和聊天回归28项通过，合计74项；另有一条既有LangChain弃用提示。冻结test、冻结清单和历史结果校验和保持不变。
