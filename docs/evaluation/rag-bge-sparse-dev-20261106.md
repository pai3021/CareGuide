# BGE-M3 真实稀疏召回对照

本轮在原 Qwen Dense + BM25 结果上，增加 `BAAI/bge-m3` 模型生成的 token 权重召回。Qwen 的模型、指令、向量索引和两路候选分数保持不变；BGE 是独立第三路。本轮采用本地 CPU 动态 INT8，冻结检索 test 未重跑。

## 模型与平台

现有硅基流动配置为 `Qwen/Qwen3-Embedding-0.6B`，1024 维，地址为 `api.siliconflow.cn`。本轮对该账户发送 1 次 BGE-M3 能力探测：公开合成短句、`model=BAAI/bge-m3`、附带 `return_sparse=true`。HTTP 200，但响应项仅有 `embedding/index/object`，返回 1024 维稠密向量，没有稀疏权重字段。

这证明本次标准 Embedding 请求没有获得 Sparse 输出，不是对该平台所有接口能力的穷尽判断。[硅基流动公开接口文档](https://docs.siliconflow.cn/docs/api/embeddings-post)同样未说明稀疏返回参数。后续没有用 BGE Dense 冒充 BGE Sparse，也没有重算 Qwen API 向量。

真实第三路使用官方开源权重：

- 模型：`BAAI/bge-m3`，版本 `5617a9f61b028005a4858fdac845db406aefb181`。
- 官方编码器 `pytorch_model.bin` SHA-256：`b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38`。
- 官方稀疏头 `sparse_linear.pt` SHA-256：`45c93804d2142b8f6d7ec6914ae23a1eee9c6a1d27d83d908a20d2afb3595ad9`。
- 编码逻辑为编码器隐状态经过已训练的稀疏线性头和 ReLU，同 token ID 保留最大权重，排除 CLS/EOS/PAD/UNK；倒排检索计算共同 token 权重的乘积之和。与 BM25 的词频统计权重不同。
- 权重从 BAAI 官方仓库下载；使用 ModelScope 镜像时仍逐文件匹配固定 Hugging Face 版本的 SHA-256 / Git blob，不依赖镜像的 `master` 名称作版本保证。

模型文件不进入 Git，完整来源校验见 [模型锁定文件](bge-m3-model-20261106.lock.json)。算法参照 [官方 FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) 的 BGE-M3 推理实现，模型说明见 [BGE 官方文档](https://bge-model.com/bge/bge_m3.html)。

## 数据与实验协议

- 语料：当前版本的 10,015 条 CareGuide KUAKE RAG 记录，指纹 `6e47b0ad01598a3f1385abb53a9ab7189ca11eb63bd5674a4688e946f4e0f8cf`。
- query：既有 1,000 条 dev，保持与 [两路实验](rag-score-fusion-dev-20261105.md) 相同的固定 500/500 调参与内部验证划分。dev 过去参与过开发，本轮不是新的独立测试集。
- BGE 文档输入采用现有 `MedicalKnowledgeBase.retrieval_text`：标题、关键词、正文；query 为原始问题，不加 Qwen 的 query 指令。
- tokenizer 最大长度 512；文档实际最大 480 tokens，query 最大 60，本轮没有截断文档或 query。
- CPU：Ryzen 5 5600GT；PyTorch 2.5.1+cpu、Transformers 4.46.3；6 推理线程，文档长度分桶 batch size 8，最终 query 单条编码。仅编码器 Linear 层采用动态 INT8，稀疏头保留 FP32。单条 query 编码与服务入口一致，避免动态量化的批次依赖。
- 三路各取 Top100；对照相同 Top20 重排候选预算。分别比较 BGE 单路、原两路 RRF、三路 RRF、三路 Min-Max 加权融合。
- 权重在调参半集的 66 组步长 0.1、权重和为 1 的配置中选择，包含权重为 0 的单路/两路消融。目标先比较 Top20 金标覆盖，再比较 MRR@10；内部验证不参与选权重。
- 本轮仅评估重排前的召回与排序。没有执行新重排、LLM 回答、真实医院工具或冻结检索 test。

## 实现与数值核对

16 条真实样本上，用相同真实编码器和稀疏头，分别执行本地适配器与未修改的 FlagEmbedding 1.2.11 参考稀疏 forward / token 聚合逻辑：FP32 与 INT8 各自的最大权重误差均为 0。这是计算实现核对，不是整个检索系统的效果证明。

同批样本编码耗时 FP32 为 6.18 秒，动态 INT8 为 1.32 秒；两种精度所得稀疏权重的余弦相似度均值为 0.9904、最小值为 0.9794。因此 INT8 与 FP32 确实有数值差异，后续完整指标只能归属于本轮 INT8 配置。小批耗时不代表线上 P95 或端到端延迟。明细见 [smoke 报告](bge-m3-smoke-20261106.json)。

## 实测结果

以下是 500 条内部验证样本，全部为重排前指标。权重列顺序为 Dense / BM25 / BGE Sparse。

| 方案 | 权重 | MRR@10 | Recall@10 | Top20 金标覆盖 |
| --- | --- | ---: | ---: | ---: |
| Qwen Dense 单路 | 1 / 0 / 0 | 0.4316 | 62.2% | 71.2% |
| BM25 单路 | 0 / 1 / 0 | 0.2604 | 39.6% | 44.8% |
| BGE Sparse 单路 | 0 / 0 / 1 | 0.3904 | 55.6% | 63.2% |
| 原两路 RRF | 4 / 1 / 0 | 0.4123 | **64.2%** | 70.2% |
| 前轮两路 Min-Max | 0.7 / 0.3 / 0 | 0.4384 | 63.2% | 70.6% |
| 本轮 RRF 网格选出的配置 | 0.9 / 0 / 0.1 | 0.4369 | 63.0% | 71.4% |
| 本轮三路 Min-Max | 0.6 / 0.1 / 0.3 | **0.4495** | 63.4% | **72.2%** |

RRF 网格选出的 BM25 权重为 0，实际只启用 Dense 和 BGE Sparse；不能把这一项描述为三路均有正贡献。Min-Max 保留了三路，调参半集上的 Top20 覆盖为 76.4%，高于原 RRF 的 75.4%。

相对原 RRF，三路 Min-Max 在内部验证 Top20 新覆盖 19 条、同时丢失 9 条，净增 10 条（+2.0 个百分点）；Top10 新覆盖 13 条、丢失 17 条，净减 4 条（-0.8 个百分点）。相对前轮两路 Min-Max，Top20 覆盖增加 1.6 个百分点，MRR@10 增加约 0.0111，Recall@10 增加 0.2 个百分点。

Top100 候选并集的覆盖率由两路 88.4% 到三路 89.8%，新增覆盖 7 条。BGE Sparse 确实提供了补充信号，但还不足以证明新方案全面优于原 RRF。增加第三路也增加了编码与索引成本，不能将本轮收益解释成无额外成本的排序改进。

完整 1,000 条 dev 上，原 RRF / 三路 Min-Max 的 Top20 覆盖为 72.8% / 74.3%，MRR@10 为 0.4272 / 0.4633，Recall@10 为 64.6% / 65.6%。这些数字包含调参样本，只作描述，不用于替换内部验证结论。

本轮保留默认融合配置，提供已验证的可选三路入口。后续若要替换默认方案，应在固定配置下比较重排后的最终 Top10 质量和实际查询成本；本轮不能宣称回答准确率或线上 P95 改善。

结果与原始候选分数见 [机器可读报告](rag-bge-sparse-dev-20261106.json) 和 [真实三路分数缓存](../../data/eval/cache/rag-recall-scores-bge-sparse-dev-20261106.json.gz)。缓存仅含公开样本/文档 ID、金标 ID 和分数，不含 API 密钥或个人信息。

## 离线复现与可选运行时入口

准备好项目依赖及上述 PyTorch / Transformers 版本后，可以下载真实模型、编码第三路并重放融合：

```powershell
python -m scripts.download_bge_sparse_model --output models/bge-m3 --mirror modelscope --workers 8

python -m scripts.extend_rag_cache_bge_sparse `
  --cache data/eval/cache/rag-recall-scores-dev-20261105.json.gz `
  --model-path models/bge-m3 `
  --checkpoint data/source_cache/bge_m3/vectors-int8.sqlite3 `
  --output tmp/bge-three-way.json `
  --quantization dynamic_int8 --batch-size 8 --threads 6

python -m scripts.evaluate_rag_score_fusion `
  --cache data/eval/cache/rag-recall-scores-bge-sparse-dev-20261106.json.gz `
  --output tmp/bge-fusion-replay.json `
  --diagnostics tmp/bge-fusion-gold-ranks.jsonl
```

SQLite 检查点逐批保存，恢复时验证模型版本、语料、输入文本及量化设置；缺失部分继续编码，配置不符则拒绝复用。query 不会作为文档加入倒排索引。

仅重放已发布的分数缓存时，可单独运行最后一条命令，不需要模型、私有配置或 API 请求。缓存中的 `processing_seconds_this_run` 描述该次执行，不包含此前恢复使用的文档检查点构建时间，不能用于推导完整建库或在线推理耗时。

运行时已提供可选配置：

```dotenv
CAREGUIDE_RETRIEVAL_MODE=hybrid
CAREGUIDE_BGE_SPARSE_MODEL_PATH=./models/bge-m3
CAREGUIDE_BGE_SPARSE_CHECKPOINT=./data/source_cache/bge_m3/vectors-int8.sqlite3
CAREGUIDE_HYBRID_FUSION=minmax
CAREGUIDE_HYBRID_WEIGHTS={"dense":0.6,"bm25":0.1,"bge_sparse":0.3}
```

上面是本轮 Min-Max 实验配置，不是新的产品默认值。`CAREGUIDE_HYBRID_WEIGHTS` 的 JSON 键为 `dense`、`bm25`、`bge_sparse`，必须与启用的支路一致。权重为 0 时跳过对应支路的查询工作；旧 `CAREGUIDE_RRF_*` 权重仍兼容。未提供独立权重时三路 RRF 默认比例为 4:1:1，不代表已验证的最优配置。

加载器只读已有完整文档检查点，校验语料指纹、文档集合、每条输入文本哈希及模型文件；查询不会下载模型或重建索引。BGE 候选遵守既有范围和有效期过滤，并统一进入原重排流程。

启用前仍需有与当前语料匹配的 Qwen Chroma 索引。容器部署需额外只读挂载模型和稀疏检查点，以上路径使用容器内路径。配置模板已补齐；本轮没有修改实际 `.env` 或重启运行服务。

## 验证

- 相关检索、范围过滤、重排边界、缓存身份校验及聊天业务回归共 60 项通过；另有一条既有 LangChain 弃用提示。
- 用真实 BGE 模型、完整稀疏检查点、真实 Qwen 缓存向量和 Chroma 副本，核对 3 条 dev query 的两种融合配置：6 项运行时 Top10 与最终单条 query 分数缓存一致。该检查不发 API 请求，不是额外质量测试集，见 [实际运行时核对](bge-m3-runtime-smoke-20261106.json)。
- 冻结 `test.jsonl`、冻结配置清单和冻结结果的 SHA-256 均保持不变。
- 本轮只有 1 次硅基流动能力探测请求，BGE 编码、三路检索、权重搜索和运行时核对均未增加 Embedding/LLM API 请求。底层缓存里的历史 `external_requests=32` 属于前轮 Qwen 缓存采集，未在本轮重复执行。
