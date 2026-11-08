# SQLite Memory 向量召回接入

日期：2026-11-08。默认 Memory 继续使用 SQLite，本次补齐 sqlite-vec 长期记忆和情景摘要召回；RAG 仍使用原 BM25 / Chroma / Hybrid / Rerank 配置。没有修改实际 `.env` 或重启服务。

## 数据与检索

原文、正式长期记忆、摘要及向量保存在同一数据库。新增 `memory_vectors` 只保存可重建的向量及其来源、版本、正文哈希和 embedding 空间指纹；它不拥有正式正文。`memory_episodes` 保存压缩摘要与原文范围，检索结果作为历史背景注入，不授予当前医疗证据或医院操作权限。

采用 sqlite-vec 的普通 BLOB 表和 `vec_distance_cosine` 标量函数，在有效用户候选上做精确 KNN；没有实现 ANN。官方提供这种用法：[SQLite 标量函数 KNN](https://alexgarcia.xyz/sqlite-vec/features/knn.html#manually-with-sql-scalar-functions)。原生扩展在每个检索连接加载后立即关闭扩展加载权限，包版本固定为 `sqlite-vec==0.1.9`。[Python 加载接口](https://alexgarcia.xyz/sqlite-vec/python.html)

- 正式条目查询先校验用户、有效状态、期限、版本、正文哈希，然后按余弦距离排序；FTS5 补充，最多 5 条 / 2400 字符。
- 情景查询先回查同用户的归档摘要及内容哈希，再按相似度召回，最多 5 条。情景摘要来自现有 token 压缩；尚未发生压缩时没有情景摘要。
- 模型、维度、服务地址和查询指令均参与 embedding 空间指纹，不跨空间混用向量。
- 每次读取最多回填 16 条缺失向量，之后生成一个 query embedding，同时用于长期记忆和情景摘要。更新失败的版本仍保持未索引状态，下次读取可重试。
- 写入向量前再次核对正式来源，防止 embedding 调用期间发生的修改或删除被旧结果覆盖。删除/过期的旧向量即使尚未物理清理，也不能被 canonical join 召回。
- SQLite 向量状态与旧 Chroma 的 `indexed` 标记分离。SQLite 召回不消耗 Chroma 的待处理摘要，切回兼容模式后仍可继续投影历史摘要。

## 配置

```dotenv
CAREGUIDE_MEMORY_MODE=sqlite
CAREGUIDE_MEMORY_CACHE=none
CAREGUIDE_MEMORY_VECTOR_MODE=auto
CAREGUIDE_MEMORY_VECTOR_TIMEOUT=1.5
CAREGUIDE_MEMORY_VECTOR_MIN_SCORE=0.3
```

`auto` 在现有 `CAREGUIDE_EMBEDDING_API_KEY` 有效配置时复用其服务、模型和维度；`remote` 要求配置密钥；`disabled` 保留 FTS5。Memory 使用独立查询指令，不改变 RAG 或业务意图检索的指令。无 Embedding 配置不会伪造向量，也不自动下载模型。

文档批次和 query 调用共享默认 1.5 秒、最多 2 秒的向量路径墙钟预算，provider 重试为零；逐用户 UTC 每日向量请求最多 100 次，由 SQLite BudgetLedger 保存。故障、超时、无匹配或预算耗尽不会删除原文，正式长期条目仍可经 FTS5 / 近期条目读取。健康信息提供 `vector_configured`、`vector_status`。

## 工程验证

`tests/test_sqlite_vector_memory.py` 包含 16 项原生扩展测试，使用明确标记的受控向量与合成消息：

| 验证 | 结果 |
| --- | --- |
| 原生 sqlite-vec 加载与余弦查询 | 本地实际加载 v0.1.9 |
| 关键词不命中时走向量路径 | 预先指定的受控向量目标进入首位；主聊天入口收到相应记忆 |
| 进程重启 | 独立 Python 进程可读取持久化向量，无需重新生成文档向量 |
| 情景摘要 | 压缩后可跨会话召回；删除原归档会话后不能再召回 |
| 用户隔离与 TopK | 其他用户更近的向量不能占用当前用户的结果 |
| 更正、删除、过期 | 旧版本不返回；迟到的 embedding 不能复活删除内容 |
| 模型、维度与指令变更 | 旧空间向量不参与新空间检索，待重新向量化 |
| 无效向量 | 拒绝零向量、NaN、Inf、维度错误、float32 溢出及下溢 |
| 超时及故障 | 返回本地条目；超时线程结束后没有迟到向量写入，后续读取可恢复 |
| 兼容模式 | SQLite 向量召回不消耗 Chroma 独立索引状态 |

首次受影响范围验证：Memory、归档及意图语义兼容共 52 项通过，保留一条既有 LangChain 弃用警告。

最终完整后端回归：**778 passed, 1 warning**，耗时 104.10 秒。相同专用 Python 3.12.14 环境实际加载 sqlite-vec v0.1.9；基础与 full Compose 均使用公共 `.env.example` 完成配置校验。原生扩展测试包含独立进程恢复；新增向量测试共 16 项。测试明细保存在本地 `tmp/sqlite-vec-final-20261108.xml`。

这些测试验证真实 sqlite-vec 存取和检索控制链路，**不代表真实 Embedding 模型的改写召回质量或提取准确率**。本轮未请求外部 Embedding 服务。原纯 FTS5 的 3/8 改写探针属于前一阶段结果，不能据此虚构接入后的准确率提升。

## 复现

```powershell
& 'D:\miniconda3\envs\nexusops-dev\python.exe' -X utf8 -m pytest -q -p no:cacheprovider tests/test_sqlite_vector_memory.py tests/test_sqlite_memory.py tests/test_session_archive.py tests/test_intent_semantic.py --basetemp tmp/sqlite-vec-check
```
