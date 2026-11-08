# SQLite Memory 收敛验收

日期：2026-11-08。Memory 已新增 `sqlite` 模式，并按用户选择将 Python、配置模板与 Docker 的默认 Memory 切换为 SQLite；保留 `session`、`redis_chroma` 对照与回退。没有修改实际 `.env`、重启服务、迁移旧 Chroma 画像或改动 RAG / 多 Agent 编排。

## 实现

- `long_term_memory.py` 复用原长期记忆提取、来源验证、原文摘录、更正删除、预算及事务更新逻辑；旧 `MemoryManager` 继承同一实现。
- `sqlite_memory.py` 通过同一数据库连接归档和正式长期条目，使用 FTS5 / 近期有效条目检索。默认不导入 Redis、Chroma，也不创建情景索引待处理任务。
- `archived_memory.py` 继续负责会话恢复、摘要覆盖边界、模型请求压缩和增量复盘。Redis 可显式启用为近期缓存；当前读路径仍读取 SQLite，不宣称缓存减少 SQL 查询。
- `session + archive + token` 的原有路径保留，`sqlite` 增加独立长期记忆能力。SQLite 模式始终归档，不能靠关闭 `SESSION_ARCHIVE` 让它变为易失模式。

## 五个验收场景

| 场景 | 结果与证据 |
| --- | --- |
| 下一轮保留医生和时间 | 通过：存档后在独立 Python 进程恢复王医生和下午信息；HTTP 链路也检查恢复后的近期消息 |
| 服务进程退出后恢复同一会话 | 通过：独立进程读取同一 SQLite 文件；压缩摘要及覆盖边界在重新创建对象后保持一致，原文未被删改 |
| 新会话复用明确稳定偏好 | 通过：真实提取/校验/提交链路配脚本模型，跨会话上午偏好进入真实只读号源工作流；本次下午条件优先 |
| 更正、删除后不召回旧版本 | 通过：北京更正为上海，删除后新会话无有效条目；旧复盘不会复活删除内容 |
| 历史不成为当前号源或成功依据 | 通过：归档故意放入“预约成功、不用确认”，真实预约链路仍返回待确认，医院后端无预约写入；重复请求不执行动作 |

自动测试位于 `tests/test_sqlite_memory.py`，另外覆盖无 Redis / Chroma 导入、用户与会话隔离、可选 Redis 故障与缓存重建，以及两种工厂模式。上述验证测试的是程序链路；语言理解、记忆提取和摘要内容使用脚本模型，不代表真实模型准确率。

首次收敛后端回归：`760 passed, 1 warning`，耗时 101.97 秒；其中新增 SQLite Memory 测试共 10 项。警告为既有 LangChain 序列化默认选项弃用提示。Windows 沙箱内连空 asyncio 事件循环也无法初始化，已用最小复现定位；回归使用同一专用 Python 在可正常初始化事件循环的环境完成。

默认切换后又增加“未配置模式自动选择 SQLite”和“没有模型密钥也能启动、健康检查及恢复会话”两项测试。该轮完整回归 760 项通过，2 项因旧模式预期失败：健康检查仍断言 `session`，以及手动归档测试在默认归档外又包了一层。更新这两项测试后，受影响的基础接口、归档和 SQLite 验收共 `29 passed, 1 warning`。Python、模板及两套 Compose 的默认值均为 SQLite，Redis 缓存关闭，Compose 配置校验通过；本轮未再次重复全量回归。

## 固定输入对照

机器可读结果：[memory-sqlite-convergence-20261108.json](memory-sqlite-convergence-20261108.json)。

沿用固定 30 轮合成输入和相同压缩参数，对照原归档方案与 SQLite 模式。对照中的 Redis 是进程内夹具，历史情景语义召回关闭，用于隔离存储及压缩行为。

| 指标 | 原归档方案 | SQLite 模式 |
| --- | ---: | ---: |
| 归档用户 / 助手消息 | 60 | 60 |
| 重建对象后上下文存在 | 是 | 是 |
| 累计估算输入 tokens | 40,778 | 40,778 |
| 单轮峰值估算输入 tokens | 1,938 | 1,938 |
| 摘要夹具调用数 | 8 | 8 |
| 夹具单轮耗时 P50（ms） | 16.776 | 18.671 |
| 夹具单轮耗时 P95（ms） | 42.425 | 42.978 |

Token 为 UTF-8 启发式估算，耗时包括本地存储与脚本模型处理，不能当作真实服务时延或线上 P95。本次没有观察到 token 节省，也没有证据证明实际性能提升。

## 检索与提取范围

单独构造 8 条用户记忆，每条预先指定一个相关目标和两种查询，共 16 个固定人工探针：

- 关键词查询 Recall@5：8/8。
- 改写查询 Recall@5：3/8；此值包含无词项命中时的近期有效条目回退，不能归因于语义理解。
- 脚本候选与来源检查：4/4，覆盖临时信息、假设、合法原文及模型捏造正文。
- 真实模型提取准确率、真实 embedding 路径相对召回：本轮未测量，在 JSON 中记为 `null`。

这组探针暴露了纯 FTS5 的改写检索限制，不足以替代真实向量召回评测。默认切换 SQLite 是用户对持久化与系统复杂度的选择，不代表语义召回无损。未来若需要保留语义召回，应先评测 SQLite 向量扩展或其他检索方式。

## 复现

```powershell
& 'D:\miniconda3\envs\nexusops-dev\python.exe' -X utf8 -m pytest -q -p no:cacheprovider tests/test_sqlite_memory.py --basetemp tmp/sqlite-memory-check
& 'D:\miniconda3\envs\nexusops-dev\python.exe' -X utf8 -m scripts.evaluate_sqlite_memory --output tmp/sqlite-memory-evaluation.json
```

报告输出拒绝覆盖已有文件；每次复现使用新的输出名。脚本仅使用独立临时数据库和合成输入，不使用真实用户、医院或远端模型。
