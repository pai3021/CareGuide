# CareGuide

**来源支持的医疗信息检索与合成预约 Agent**

CareGuide 将意图识别、来源检索和受限工具调用串在同一条请求链路中。CareGuide-13 在 KUAKE-QIC 11 类基础上增加预约动作和急症意图，并支持 Transformer 与轻量分类模式；编排器据此选择分诊、用药、预约或急症提示角色，按角色授权工具。

知识请求使用带版本记录的资料来源，默认通过 BM25 检索，也可配置 Embedding、加权 RRF 和 Qwen Reranker。预约写操作须经用户确认，且只修改合成医院数据。系统支持进程内会话记忆及可选 Redis/Chroma，记录 Trace，并分别评测意图识别、检索和端到端工作流。

> CareGuide 不提供诊断、处方或个体化剂量建议，不连接真实医院，也不使用真实患者数据。

## 系统架构

![CareGuide 系统架构](figures/careguide-system-architecture.png)

可编辑的请求流程图：[查看 Mermaid 源文件](figures/careguide-runtime.mmd)。

## 仓库结构

| 路径 | 内容 |
| --- | --- |
| `careguide/` | API、路由编排、Agent、工具、检索、记忆与评测实现 |
| `frontend/` | Vue 3 界面 |
| `data/` | 版本化资料、合成医院数据集和冻结评测摘要 |
| `scripts/` | 服务启动、语料构建、索引、训练与评测入口 |
| `tests/` | 单元测试、契约测试与工作流回归 |

## 快速开始

需要 Python 3.12。默认配置使用 BM25、进程内会话记忆和内存合成医院；基础模式不需要模型 API 密钥。

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements-semantic.txt
cp .env.example .env
python -m scripts.serve --config-file .env --host 127.0.0.1 --port 8010
```

PowerShell 中可用 `Copy-Item .env.example .env` 代替 `cp`。API 地址为 <http://127.0.0.1:8010>，健康检查为 <http://127.0.0.1:8010/health>。

启动前端：

```bash
cd frontend
npm ci
npm run dev
```

Docker 完整模式：

```bash
docker compose --env-file .env \
  -f docker-compose.yml -f docker-compose.full.yml \
  -f docker-compose.integrations.yml \
  --profile memory --profile integrations up -d --build
```

Dense/Hybrid 检索需要配置 Embedding 服务并显式建立版本化 Chroma 索引；Reranker 和 Transformer 模式也需要各自的模型服务或本地 checkpoint。模型权重和 API 密钥不随仓库发布，配置模板见 [`.env.example`](.env.example)。

## 评测结果

| 任务 | 评测集 | 结果 |
| --- | --- | --- |
| 意图识别 | 1,097 条冻结测试集；RoBERTa-wwm-ext | Accuracy 83.77%；Macro-F1 83.66% ([评测说明](data/eval/careguide_intent_v4/README.md)) |
| RAG 检索 | 1,000 条冻结测试集 | MRR@10 0.4857；Recall@10 0.6680；nDCG@10 0.5292 ([评测说明](data/eval/careguide_kuake_rag_v1/README.md)) |
| 端到端开发集 | 60 条合成工作流 | 确定性任务通过率 100%；Judge overall 0.9273 |
| 端到端固定契约回归 | 60 条合成工作流 | 确定性任务通过率 88.33%；证据正确率 80%；Judge overall 0.9015 ([冻结摘要](data/eval/careguide_e2e_v1/frozen_provider_heldout_result_v4.json)) |

各项结果来自不同评测任务，不能合并为“医疗准确率”。固定契约回归集曾在离线开发中运行过，不是全新未见测试集；评测未验证临床有效性。数据划分、模型配置和哈希见对应目录中的清单文件。

## 数据与范围

项目中的医院、医生、号源和预约均为合成数据。公开健康资料和 KUAKE 派生语料未经 CareGuide 临床审核；检索分数不代表临床正确性。第三方数据、模型和代码的来源及许可见 [`NOTICE.md`](NOTICE.md)。
