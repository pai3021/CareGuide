# CareGuide

**医疗信息检索与模拟预约 Agent**

CareGuide 是一个用于医疗资料查询和预约流程演示的 Agent 项目。用户可以在网页中提问并查看回答所依据的资料，也可以查询合成医院里的科室、医生和号源，在确认后模拟创建、改期或取消预约。

系统用 CareGuide-13 判断请求类型，再进入分诊、用药、预约或急症提示流程。知识检索默认使用 BM25，也可启用 Dense 检索、加权 RRF 和 Qwen Reranker。项目不连接真实医院，也不使用真实患者数据；检索资料未经临床审核，不用于诊断、处方或个体化剂量建议。

## 系统架构

![CareGuide 系统架构](figures/careguide-system-architecture.png)

可编辑的请求流程图：[查看 Mermaid 源文件](figures/careguide-runtime.mmd)。

复合请求由模型 Planner 读取完整原文并提出任务 DAG，程序校验权限、参数和输入绑定后，由 LangGraph 外层流程调度。支持局部变更、等待补参或选择、跨进程恢复和累计预算；预约继续使用原有确认入口。简单查询保留直接路由。配置与验收证据见 [编排架构与运行说明](docs/orchestration/implementation.md) 和 [验收矩阵](docs/orchestration/acceptance.md)。

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

多任务动态规划需要配置兼容 Chat Completions 的模型。运行与恢复说明见上述编排文档；离线验证可以运行下列脚本，不需要密钥：

```bash
python scripts/reproduce_orchestration.py --scenario dependency
python scripts/reproduce_orchestration.py --scenario replan
python scripts/reproduce_orchestration.py --scenario wait
python scripts/reproduce_orchestration.py --scenario partial
python scripts/reproduce_orchestration.py --scenario memory
```

脚本使用可控 Planner 响应与真实调度／业务代码，不能据此推断真实模型的规划准确率或线上费用。

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
