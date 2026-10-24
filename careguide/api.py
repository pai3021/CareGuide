"""Read-only sourced information API; clinical answering is not enabled."""
from datetime import datetime, timezone
import asyncio
import uuid
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic import ConfigDict, ValidationError

from careguide.domain import AgentType, IntentCategory, capabilities, plan
from careguide.runtime import get_registry, get_chat_runtime
from careguide.tools import MedicalToolRegistry
from careguide.orchestration.store import VersionConflict, WorkflowExpired
from careguide.orchestration.budget import BudgetExhausted

@asynccontextmanager
async def lifespan(app):
    engine, memory, monitor = get_chat_runtime()
    await monitor.start()
    yield
    await monitor.stop()
    await memory.close()
    if getattr(engine, 'knowledge_backend', None) is not None:
        await engine.knowledge_backend.close()
    if getattr(engine, 'appointment_service', None) is not None:
        await engine.appointment_service.backend.close()
    if engine.client is not None:
        await engine.client.close()
    get_chat_runtime.cache_clear()


app = FastAPI(title='CareGuide', version='0.3.0', lifespan=lifespan)
app.add_middleware(CORSMiddleware,
    allow_origins=os.environ.get('CAREGUIDE_CORS_ORIGINS', 'http://127.0.0.1:5175,http://localhost:5175').split(','),
    allow_methods=['GET', 'POST'], allow_headers=['Content-Type'])
_conversation_locks = {}
_MEMORY_READ_TIMEOUT_S = 2
_MEMORY_WRITE_TIMEOUT_S = 8


@asynccontextmanager
async def _versioned_change():
    # Structured changes use the workflow's atomic version checks, allowing
    # them to invalidate a read while an earlier HTTP request is still active.
    yield


class WorkflowEvent(BaseModel):
    model_config = ConfigDict(extra='forbid')
    kind: str = Field(pattern='^(answer|change|run|cancel)$')
    workflow_id: str = Field(min_length=1, max_length=64)
    plan_version: int = Field(ge=1)
    task_id: str | None = Field(default=None, min_length=1, max_length=64)
    task_revision: int | None = Field(default=None, ge=1)
    values: dict = Field(default_factory=dict, max_length=12)
    event_id: str | None = Field(default=None, min_length=1, max_length=100)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    message: str = Field(min_length=1, max_length=4000)
    user_id: str = Field(default='demo-user', min_length=1, max_length=100)
    conv_id: str | None = Field(default=None, min_length=1, max_length=100)
    workflow_event: WorkflowEvent | None = None


@app.post('/chat')
async def chat(request: ChatRequest, runtime=Depends(get_chat_runtime)):
    engine, memory, monitor = runtime
    conv_id = request.conv_id or uuid.uuid4().hex
    key = (request.user_id, conv_id)
    if key not in _conversation_locks:
        if len(_conversation_locks) >= 1000:
            available = next((key for key, lock in _conversation_locks.items() if not lock.locked()), None)
            if available is None:
                raise HTTPException(503, '会话服务繁忙')
            del _conversation_locks[available]
        _conversation_locks[key] = asyncio.Lock()
    guard = (_versioned_change() if request.workflow_event and request.workflow_event.kind in {'change', 'cancel'}
             else _conversation_locks[key])
    async with guard:
        try:
            memory_status = 'available'
            try:
                async with asyncio.timeout(_MEMORY_READ_TIMEOUT_S):
                    history = await memory.history(request.user_id, conv_id, query=request.message)
            except Exception:
                history, memory_status = [], 'unavailable'
            if request.workflow_event:
                event = request.workflow_event
                if not request.conv_id or (event.kind in {'answer', 'cancel'} and (
                        not event.task_id or event.task_revision is None)) or (event.kind == 'answer' and not event.values):
                    raise HTTPException(422, '恢复需要会话、任务版本和本次回答')
                if not hasattr(engine, 'store'):
                    raise HTTPException(409, '当前运行模式不支持任务恢复')
                result = await engine.process(request.message, request.user_id, conv_id, history,
                    workflow_event=event.model_dump())
            else:
                result = await engine.process(request.message, request.user_id, conv_id, history)
            try:
                async with asyncio.timeout(_MEMORY_WRITE_TIMEOUT_S):
                    await memory.add(request.user_id, conv_id, 'user', request.message)
                    memory_update = await memory.add(request.user_id, conv_id, 'assistant', result['response'])
                if memory_update is not None:
                    result['memory_update'] = memory_update
            except Exception:
                memory_status = 'unavailable'
            result['memory_status'] = memory_status
            return result
        except HTTPException:
            raise
        except LookupError as exc:
            raise HTTPException(404, '未找到当前用户和会话的任务') from exc
        except WorkflowExpired as exc:
            raise HTTPException(410, '任务或候选项已过期，请重新查询') from exc
        except VersionConflict as exc:
            raise HTTPException(409, '任务已更新，请刷新后使用当前问题或确认入口') from exc
        except (ValueError, PermissionError) as exc:
            raise HTTPException(422, '回答不符合当前任务或候选项约束') from exc
        except Exception as exc:
            raise HTTPException(503, '对话服务暂不可用') from exc


@app.get('/skills')
def skills(runtime=Depends(get_chat_runtime)):
    return runtime[0].skills.summary()


@app.get('/workflows/latest')
def latest_workflow(user_id: str, conv_id: str, runtime=Depends(get_chat_runtime)):
    engine = runtime[0]
    if not hasattr(engine, 'store'):
        return {'workflow': None}
    state = engine.store.latest(user_id, conv_id)
    if state is None:
        return {'workflow': None}
    state = engine.store.expire_waits(state.id, user_id, conv_id)
    state = engine.store.sync_actions(state.id, user_id, conv_id)
    from careguide.orchestration.replies import compose_workflow
    return compose_workflow(state, engine.ledger(state))


@app.get('/workflows/{identity}')
def get_workflow(identity: str, user_id: str, conv_id: str, runtime=Depends(get_chat_runtime)):
    engine = runtime[0]
    if not hasattr(engine, 'store'):
        raise HTTPException(404, '任务编排未启用')
    try:
        state = engine.store.expire_waits(identity, user_id, conv_id)
        state = engine.store.sync_actions(identity, user_id, conv_id)
    except LookupError as exc:
        raise HTTPException(404, '未找到当前用户和会话的任务') from exc
    from careguide.orchestration.replies import compose_workflow
    return compose_workflow(state, engine.ledger(state))


@app.post('/skills/reload')
def reload_skills(runtime=Depends(get_chat_runtime)):
    runtime[0].skills.reload()
    return runtime[0].skills.summary()


@app.get('/monitor')
def monitor(runtime=Depends(get_chat_runtime)):
    return runtime[2].summary()


@app.get('/metrics')
def metrics(runtime=Depends(get_chat_runtime)):
    from prometheus_client import CollectorRegistry, Gauge, generate_latest, CONTENT_TYPE_LATEST
    registry = CollectorRegistry()
    successes = Gauge('careguide_agent_success_rate', 'Agent execution success rate', ['agent'], registry=registry)
    latency = Gauge('careguide_agent_avg_ms', 'Agent average execution latency', ['agent'], registry=registry)
    tool_successes = Gauge('careguide_tool_success_rate', 'Tool execution success rate', ['tool'], registry=registry)
    tool_latency = Gauge('careguide_tool_avg_ms', 'Tool average execution latency', ['tool'], registry=registry)
    for name, stats in runtime[0].get_stats().items():
        successes.labels(name).set(stats['success_rate'])
        latency.labels(name).set(stats['avg_ms'])
    for name, stats in runtime[0].tool_manager.get_stats().items():
        tool_successes.labels(name).set(stats.get('success_rate', 0))
        tool_latency.labels(name).set(stats.get('avg_latency_ms', 0))
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)


@app.get('/trace/tool/{request_id}')
def trace(request_id: str, runtime=Depends(get_chat_runtime)):
    result = runtime[0].traces.get(request_id)
    return {'request_id': request_id, 'found': result is not None, 'trace': result}


@app.get('/trace/tools')
def recent_traces(limit: int = 20, runtime=Depends(get_chat_runtime)):
    return {'traces': runtime[0].traces.list_recent(limit)}


@app.get('/knowledge/stats')
def knowledge_stats(registry=Depends(get_registry)):
    return {'total_chunks': len(registry.corpus.records), 'read_only': True,
            'coverage': registry.corpus.coverage(datetime.now(timezone.utc).date())}


@app.post('/search')
async def search(query: str, top_k: int = 5, rewrite: bool = False, rerank: bool = False,
                 runtime=Depends(get_chat_runtime)):
    if not 1 <= top_k <= 10:
        raise HTTPException(422, 'top_k must be between 1 and 10')
    engine = runtime[0]
    if engine.knowledge_backend.name != 'embedded':
        if rewrite or rerank:
            raise HTTPException(409, '外部知识后端不支持本地查询改写或重排')
        result = await engine.knowledge_backend.search(query, top_k=top_k)
        return {'results': result['evidence'], 'success': result['status'] == 'found',
                'status': result['status'], 'retrieval_backend': result['retrieval_backend'],
                'reranked': False}
    if (rewrite or rerank) and engine.client is None:
        raise HTTPException(503, '查询改写和重排需要独立配置模型服务')
    result = await engine.tool_manager.search_with_rewrite('knowledge_search', query, top_k,
                                                         use_rewrite=rewrite, use_rerank=rerank)
    return {'results': result.data or [], 'success': result.success, 'reranked': result.reranked}


@app.get('/handoffs')
def list_handoffs(user_id: str, runtime=Depends(get_chat_runtime)):
    return {'records': runtime[0].business_store.list_handoffs(user_id), 'hospital_dispatch_performed': False}


@app.get('/handoffs/{identity}')
def get_handoff(identity: str, user_id: str, runtime=Depends(get_chat_runtime)):
    item = runtime[0].business_store.get_handoff(identity, user_id)
    if item is None:
        raise HTTPException(404, '未找到咨询记录')
    return item


class HandoffUpdate(BaseModel):
    model_config = ConfigDict(extra='forbid')
    user_id: str = Field(min_length=1, max_length=100)
    status: str = Field(pattern='^(in_progress|resolved|cancelled)$')
    note: str = Field(default='', max_length=1000)


@app.post('/handoffs/{identity}/status')
def update_handoff(identity: str, request: HandoffUpdate, runtime=Depends(get_chat_runtime)):
    try:
        return runtime[0].business_store.transition_handoff(identity, request.user_id, request.status, request.note)
    except LookupError as error:
        raise HTTPException(404, '未找到咨询记录') from error
    except ValueError as error:
        raise HTTPException(409, '不允许此状态流转') from error


def knowledge_workflow(runtime):
    from careguide.knowledge_workflow import KnowledgeWorkflow
    return KnowledgeWorkflow(runtime[0].business_store)


@app.get('/knowledge/submissions')
def knowledge_submissions(runtime=Depends(get_chat_runtime)):
    return {'submissions': knowledge_workflow(runtime).list()}


@app.post('/knowledge/add')
@app.post('/knowledge/upload')
async def add_knowledge(request: Request, submitter: str = 'local-developer', runtime=Depends(get_chat_runtime)):
    from careguide.knowledge_workflow import KnowledgeBundle
    body = await request.body()
    if len(body) > 500_000:
        raise HTTPException(413, '资料包过大')
    try:
        bundle = KnowledgeBundle.model_validate_json(body)
    except ValidationError as error:
        raise HTTPException(422, '需要带来源、核验记录和适用范围的JSON资料包') from error
    return knowledge_workflow(runtime).submit(bundle, submitter)


class KnowledgeReview(BaseModel):
    model_config = ConfigDict(extra='forbid')
    reviewer: str = Field(min_length=1, max_length=100)
    approve: bool
    note: str = Field(min_length=1, max_length=2000)


@app.post('/knowledge/{identity}/review')
def review_knowledge(identity: str, request: KnowledgeReview, runtime=Depends(get_chat_runtime)):
    try:
        return knowledge_workflow(runtime).review(identity, request.reviewer, request.approve, request.note)
    except LookupError as error:
        raise HTTPException(404, '未找到资料包') from error
    except ValueError as error:
        raise HTTPException(409, '资料包不可重复审核') from error


@app.post('/knowledge/{identity}/publish')
def publish_knowledge(identity: str, runtime=Depends(get_chat_runtime)):
    try:
        prepare_index = None
        if runtime[0].registry.knowledge.mode != 'bm25':
            from careguide.runtime import chroma_client, semantic_embedding
            from careguide.retrieval.knowledge_base import index_collection
            prepare_index = lambda corpus: index_collection(chroma_client(), corpus, semantic_embedding(), versioned=True)
        item = knowledge_workflow(runtime).publish(identity, runtime[0].registry.corpus, prepare_index)
        get_registry.cache_clear()
        updated = get_registry()
        runtime[0].registry.corpus = updated.corpus
        runtime[0].registry.knowledge = updated.knowledge
        runtime[0].recognizer.corpus = updated.corpus
        runtime[0].recognizer._cache.clear()
        return item
    except LookupError as error:
        raise HTTPException(404, '未找到资料包') from error
    except ValueError as error:
        raise HTTPException(409, '资料包需要审核且不得覆盖现有资料') from error


class EvalRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    limit: int = Field(default=12, ge=1, le=240)
    dialogs: list[list[str]] = Field(default_factory=list, max_length=10)
    judge: bool = False


def evaluation_directory():
    from pathlib import Path
    return Path(os.environ.get('CAREGUIDE_EVAL_DIR', str(Path(__file__).resolve().parents[1]/'data'/'runtime'/'evaluations')))


@app.get('/eval/reports')
def evaluation_reports():
    import json
    return {'reports': [json.loads(path.read_text(encoding='utf-8')) for path in evaluation_directory().glob('????????????????????????????????.json')]}


@app.post('/eval/baseline/{identity}')
def promote_evaluation_baseline(identity: str):
    import json
    import re
    if not re.fullmatch('[a-f0-9]{32}', identity):
        raise HTTPException(422, '非法报告编号')
    directory = evaluation_directory()
    report = directory/(identity+'.json')
    if not report.is_file():
        raise HTTPException(404, '未找到开发评测报告')
    payload = json.loads(report.read_text(encoding='utf-8'))
    temporary = directory/'baseline.tmp'
    temporary.write_text(json.dumps(payload['report'], ensure_ascii=False), encoding='utf-8')
    temporary.replace(directory/'baseline.json')
    return {'baseline_report_id': identity, 'split': 'dev', 'historical_evidence_changed': False}


@app.post('/eval/run')
async def run_evaluation(request: EvalRequest, runtime=Depends(get_chat_runtime)):
    from dataclasses import asdict
    from pathlib import Path
    import json
    from careguide.intent_dataset import load_development_cases
    from careguide.evaluator import EndToEndEvaluator, IntentTestCase
    engine = runtime[0]
    if engine.client is None:
        raise HTTPException(503, '评测需要真实模型配置')
    from careguide.llm_utils import create_llm_client
    judge_key = os.environ.get('CAREGUIDE_JUDGE_API_KEY', '') if request.judge else ''
    judge_client = create_llm_client(judge_key, os.environ.get('CAREGUIDE_JUDGE_BASE_URL') or None) if judge_key else None
    evaluator = EndToEndEvaluator(engine, engine.recognizer, api_key='', client=engine.client,
        model=engine.model, judge_enabled=request.judge,
        judge_client=judge_client, judge_model=os.environ.get('CAREGUIDE_JUDGE_MODEL') or engine.model,
        baseline_path=str(evaluation_directory()/'baseline.json'))
    cases = load_development_cases()[:request.limit]
    try:
        report = await evaluator.run(
            intent_cases=[IntentTestCase(case.message, case.primary_intent.value if case.primary_intent else
                                        case.disposition.value, {'history': [turn.model_dump() for turn in case.history]})
                          for case in cases],
            dialog_cases=[{'turns': turns} for turns in request.dialogs])
    finally:
        if judge_client is not None:
            await judge_client.close()
    payload = {'id': uuid.uuid4().hex, 'split': 'dev', 'heldout_used': False,
               'clinical_validity_evaluated': False, 'baseline_promoted': False, 'report': asdict(report)}
    output = evaluation_directory()
    output.mkdir(parents=True, exist_ok=True)
    with (output/(payload['id']+'.json')).open('x', encoding='utf-8') as file:
        json.dump(payload, file, ensure_ascii=False, default=str)
    return payload


class PlanRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    intents: list[IntentCategory] = Field(min_length=1)


class ToolRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    agent: AgentType
    tool: str = Field(min_length=1, max_length=100)
    arguments: dict


class ActionIdentityRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    user_id: str = Field(min_length=1, max_length=100)
    conv_id: str = Field(min_length=1, max_length=100)


def _intent_classifier_status(engine) -> dict:
    recognizer = engine.recognizer
    classifier = getattr(recognizer, 'semantic_classifier', None)
    if classifier is not None:
        backend = 'transformer'
        model = getattr(classifier, 'version', None)
    elif recognizer.__class__.__name__ == 'QICIntentRecognizer':
        backend = 'fusion'
        model = 'tfidf-qwen-centroid-mimo-gate'
    else:
        backend = 'legacy'
        model = recognizer.__class__.__name__
    return {'status': 'available', 'backend': backend, 'model': model}


@app.get('/health')
async def health(runtime=Depends(get_chat_runtime)):
    engine, memory, _ = runtime
    intent = _intent_classifier_status(engine)
    knowledge, hospital, memory_status = await asyncio.gather(
        engine.knowledge_backend.health(), engine.appointment_service.backend.health(),
        memory.health())
    return {'status': 'ok', 'stage': 'medical_chat_runtime',
            'integrations': {'intent_classifier': intent,
                             'knowledge': knowledge, 'hospital': hospital,
                             'memory': memory_status}}


@app.get('/capabilities')
def get_capabilities(registry: MedicalToolRegistry = Depends(get_registry),
                     runtime=Depends(get_chat_runtime)):
    engine = runtime[0]
    intent = _intent_classifier_status(engine)
    return {**capabilities(), 'retrieval_mode': registry.knowledge.mode,
            'intent_classifier_backend': intent['backend'],
            'intent_classifier_model': intent['model'],
            'knowledge_backend': engine.knowledge_backend.name,
            'hospital_backend': engine.appointment_service.backend.name,
            'hospital_integration_enabled': engine.appointment_service.backend.name != 'disabled',
            'memory': runtime[1].info(),
            'simulation': True,
            'coverage': registry.corpus.coverage(datetime.now(timezone.utc).date()),
            'registered_tools': {agent.value: [d['function']['name'] for d in registry.definitions(agent)]
                                 for agent in AgentType}}


@app.get('/actions/{identity}')
def get_action(identity: str, user_id: str, runtime=Depends(get_chat_runtime)):
    item = runtime[0].appointment_service.actions.get(identity, user_id)
    if not item:
        raise HTTPException(404, 'Action not found')
    return item


@app.get('/trace/action/{identity}')
def action_trace(identity: str, user_id: str, runtime=Depends(get_chat_runtime)):
    """Return the persistent audit trail for a confirmed synthetic write."""
    item = runtime[0].appointment_service.actions.get(identity, user_id)
    if not item:
        raise HTTPException(404, 'Action not found')
    return {'action_id': identity, 'simulation': True, 'events': item['events'],
            'arguments_hash': item['arguments_hash'], 'status': item['status'],
            'result': item.get('result')}


@app.post('/actions/{identity}/confirm')
async def confirm_action(identity: str, request: ActionIdentityRequest,
                         runtime=Depends(get_chat_runtime)):
    try:
        handler = getattr(runtime[0], 'confirm_action', runtime[0].appointment_service.confirm)
        return await handler(identity, request.user_id, request.conv_id)
    except LookupError as exc:
        raise HTTPException(404, 'Action not found') from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    except BudgetExhausted as exc:
        raise HTTPException(409, '累计预算已耗尽，已保存现有结果和操作状态') from exc
    except PermissionError as exc:
        raise HTTPException(409, '操作权限或累计预算记录无法核验') from exc


@app.post('/actions/{identity}/reject')
def reject_action(identity: str, request: ActionIdentityRequest,
                  runtime=Depends(get_chat_runtime)):
    try:
        handler = getattr(runtime[0], 'reject_action', runtime[0].appointment_service.reject)
        return handler(identity, request.user_id, request.conv_id)
    except LookupError as exc:
        raise HTTPException(404, 'Action not found') from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get('/appointments')
async def get_appointments(user_id: str, runtime=Depends(get_chat_runtime)):
    items = await runtime[0].appointment_service.backend.list_appointments(user_id)
    return {'appointments': items, 'simulation': True,
            'backend': runtime[0].appointment_service.backend.name}


@app.post('/plan')
def get_plan(request: PlanRequest):
    return plan(request.intents)


@app.get('/tools')
def get_tools(agent: AgentType, registry: MedicalToolRegistry = Depends(get_registry)):
    return {'agent': agent.value, 'read_only': True, 'tools': registry.definitions(agent)}


@app.post('/tools/query')
def query_tool(request: ToolRequest, registry: MedicalToolRegistry = Depends(get_registry)):
    try:
        return registry.query(request.agent, request.tool, request.arguments)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValidationError as exc:
        # Do not echo arbitrary argument values back into logs or HTTP errors.
        raise HTTPException(status_code=422, detail=[{'loc': list(error['loc']),
                            'type': error['type'], 'msg': error['msg']}
                            for error in exc.errors()]) from exc
