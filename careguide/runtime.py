"""Explicit medical retrieval configuration. Does not load any .env file."""
import os
import json
from functools import lru_cache
from pathlib import Path

from careguide.corpus import Corpus
from careguide.retrieval.knowledge_base import COLLECTION_NAME, MedicalKnowledgeBase
from careguide.tools import MedicalToolRegistry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPEN_CORPUS_ROOT = PROJECT_ROOT / 'data' / 'careguide_kuake_rag_v1'


def semantic_embedding():
    from careguide.retrieval.embedding_provider import OpenAICompatibleEmbeddingFunction
    key = os.environ.get('CAREGUIDE_EMBEDDING_API_KEY', '')
    if not key or key.startswith('YOUR_'):
        raise ValueError('Semantic retrieval requires CAREGUIDE_EMBEDDING_API_KEY')
    return OpenAICompatibleEmbeddingFunction(
        api_key=key,
        base_url=os.environ.get('CAREGUIDE_EMBEDDING_BASE_URL', 'https://api.siliconflow.cn/v1'),
        model=os.environ.get('CAREGUIDE_EMBEDDING_MODEL', 'Qwen/Qwen3-Embedding-0.6B'),
        dimensions=int(os.environ.get('CAREGUIDE_EMBEDDING_DIMENSIONS', '1024')),
        encoding_format=os.environ.get('CAREGUIDE_EMBEDDING_FORMAT', 'float'))


def memory_embedding():
    """Memory has its own query instruction; RAG provider settings stay unchanged."""
    mode = os.environ.get('CAREGUIDE_MEMORY_VECTOR_MODE', 'auto').strip().casefold()
    if mode not in {'auto', 'disabled', 'remote'}:
        raise ValueError('Unknown SQLite memory vector mode')
    if mode == 'disabled':
        return None
    key = os.environ.get('CAREGUIDE_EMBEDDING_API_KEY', '')
    if not key or key.startswith('YOUR_'):
        if mode == 'remote':
            raise ValueError('Memory vector retrieval requires CAREGUIDE_EMBEDDING_API_KEY')
        return None
    from careguide.retrieval.embedding_provider import OpenAICompatibleEmbeddingFunction
    return OpenAICompatibleEmbeddingFunction(api_key=key,
        base_url=os.environ.get('CAREGUIDE_EMBEDDING_BASE_URL', 'https://api.siliconflow.cn/v1'),
        model=os.environ.get('CAREGUIDE_EMBEDDING_MODEL', 'Qwen/Qwen3-Embedding-0.6B'),
        dimensions=int(os.environ.get('CAREGUIDE_EMBEDDING_DIMENSIONS', '1024')),
        encoding_format=os.environ.get('CAREGUIDE_EMBEDDING_FORMAT', 'float'),
        query_instruction='Retrieve relevant user-authored personal memories and past conversation summaries.',
        batch_size=16, timeout=float(os.environ.get('CAREGUIDE_MEMORY_VECTOR_TIMEOUT', '1.5')),
        max_retries=0)


def intent_semantic_retriever(mode=None):
    """Separate business index and instruction, reusing the existing provider configuration."""
    from careguide.orchestration.semantic_intent import SemanticIntentRetriever, INSTRUCTION, record_response
    from careguide.retrieval.embedding_provider import OpenAICompatibleEmbeddingFunction
    mode = (mode or os.environ.get('CAREGUIDE_INTENT_SEMANTIC_MODE', 'auto')).strip().casefold()
    if mode not in {'auto', 'remote', 'local_ngram'}:
        raise ValueError('Unknown intent semantic mode')
    key = os.environ.get('CAREGUIDE_EMBEDDING_API_KEY', '')
    configured = bool(key.strip()) and not key.startswith('YOUR_')
    if mode == 'local_ngram' or mode == 'auto' and not configured:
        return None, 'local_ngram' if mode == 'local_ngram' else 'provider_unconfigured'
    if not configured:
        raise ValueError('Remote intent retrieval requires CAREGUIDE_EMBEDDING_API_KEY')
    timeout = float(os.environ.get('CAREGUIDE_INTENT_SEMANTIC_TIMEOUT', '3'))
    if not 0 < timeout <= 10:
        raise ValueError('Intent semantic timeout must be between 0 and 10 seconds')
    provider = OpenAICompatibleEmbeddingFunction(api_key=key,
        base_url=os.environ.get('CAREGUIDE_EMBEDDING_BASE_URL', 'https://api.siliconflow.cn/v1'),
        model=os.environ.get('CAREGUIDE_EMBEDDING_MODEL', 'Qwen/Qwen3-Embedding-0.6B'),
        dimensions=int(os.environ.get('CAREGUIDE_EMBEDDING_DIMENSIONS', '1024')),
        query_instruction=INSTRUCTION, timeout=timeout, max_retries=0, response_observer=record_response)
    return SemanticIntentRetriever(provider, timeout=timeout,
        index_path=os.environ.get('CAREGUIDE_INTENT_SEMANTIC_INDEX') or None), 'remote'


def semantic_reranker():
    from careguide.retrieval.reranker import DEFAULT_RERANK_INSTRUCTION, OpenAICompatibleReranker
    key = os.environ.get('CAREGUIDE_RERANK_API_KEY') or os.environ.get(
        'CAREGUIDE_EMBEDDING_API_KEY', '')
    if not key or key.startswith('YOUR_'):
        raise ValueError('Reranking requires CAREGUIDE_RERANK_API_KEY or embedding API key')
    return OpenAICompatibleReranker(
        api_key=key,
        base_url=os.environ.get('CAREGUIDE_RERANK_BASE_URL') or os.environ.get(
            'CAREGUIDE_EMBEDDING_BASE_URL', 'https://api.siliconflow.cn/v1'),
        model=os.environ.get('CAREGUIDE_RERANK_MODEL', 'Qwen/Qwen3-Reranker-0.6B'),
        instruction=os.environ.get('CAREGUIDE_RERANK_INSTRUCTION', '').strip() or DEFAULT_RERANK_INSTRUCTION,
        concurrency=int(os.environ.get('CAREGUIDE_RERANK_CONCURRENCY', '8')),
    )


def chroma_client():
    import chromadb
    configured=os.environ.get('CAREGUIDE_CHROMA_PATH','').strip()
    index_path=Path(configured) if configured else PROJECT_ROOT/'data'/'chroma'
    if not index_path.is_absolute():
        index_path=PROJECT_ROOT/index_path
    return chromadb.PersistentClient(path=str(index_path.resolve()),
                                    settings=chromadb.Settings(anonymized_telemetry=False))


@lru_cache(maxsize=1)
def get_registry():
    # Keep Corpus() as the frozen V1 compatibility baseline.  The product
    # runtime defaults to the versioned open-data corpus and can still select
    # any frozen corpus explicitly through CAREGUIDE_CORPUS_DIR.
    corpus_root = Path(os.environ.get('CAREGUIDE_CORPUS_DIR') or OPEN_CORPUS_ROOT)
    corpus = Corpus(corpus_root)
    from careguide.business_store import BusinessStore
    from careguide.knowledge_workflow import KnowledgeWorkflow
    corpus = KnowledgeWorkflow(BusinessStore(os.environ.get('CAREGUIDE_BUSINESS_DB',
        str(PROJECT_ROOT/'data'/'runtime'/'medical.sqlite3')))).extend(corpus)
    mode = os.environ.get('CAREGUIDE_RETRIEVAL_MODE', 'bm25').strip().casefold()
    kwargs = {}
    if mode in {'dense', 'hybrid'}:
        # Attach only. Never implicitly create, rebuild or fall back to hash vectors.
        from careguide.retrieval.knowledge_base import versioned_collection_name
        client = chroma_client()
        names = {collection.name for collection in client.list_collections()}
        name = versioned_collection_name(corpus) if versioned_collection_name(corpus) in names else COLLECTION_NAME
        kwargs = {'collection': client.get_collection(name, embedding_function=None),
                  'embedding': semantic_embedding(),
                  'dense_search': os.environ.get('CAREGUIDE_DENSE_SEARCH', 'ann').strip().casefold(),
                  'min_dense_score': float(os.environ.get('CAREGUIDE_MIN_DENSE_SCORE', '-1.0')),
                  'dense_weight': float(os.environ.get('CAREGUIDE_RRF_DENSE_WEIGHT', '4')),
                  'lexical_weight': float(os.environ.get('CAREGUIDE_RRF_BM25_WEIGHT', '1')),
                  'rrf_constant': int(os.environ.get('CAREGUIDE_RRF_CONSTANT', '60')),
                  'hybrid_candidates': int(os.environ.get(
                      'CAREGUIDE_HYBRID_CANDIDATES', '100'))}
        if mode == 'hybrid':
            kwargs['hybrid_fusion'] = os.environ.get('CAREGUIDE_HYBRID_FUSION', 'rrf')
            if os.environ.get('CAREGUIDE_HYBRID_WEIGHTS', '').strip():
                kwargs['hybrid_weights'] = json.loads(os.environ['CAREGUIDE_HYBRID_WEIGHTS'])
            sparse_model = os.environ.get('CAREGUIDE_BGE_SPARSE_MODEL_PATH', '').strip()
            sparse_checkpoint = os.environ.get('CAREGUIDE_BGE_SPARSE_CHECKPOINT', '').strip()
            if bool(sparse_model) != bool(sparse_checkpoint):
                raise ValueError('BGE sparse model and complete document checkpoint are both required')
            if sparse_model:
                from careguide.retrieval.bge_sparse import LocalBGESparseRetriever
                kwargs['sparse_retriever'] = LocalBGESparseRetriever(corpus, sparse_model, sparse_checkpoint)
        if os.environ.get('CAREGUIDE_RERANK_ENABLED', 'false').strip().casefold() == 'true':
            kwargs.update({'reranker': semantic_reranker(),
                           'rerank_candidates': int(os.environ.get(
                               'CAREGUIDE_RERANK_CANDIDATES', '20'))})
    return MedicalToolRegistry(MedicalKnowledgeBase(corpus, mode=mode, **kwargs))


@lru_cache(maxsize=1)
def get_chat_runtime():
    from careguide.agent_orchestrator import AgentOrchestrator
    from careguide.domain import AGENTS
    from careguide.intent_recognizer import IntentRecognizer
    from careguide.intent_v2_recognizer import V2IntentRecognizer
    from careguide.llm_utils import create_llm_client
    from careguide.performance_monitor import PerformanceMonitor
    from careguide.skill_registry import SkillRegistry
    from careguide.task_context import PersistentMemory, SessionMemory
    from careguide.trace_store import TraceStore
    from careguide.business_store import BusinessStore
    from careguide.action_store import ActionStore
    from careguide.appointment_service import AppointmentService
    from careguide.hospital_workflow import (DisabledHospitalBackend,
        FHIRHospitalBackend, InMemoryHospitalBackend)
    from careguide.knowledge_backends import EmbeddedKnowledgeBackend, WeKnoraMCPBackend
    key = os.environ.get('CAREGUIDE_LLM_API_KEY', '')
    base_url = os.environ.get('CAREGUIDE_LLM_BASE_URL') or 'https://api.xiaomimimo.com/v1'
    client = create_llm_client(key, base_url) if key and not key.startswith('YOUR_') else None
    model = os.environ.get('CAREGUIDE_LLM_MODEL', 'mimo-v2.6-flash')
    orchestration = os.environ.get('CAREGUIDE_ORCHESTRATION_MODE', 'unified').strip().casefold()
    if orchestration not in {'unified', 'langgraph', 'direct'}:
        raise ValueError('Unknown orchestration mode')
    if orchestration == 'unified':
        # Keep legacy recognition offline. UnifiedRuntime owns the explicit,
        # budgeted semantic-candidate phase before its single planner decision.
        recognizer = IntentRecognizer(embedding_mode='disabled')
    else:
        embedding_mode = os.environ.get('CAREGUIDE_INTENT_EMBEDDING_MODE', 'local_ngram')
        embedding = semantic_embedding() if embedding_mode == 'remote' else None
        taxonomy = os.environ.get('CAREGUIDE_INTENT_TAXONOMY', 'v1').strip().casefold()
        if taxonomy not in {'v1', 'v2', 'v3', 'qic'}:
            raise ValueError('Unknown intent taxonomy')
        if taxonomy == 'qic':
            from careguide.qic_intent_recognizer import QICIntentRecognizer
            recognizer_class = QICIntentRecognizer
        elif taxonomy == 'v3':
            from careguide.intent_v3_recognizer import V3IntentRecognizer
            recognizer_class = V3IntentRecognizer
        else:
            recognizer_class = V2IntentRecognizer if taxonomy == 'v2' else IntentRecognizer
        recognizer_kwargs = {'embedding': embedding, 'embedding_mode': embedding_mode}
        if taxonomy == 'qic':
            recognizer_kwargs.update({
                'llm_gate_margin': float(os.environ.get('CAREGUIDE_QIC_LLM_GATE_MARGIN', '0.40')),
                'llm_override_confidence': float(os.environ.get(
                    'CAREGUIDE_QIC_LLM_OVERRIDE_CONFIDENCE', '0.95')),
            })
            qic_backend = os.environ.get('CAREGUIDE_QIC_CLASSIFIER_BACKEND', 'fusion').strip().casefold()
            if qic_backend not in {'fusion', 'transformer'}:
                raise ValueError('Unknown CareGuide QIC classifier backend')
            if qic_backend == 'transformer':
                from careguide.transformer_intent_classifier import TransformerIntentClassifier
                model_path = os.environ.get('CAREGUIDE_QIC_MODEL_PATH', '').strip()
                if not model_path:
                    raise ValueError('Transformer QIC backend requires CAREGUIDE_QIC_MODEL_PATH')
                recognizer_kwargs['semantic_classifier'] = TransformerIntentClassifier(
                    model_path,
                    device=os.environ.get('CAREGUIDE_QIC_MODEL_DEVICE', 'auto').strip(),
                )
        recognizer = recognizer_class(client, model, **recognizer_kwargs)
    skills = SkillRegistry(root_dir=PROJECT_ROOT/'data'/'skills',
                           tool_scopes={agent.value: set(spec.tools) for agent, spec in AGENTS.items()})
    skills.load()
    trace_path = PROJECT_ROOT/'data'/'traces'/'requests.jsonl' if os.environ.get('CAREGUIDE_PERSIST_TRACES') == 'true' else None
    registry = get_registry()
    # Entity extraction and retrieval must inspect the same corpus.  The
    # recognizer constructors retain Corpus() for isolated V1/V2 tests.
    recognizer.corpus = registry.corpus
    knowledge_mode = os.environ.get('CAREGUIDE_KNOWLEDGE_BACKEND', 'embedded').strip().casefold()
    if knowledge_mode == 'weknora_mcp':
        knowledge_backend = WeKnoraMCPBackend(
            os.environ.get('CAREGUIDE_WEKNORA_MCP_URL', ''),
            token=os.environ.get('CAREGUIDE_WEKNORA_MCP_TOKEN', ''),
            search_tool=os.environ.get('CAREGUIDE_WEKNORA_SEARCH_TOOL', 'hybrid_search'),
            knowledge_base_id=os.environ.get('CAREGUIDE_WEKNORA_KB_ID', ''))
    elif knowledge_mode == 'embedded':
        knowledge_backend = EmbeddedKnowledgeBackend(registry.knowledge, list(registry.corpus.records))
    else:
        raise ValueError('Unknown knowledge backend')
    hospital_mode = os.environ.get('CAREGUIDE_HOSPITAL_BACKEND', 'disabled').strip().casefold()
    if hospital_mode == 'fhir':
        hospital_backend = FHIRHospitalBackend(os.environ.get('CAREGUIDE_FHIR_BASE_URL', 'http://127.0.0.1:8080/fhir'))
    elif hospital_mode == 'memory':
        hospital_backend = InMemoryHospitalBackend()
    elif hospital_mode == 'disabled':
        hospital_backend = DisabledHospitalBackend()
    else:
        raise ValueError('Unknown hospital backend')
    business_path = os.environ.get('CAREGUIDE_BUSINESS_DB', str(PROJECT_ROOT/'data'/'runtime'/'medical.sqlite3'))
    appointment_service = AppointmentService(hospital_backend, ActionStore(business_path))
    engine = AgentOrchestrator(recognizer, registry, client=client, model=model,
        skills=skills, traces=TraceStore(trace_path),
        max_tool_rounds=int(os.environ.get('CAREGUIDE_MAX_TOOL_ROUNDS', '5')),
        business_store=BusinessStore(business_path), knowledge_backend=knowledge_backend,
        appointment_service=appointment_service)
    memory_mode = os.environ.get('CAREGUIDE_MEMORY_MODE', 'sqlite')
    if memory_mode == 'redis_chroma':
        from careguide.conversation_memory import MemoryManager
        manager = MemoryManager(redis_url=os.environ.get('CAREGUIDE_REDIS_URL', 'redis://127.0.0.1:6380/0'),
            chroma_host=os.environ.get('CAREGUIDE_CHROMA_HOST', '127.0.0.1'),
            chroma_port=int(os.environ.get('CAREGUIDE_CHROMA_PORT', '8002')),
            chroma_path=str(PROJECT_ROOT/'data'/'chroma_memory'), api_key=key,
            base_url=base_url, model=model, client=client,
            embedding_function=semantic_embedding() if os.environ.get('CAREGUIDE_MEMORY_EMBEDDING_MODE') == 'remote' else None,
            collection_prefix='careguide_semantic' if os.environ.get('CAREGUIDE_MEMORY_EMBEDDING_MODE') == 'remote' else 'careguide',
            profile_path=business_path,
            legacy_profile_projection=os.environ.get('CAREGUIDE_LEGACY_PROFILE_PROJECTION', '0') == '1')
        memory = PersistentMemory(manager)
    elif memory_mode == 'session':
        memory = SessionMemory()
    elif memory_mode == 'sqlite':
        from careguide.sqlite_memory import SQLiteMemoryBackend
        cache_mode = os.environ.get('CAREGUIDE_MEMORY_CACHE', 'none')
        if cache_mode not in {'none', 'redis'}:
            raise ValueError('Unknown SQLite memory cache')
        memory = SQLiteMemoryBackend(business_path, client=client, model=model,
            embedding=memory_embedding(),
            vector_timeout=float(os.environ.get('CAREGUIDE_MEMORY_VECTOR_TIMEOUT', '1.5')),
            min_score=float(os.environ.get('CAREGUIDE_MEMORY_VECTOR_MIN_SCORE', '0.3')),
            redis_url=os.environ.get('CAREGUIDE_REDIS_URL', 'redis://127.0.0.1:6380/0')
            if cache_mode == 'redis' else None)
    else:
        raise ValueError('Unknown memory mode')
    if memory_mode == 'sqlite' or os.environ.get('CAREGUIDE_SESSION_ARCHIVE', '0') == '1':
        from careguide.archived_memory import ArchivedMemory
        from careguide.context_memory import ContextSettings
        if int(os.environ.get('CAREGUIDE_MEMORY_REVIEW_INTERVAL', '0')) and memory_mode not in {'redis_chroma', 'sqlite'}:
            raise ValueError('Periodic profile review requires sqlite or redis_chroma mode')
        settings = None
        if os.environ.get('CAREGUIDE_TOKEN_CONTEXT', '0') == '1':
            if client is None:
                raise ValueError('Token context requires a configured summary model client')
            settings = ContextSettings(
                window=int(os.environ.get('CAREGUIDE_CONTEXT_WINDOW', '32768')),
                input_cap=int(os.environ.get('CAREGUIDE_CONTEXT_INPUT_CAP', '24000')),
                output_reserve=int(os.environ.get('CAREGUIDE_CONTEXT_OUTPUT_RESERVE', '2048')),
                margin=int(os.environ.get('CAREGUIDE_CONTEXT_MARGIN', '1024')),
                trigger_ratio=float(os.environ.get('CAREGUIDE_CONTEXT_TRIGGER_RATIO', '.85')),
                target_ratio=float(os.environ.get('CAREGUIDE_CONTEXT_TARGET_RATIO', '.60')),
                model_windows=json.loads(os.environ.get('CAREGUIDE_CONTEXT_MODEL_WINDOWS', '{}')))
        memory = ArchivedMemory(memory, business_path, client=client, model=model, settings=settings,
            review_interval=int(os.environ.get('CAREGUIDE_MEMORY_REVIEW_INTERVAL', '0')))
    elif os.environ.get('CAREGUIDE_TOKEN_CONTEXT', '0') == '1':
        raise ValueError('Token context requires the durable session archive')
    if orchestration in {'unified', 'langgraph'}:
        from careguide.orchestration.runtime import OrchestrationRuntime
        from careguide.orchestration.budget import BudgetLimits
        runtime_class, runtime_options = OrchestrationRuntime, {}
        if orchestration == 'unified':
            from careguide.orchestration.unified import UnifiedRuntime
            from careguide.orchestration.understanding import LocalUnderstanding, LocalClassifier
            local_path = os.environ.get('CAREGUIDE_QIC_MODEL_PATH') or str(
                PROJECT_ROOT.parent/'models'/'CareGuide'/'careguide-intent-roberta-v1')
            classifier_mode = os.environ.get('CAREGUIDE_INTENT_CLASSIFIER_MODE', 'auto').strip().casefold()
            if classifier_mode not in {'auto', 'disabled'}:
                raise ValueError('Unknown intent classifier mode')
            classifier = (LocalClassifier(local_path, os.environ.get('CAREGUIDE_QIC_MODEL_DEVICE', 'auto'))
                          if classifier_mode == 'auto' else None)
            runtime_class = UnifiedRuntime
            semantic, semantic_mode = intent_semantic_retriever()
            runtime_options['understanding'] = LocalUnderstanding(registry.corpus, classifier,
                semantic=semantic, semantic_mode=semantic_mode,
                semantic_policy=os.environ.get('CAREGUIDE_INTENT_SEMANTIC_POLICY', 'parallel'),
                candidate_routing=os.environ.get('CAREGUIDE_INTENT_CANDIDATE_ROUTING', 'read_only'))
        engine = runtime_class(engine, business_path, **runtime_options, limits=BudgetLimits(
            model=int(os.environ.get('CAREGUIDE_WORKFLOW_MODEL_CALLS', '20')),
            tool=int(os.environ.get('CAREGUIDE_WORKFLOW_TOOL_CALLS', '30')),
            retry=int(os.environ.get('CAREGUIDE_WORKFLOW_RETRIES', '3')),
            replan=int(os.environ.get('CAREGUIDE_WORKFLOW_REPLANS', '3')),
            execution_seconds=float(os.environ.get('CAREGUIDE_WORKFLOW_EXECUTION_SECONDS', '120')),
            request_seconds=float(os.environ.get('CAREGUIDE_WORKFLOW_REQUEST_SECONDS', '45'))))
    monitor = PerformanceMonitor(engine, engine.tool_manager, webhook_url=None)
    return engine, memory, monitor
