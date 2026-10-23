"""Explicit medical retrieval configuration. Does not load any .env file."""
import os
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


def semantic_reranker():
    from careguide.retrieval.reranker import OpenAICompatibleReranker
    key = os.environ.get('CAREGUIDE_RERANK_API_KEY') or os.environ.get(
        'CAREGUIDE_EMBEDDING_API_KEY', '')
    if not key or key.startswith('YOUR_'):
        raise ValueError('Reranking requires CAREGUIDE_RERANK_API_KEY or embedding API key')
    return OpenAICompatibleReranker(
        api_key=key,
        base_url=os.environ.get('CAREGUIDE_RERANK_BASE_URL') or os.environ.get(
            'CAREGUIDE_EMBEDDING_BASE_URL', 'https://api.siliconflow.cn/v1'),
        model=os.environ.get('CAREGUIDE_RERANK_MODEL', 'Qwen/Qwen3-Reranker-0.6B'),
        concurrency=int(os.environ.get('CAREGUIDE_RERANK_CONCURRENCY', '8')),
    )


def chroma_client():
    import chromadb
    return chromadb.PersistentClient(path=str(PROJECT_ROOT/'data'/'chroma'),
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
                  'min_dense_score': float(os.environ.get('CAREGUIDE_MIN_DENSE_SCORE', '-1.0')),
                  'dense_weight': float(os.environ.get('CAREGUIDE_RRF_DENSE_WEIGHT', '4')),
                  'lexical_weight': float(os.environ.get('CAREGUIDE_RRF_BM25_WEIGHT', '1')),
                  'hybrid_candidates': int(os.environ.get(
                      'CAREGUIDE_HYBRID_CANDIDATES', '100'))}
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
    client = create_llm_client(key, os.environ.get('CAREGUIDE_LLM_BASE_URL') or None) if key and not key.startswith('YOUR_') else None
    model = os.environ.get('CAREGUIDE_LLM_MODEL', 'gpt-4o-mini')
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
    memory_mode = os.environ.get('CAREGUIDE_MEMORY_MODE', 'session')
    if memory_mode == 'redis_chroma':
        from careguide.conversation_memory import MemoryManager
        manager = MemoryManager(redis_url=os.environ.get('CAREGUIDE_REDIS_URL', 'redis://127.0.0.1:6380/0'),
            chroma_host=os.environ.get('CAREGUIDE_CHROMA_HOST', '127.0.0.1'),
            chroma_port=int(os.environ.get('CAREGUIDE_CHROMA_PORT', '8002')),
            chroma_path=str(PROJECT_ROOT/'data'/'chroma_memory'), api_key=key,
            base_url=os.environ.get('CAREGUIDE_LLM_BASE_URL') or None, model=model, client=client,
            embedding_function=semantic_embedding() if os.environ.get('CAREGUIDE_MEMORY_EMBEDDING_MODE') == 'remote' else None,
            collection_prefix='careguide_semantic' if os.environ.get('CAREGUIDE_MEMORY_EMBEDDING_MODE') == 'remote' else 'careguide',
            profile_path=business_path)
        memory = PersistentMemory(manager)
    elif memory_mode == 'session':
        memory = SessionMemory()
    else:
        raise ValueError('Unknown memory mode')
    orchestration = os.environ.get('CAREGUIDE_ORCHESTRATION_MODE', 'langgraph').strip().casefold()
    if orchestration not in {'langgraph', 'direct'}:
        raise ValueError('Unknown orchestration mode')
    if orchestration == 'langgraph':
        from careguide.orchestration.runtime import OrchestrationRuntime
        from careguide.orchestration.budget import BudgetLimits
        engine = OrchestrationRuntime(engine, business_path, limits=BudgetLimits(
            model=int(os.environ.get('CAREGUIDE_WORKFLOW_MODEL_CALLS', '20')),
            tool=int(os.environ.get('CAREGUIDE_WORKFLOW_TOOL_CALLS', '30')),
            retry=int(os.environ.get('CAREGUIDE_WORKFLOW_RETRIES', '3')),
            replan=int(os.environ.get('CAREGUIDE_WORKFLOW_REPLANS', '3')),
            execution_seconds=float(os.environ.get('CAREGUIDE_WORKFLOW_EXECUTION_SECONDS', '120')),
            request_seconds=float(os.environ.get('CAREGUIDE_WORKFLOW_REQUEST_SECONDS', '45'))))
    monitor = PerformanceMonitor(engine, engine.tool_manager, webhook_url=None)
    return engine, memory, monitor
