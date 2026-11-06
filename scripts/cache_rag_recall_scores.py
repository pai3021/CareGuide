"""Cache genuine dev retrieval scores once; no LLM, reranker or test access.

Remote embedding requires an explicit flag and a durable pre-request ledger.
Use an expendable COPY of the Chroma directory, never the live data directory.
An optional existing local BGE-M3 directory enables the third retrieval leg.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from urllib.parse import urlparse

from careguide.corpus import Corpus
from careguide.retrieval.embedding_provider import (DEFAULT_QUERY_INSTRUCTION,
                                                   OpenAICompatibleEmbeddingFunction)
from careguide.retrieval.knowledge_base import MedicalKnowledgeBase, versioned_collection_name
from careguide.retrieval.sparse_index import SparseLexicalIndex
from scripts.private_config import load_config
from scripts.run_careguide_kuake_rag_evaluation import CORPUS_DIR, ROOT, load_cases


def atomic_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temporary.replace(path)


def query_fingerprint(cases):
    return hashlib.sha256(json.dumps([(c['id'], c['query']) for c in cases],
                                    ensure_ascii=False).encode()).hexdigest()


class RequestBudget:
    """Persist attempted requests BEFORE sending, including failed requests."""
    def __init__(self, path: Path, limit: int):
        if not 1 <= limit <= 32:
            raise ValueError('Request budget must be between 1 and 32')
        self.path, self.limit = path, limit
        self.used = json.loads(path.read_text())['attempted_requests'] if path.exists() else 0

    def reserve(self):
        if self.used >= self.limit:
            raise RuntimeError('Embedding request budget exhausted')
        self.used += 1
        atomic_json(self.path, {'attempted_requests': self.used, 'maximum_requests': self.limit,
                               'endpoint': 'https://api.siliconflow.cn/v1/embeddings'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate-k', type=int, default=100)
    parser.add_argument('--chroma-copy', type=Path)
    parser.add_argument('--config-file', type=Path)
    parser.add_argument('--allow-remote-embeddings', action='store_true')
    parser.add_argument('--max-api-requests', type=int, default=32)
    parser.add_argument('--local-bge-model', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refusing to overwrite an existing score cache')
    if args.candidate_k < 1:
        parser.error('candidate-k must be positive')
    if args.allow_remote_embeddings and not args.chroma_copy:
        parser.error('Remote mode requires an explicit Chroma COPY')
    if args.chroma_copy and args.chroma_copy.resolve() == (ROOT/'data/chroma').resolve():
        parser.error('Use an expendable Chroma copy, not the live index')
    if args.local_bge_model and not args.local_bge_model.is_dir():
        parser.error('BGE-M3 must already exist as a local model directory')
    cases, manifest = load_cases('dev')
    corpus = Corpus(CORPUS_DIR)
    knowledge = MedicalKnowledgeBase(corpus)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = [{'id': case['id'], 'gold_record_id': case['gold_record_id'], 'streams': {}}
            for case in cases]
    for case, row in zip(cases, rows):
        row['streams']['bm25'] = [(hit.chunk_id, hit.score) for hit in
                                 knowledge.lexical_index.search(case['query'], top_k=args.candidate_k)]
    print('BM25 scores cached for 1000 dev queries', flush=True)
    meta = {'version': 'careguide-rag-score-cache-v1', 'split': 'dev',
            'corpus_fingerprint': corpus.fingerprint,
            'dev_sha256': manifest['files']['dev.jsonl']['sha256'],
            'query_fingerprint': query_fingerprint(cases), 'candidate_k': args.candidate_k,
            'record_count': len(corpus.records), 'external_requests': 0,
            'query_instruction': DEFAULT_QUERY_INSTRUCTION}
    vector_path = args.output.with_suffix('.vectors.json')
    if args.chroma_copy:
        import chromadb
        from types import SimpleNamespace
        client = chromadb.PersistentClient(path=str(args.chroma_copy.resolve()),
                                          settings=chromadb.Settings(anonymized_telemetry=False))
        collection = client.get_collection(versioned_collection_name(corpus), embedding_function=None)
        model = 'Qwen/Qwen3-Embedding-0.6B'
        dimensions = 1024
        # Verify complete index identity BEFORE spending any requests.
        MedicalKnowledgeBase(corpus, collection=collection,
                             embedding=SimpleNamespace(model=model, dimensions=dimensions), mode='dense')
        identity = {'query_fingerprint': meta['query_fingerprint'], 'model': model,
                    'dimensions': dimensions, 'instruction': DEFAULT_QUERY_INSTRUCTION}
        if vector_path.exists():
            cached = json.loads(vector_path.read_text(encoding='utf-8'))
            if cached['identity'] != identity:
                raise ValueError('Cached query embeddings do not match this experiment')
            vectors = cached['vectors']
        else:
            vectors = []
        if (len(vectors) > len(cases) or any(len(v) != dimensions or
                any(not math.isfinite(x) for x in v) for v in vectors)):
            raise ValueError('Invalid cached query embeddings')
        budget = RequestBudget(args.output.with_suffix('.requests.json'), args.max_api_requests)
        if len(vectors) < len(cases):
            if not args.allow_remote_embeddings:
                raise ValueError('Missing query vectors; remote calls are opt-in')
            load_config(args.config_file)
            base_url = os.environ.get('CAREGUIDE_EMBEDDING_BASE_URL', '').rstrip('/')
            endpoint = urlparse(base_url)
            if (endpoint.scheme, endpoint.hostname, endpoint.path, endpoint.port) != (
                    'https', 'api.siliconflow.cn', '/v1', None) or endpoint.query or endpoint.fragment or endpoint.username is not None:
                raise ValueError('Only the explicitly authorized SiliconFlow endpoint is permitted')
            provider = OpenAICompatibleEmbeddingFunction(
                api_key=os.environ.get('CAREGUIDE_EMBEDDING_API_KEY', ''), base_url=base_url,
                model=model, dimensions=dimensions, batch_size=32, max_retries=0, timeout=60)
            needed = math.ceil((len(cases)-len(vectors))/32)
            if budget.used + needed > budget.limit:
                raise ValueError('Remaining budget cannot complete this cache')
            try:
                while len(vectors) < len(cases):
                    batch = cases[len(vectors):len(vectors)+32]
                    budget.reserve()
                    vectors.extend(provider.embed_queries([case['query'] for case in batch]))
                    atomic_json(vector_path, {'identity': identity, 'vectors': vectors})
                    print(f'query embeddings {len(vectors)}/{len(cases)}; requests {budget.used}', flush=True)
            finally:
                provider.close()
        meta.update({'embedding_model': model, 'embedding_dimensions': dimensions,
                     'external_requests': budget.used})
        for start in range(0, len(cases), 32):
            result = collection.query(query_embeddings=vectors[start:start+32],
                                      n_results=min(args.candidate_k, len(corpus.records)),
                                      include=['distances'])
            for row, ids, distances in zip(rows[start:start+32], result['ids'], result['distances']):
                row['streams']['dense'] = [(str(key), 1-float(distance))
                                           for key, distance in zip(ids, distances)]
        print('Dense scores read from copied index', flush=True)
    if args.local_bge_model:
        # Fail closed: never let a missing file silently trigger a model download.
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        from FlagEmbedding import BGEM3FlagModel
        model = BGEM3FlagModel(str(args.local_bge_model.resolve()), use_fp16=False, devices=['cpu'])
        documents = model.encode([knowledge.retrieval_text(r) for r in corpus.records], batch_size=8,
                                 max_length=512, return_dense=False, return_sparse=True,
                                 return_colbert_vecs=False)['lexical_weights']
        sparse = SparseLexicalIndex({record.id: {str(k): float(v) for k, v in vector.items()}
                                     for record, vector in zip(corpus.records, documents)})
        queries = model.encode([c['query'] for c in cases], batch_size=8, max_length=512,
                               return_dense=False, return_sparse=True,
                               return_colbert_vecs=False)['lexical_weights']
        if len(documents) != len(corpus.records) or len(queries) != len(cases):
            raise ValueError('BGE sparse encoding count mismatch')
        for row, vector in zip(rows, queries):
            row['streams']['bge_sparse'] = sparse.search(
                {str(k): float(v) for k, v in vector.items()}, top_k=args.candidate_k)
        meta['bge_sparse'] = {'encoder': 'BGEM3FlagModel', 'local_model_name': args.local_bge_model.name, 'max_length': 512,
                              'local_config_sha256': hashlib.sha256(
                                  (args.local_bge_model/'config.json').read_bytes()).hexdigest()}
    atomic_json(args.output, {'metadata': meta, 'rows': rows})
    print(f'Score cache saved: {args.output}', flush=True)


if __name__ == '__main__':
    main()
