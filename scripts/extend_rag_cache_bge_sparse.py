"""Add genuine local BGE-M3 sparse scores to an existing Dense/BM25 dev cache."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sqlite3
import time
from pathlib import Path

from careguide.corpus import Corpus
from careguide.retrieval.bge_sparse import BGEM3SparseEncoder
from careguide.retrieval.knowledge_base import MedicalKnowledgeBase
from careguide.retrieval.sparse_index import SparseLexicalIndex
from scripts.cache_rag_recall_scores import atomic_json
from scripts.download_bge_sparse_model import file_hash
from scripts.evaluate_rag_score_fusion import validate_cache
from scripts.run_careguide_kuake_rag_evaluation import CORPUS_DIR, load_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--quantization', choices=('none', 'dynamic_int8'), default='none')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--reference-dir', type=Path,
                        help='Optional unmodified FlagEmbedding 1.2.11 source files for smoke parity')
    args = parser.parse_args()
    if args.output.exists() or args.batch_size < 1 or not 2 <= args.max_length <= 8192:
        parser.error('Invalid arguments or output already exists')
    cache = json.loads(gzip.decompress(args.cache.read_bytes())) if args.cache.suffix == '.gz' else json.loads(args.cache.read_bytes())
    validate_cache(cache)
    if any('bge_sparse' in row['streams'] for row in cache['rows']):
        raise ValueError('BGE stream already present')
    lock = json.loads((args.model_path/'careguide_model_lock.json').read_text())
    if lock['model'] != 'BAAI/bge-m3':
        raise ValueError('This experiment requires the actual BAAI/bge-m3 model')
    for name, expected in lock['files'].items():
        if file_hash(args.model_path/name) != expected['sha256']:
            raise ValueError('BGE-M3 local model checksum mismatch')
    print('model_files_verified', flush=True)
    corpus = Corpus(CORPUS_DIR)
    cases, _ = load_cases('dev')
    documents = [('doc:'+r.id, MedicalKnowledgeBase.retrieval_text(r)) for r in corpus.records]
    queries = [('query:'+c['id'], c['query']) for c in cases]
    encoder = BGEM3SparseEncoder(args.model_path, threads=args.threads, quantization=args.quantization)
    identity = {'model': lock['model'], 'revision': lock['revision'], 'files': lock['files'],
                'corpus_fingerprint': corpus.fingerprint, 'max_length': args.max_length,
                'dev_sha256': cache['metadata']['dev_sha256'],
                'query_fingerprint': cache['metadata']['query_fingerprint'],
                'quantization': args.quantization, 'query_instruction': None,
                'encoder': 'AutoModel+official_sparse_linear+ReLU+token_max',
                'document_text': 'MedicalKnowledgeBase.retrieval_text'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.benchmark:
        selected = queries[:8]+documents[:8]
        start = time.perf_counter()
        vectors = []
        for i in range(0, len(selected), args.batch_size):
            vectors.extend(encoder.encode_batch([text for _, text in selected[i:i+args.batch_size]],
                                                max_length=args.max_length))
        elapsed = time.perf_counter()-start
        parity = None
        if args.reference_dir:
            import ast
            import importlib.util
            import sys
            from collections import defaultdict
            from types import SimpleNamespace, MethodType
            import numpy as np
            import torch
            modeling = args.reference_dir/'modeling.py'
            wrapper = args.reference_dir/'bge_m3.py'
            spec = importlib.util.spec_from_file_location('careguide_bge_reference', modeling)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            functions = [node for node in ast.walk(ast.parse(wrapper.read_text()))
                         if isinstance(node, ast.FunctionDef) and node.name == '_process_token_weights']
            if len(functions) != 1:
                raise ValueError('Unexpected official reference wrapper')
            namespace = {'np': np, 'defaultdict': defaultdict, 'self': encoder}
            exec(compile(ast.Module(body=functions, type_ignores=[]), str(wrapper), 'exec'), namespace)
            reference = SimpleNamespace(model=encoder.model, sparse_linear=encoder.head, normlized=True)
            reference.sparse_embedding = MethodType(module.BGEM3Model.sparse_embedding, reference)
            expected = []
            for i in range(0, len(selected), args.batch_size):
                inputs = encoder.tokenizer([text for _, text in selected[i:i+args.batch_size]],
                                           truncation=True, max_length=args.max_length,
                                           padding=True, return_tensors='pt')
                with torch.inference_mode():
                    output = module.BGEM3ForInference.forward(reference, text_input=inputs,
                        return_dense=False, return_sparse=True, return_colbert=False,
                        return_sparse_embedding=False)['sparse_vecs'].squeeze(-1).numpy()
                expected.extend(namespace['_process_token_weights'](values, ids)
                                for values, ids in zip(output, inputs['input_ids'].tolist()))
            if any(set(actual) != set(target) for actual, target in zip(vectors, expected)):
                raise ValueError('Local sparse token keys disagree with official reference')
            difference = max((abs(actual[key]-float(target[key])) for actual, target in zip(vectors, expected)
                              for key in actual), default=0.0)
            if difference > 1e-6:
                raise ValueError('Local sparse weights disagree with official reference')
            parity = {'reference_version': 'FlagEmbedding 1.2.11', 'case_count': len(vectors),
                      'maximum_absolute_weight_error': difference,
                      'modeling_sha256': file_hash(modeling), 'wrapper_sha256': file_hash(wrapper),
                      'scope': 'Same real encoder/head; independent official sparse forward and token reduction.'}
        atomic_json(args.output, {'identity': identity, 'elapsed_seconds': elapsed, 'reference_parity': parity,
                                  'rows': [{'id': key, 'vector': vector} for (key, _), vector in zip(selected, vectors)]})
        print('benchmark_seconds', elapsed, 'reference_parity', parity, flush=True)
        return
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(args.checkpoint)
    db.execute('create table if not exists config (identity text not null)')
    db.execute('create table if not exists vectors (id text primary key, text_sha256 text not null, weights text not null)')
    prior = db.execute('select identity from config').fetchone()
    serialized = json.dumps(identity, sort_keys=True)
    if prior and prior[0] != serialized:
        raise ValueError('Sparse checkpoint configuration mismatch')
    if not prior:
        db.execute('insert into config values (?)', (serialized,))
        db.commit()
    items = documents+queries
    pending = []
    for key, text in items:
        text_sha = hashlib.sha256(text.encode()).hexdigest()
        old = db.execute('select text_sha256 from vectors where id=?', (key,)).fetchone()
        if old and old[0] != text_sha:
            raise ValueError('Sparse checkpoint input text changed')
        if not old:
            pending.append((key, text, text_sha))
    # Bucket by true tokenizer length to minimize CPU padding, restore by ID.
    lengths = encoder.tokenizer([text for _, text, _ in pending], truncation=True,
                                max_length=args.max_length)['input_ids'] if pending else []
    pending = [item for _, item in sorted(zip(map(len, lengths), pending), key=lambda pair: pair[0])]
    started = time.perf_counter()
    last_progress = started
    for offset in range(0, len(pending), args.batch_size):
        batch = pending[offset:offset+args.batch_size]
        vectors = encoder.encode_batch([text for _, text, _ in batch], max_length=args.max_length)
        if len(vectors) != len(batch):
            raise ValueError('Sparse model output count mismatch')
        db.executemany('insert into vectors values (?,?,?)',
                       [(key, sha, json.dumps(vector)) for (key, _, sha), vector in zip(batch, vectors)])
        db.commit()
        done = min(offset+len(batch), len(pending))
        if time.perf_counter()-last_progress >= 20 or done == len(pending):
            elapsed = time.perf_counter()-started
            print(f'encoded {done}/{len(pending)} pending texts; {elapsed:.1f}s elapsed', flush=True)
            atomic_json(args.output.with_suffix('.progress.json'), {'encoded_this_run': done,
                'pending_at_start': len(pending), 'elapsed_seconds': elapsed,
                'phase': 'batch_encoding', 'batch_encoding_complete': done == len(pending), 'complete': False})
            last_progress = time.perf_counter()
    # Dynamic activation quantization can depend on other texts in a batch.
    # Score the final queries as singleton requests, matching the runtime path.
    # Document vectors stay fixed; this pass is resumable by safely repeating it.
    for offset, (key, text) in enumerate(queries, 1):
        vector = encoder.encode_batch([text], max_length=args.max_length)[0]
        db.execute('update vectors set weights=? where id=?', (json.dumps(vector), key))
        if offset % 32 == 0 or offset == len(queries):
            db.commit()
        if offset % 200 == 0 or offset == len(queries):
            print(f'single-query encoding {offset}/{len(queries)}', flush=True)
            atomic_json(args.output.with_suffix('.progress.json'), {'phase': 'single_query_encoding',
                'queries_encoded': offset, 'query_count': len(queries), 'complete': False})
    encoded = {key: json.loads(weights) for key, weights in db.execute('select id,weights from vectors')}
    sparse = SparseLexicalIndex({key.removeprefix('doc:'): encoded[key] for key, _ in documents})
    for case, row in zip(cases, cache['rows']):
        row['streams']['bge_sparse'] = sparse.search(encoded['query:'+case['id']],
                                                    top_k=cache['metadata']['candidate_k'])
    identity['processing_seconds_this_run'] = time.perf_counter()-started
    identity['document_count'] = len(documents)
    identity['query_count'] = len(queries)
    identity['query_batch_size'] = 1
    cache['metadata']['bge_sparse'] = identity
    cache['metadata']['additional_embedding_api_requests'] = 0
    cache['metadata']['base_cache_sha256'] = hashlib.sha256(args.cache.read_bytes()).hexdigest()
    atomic_json(args.output, cache)
    db.close()
    atomic_json(args.output.with_suffix('.progress.json'), {'phase': 'complete', 'complete': True,
        'document_count': len(documents), 'query_count': len(queries)})
    print('three_way_score_cache_complete', flush=True)


if __name__ == '__main__':
    main()
