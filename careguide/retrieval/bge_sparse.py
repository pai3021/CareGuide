"""Local BGE-M3 sparse inference using the official encoder and sparse head.

Equivalent sparse path: ReLU(sparse_linear(last_hidden_state)), max per token ID,
excluding CLS/EOS/PAD/UNK. No dense-vector thresholding or BM25 approximation.
Reference: FlagOpen/FlagEmbedding, encoder_only/m3 sparse embedding/inference.
Heavy dependencies are imported only when a local model is explicitly selected.
"""
from __future__ import annotations

from pathlib import Path
import math


def token_weights_to_sparse(ids, weights, excluded):
    if len(ids) != len(weights) or any(not math.isfinite(float(w)) for w in weights):
        raise ValueError('Invalid sparse token weights')
    result = {}
    for token, weight in zip(ids, weights):
        if token not in excluded and weight > 0:
            key = str(token)
            result[key] = max(result.get(key, 0.0), float(weight))
    return result


class BGEM3SparseEncoder:
    def __init__(self, model_path, *, threads=6, quantization='none'):
        if quantization not in {'none', 'dynamic_int8'}:
            raise ValueError('Unknown quantization')
        self.path = Path(model_path)
        if not all((self.path/name).is_file() for name in ('pytorch_model.bin', 'config.json', 'sparse_linear.pt')):
            raise ValueError('BGE-M3 encoder and sparse head must already exist locally')
        import torch
        from transformers import AutoModel, AutoTokenizer
        if threads is not None:
            torch.set_num_threads(threads)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.path), local_files_only=True,
                                                      trust_remote_code=False, use_fast=True)
        self.model = AutoModel.from_pretrained(str(self.path), local_files_only=True,
                                              trust_remote_code=False, torch_dtype=torch.float32)
        self.head = torch.nn.Linear(self.model.config.hidden_size, 1)
        self.head.load_state_dict(torch.load(self.path/'sparse_linear.pt', map_location='cpu',
                                             weights_only=True), strict=True)
        if quantization == 'dynamic_int8':
            self.model = torch.ao.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
        self.model.eval()
        self.head.eval()
        self.excluded = {getattr(self.tokenizer, name+'_token_id')
                         for name in ('cls', 'eos', 'pad', 'unk')}
        self.quantization = quantization

    def encode_batch(self, texts, *, max_length=512):
        import torch
        inputs = self.tokenizer(list(texts), truncation=True, max_length=max_length,
                                padding=True, return_tensors='pt')
        with torch.inference_mode():
            hidden = self.model(**inputs, return_dict=True).last_hidden_state
            weights = torch.relu(self.head(hidden)).squeeze(-1).tolist()
        return [token_weights_to_sparse(ids, values, self.excluded)
                for ids, values in zip(inputs['input_ids'].tolist(), weights)]


class LocalBGESparseRetriever:
    """Attach a complete, verified document checkpoint without re-encoding it."""
    def __init__(self, corpus, model_path, checkpoint):
        import hashlib
        import json
        import sqlite3
        from contextlib import closing
        from careguide.retrieval.knowledge_base import MedicalKnowledgeBase
        from careguide.retrieval.sparse_index import SparseLexicalIndex
        model_path = Path(model_path)
        lock = json.loads((model_path/'careguide_model_lock.json').read_text())
        with closing(sqlite3.connect(Path(checkpoint).resolve().as_uri()+'?mode=ro', uri=True)) as db:
            identity = json.loads(db.execute('select identity from config').fetchone()[0])
            documents = {key.removeprefix('doc:'): (sha, json.loads(vector)) for key, sha, vector
                         in db.execute("select id,text_sha256,weights from vectors where id like 'doc:%'")}
        if (identity['corpus_fingerprint'] != corpus.fingerprint
                or identity['model'] != 'BAAI/bge-m3'
                or identity['revision'] != lock['revision'] or identity['files'] != lock['files']
                or set(documents) != {r.id for r in corpus.records}):
            raise ValueError('BGE sparse checkpoint corpus/model/record identity mismatch')
        for record in corpus.records:
            if documents[record.id][0] != hashlib.sha256(
                    MedicalKnowledgeBase.retrieval_text(record).encode()).hexdigest():
                raise ValueError('BGE sparse checkpoint document text changed')
        for name, expected in lock['files'].items():
            with (model_path/name).open('rb') as handle:
                if hashlib.file_digest(handle, 'sha256').hexdigest() != expected['sha256']:
                    raise ValueError('BGE sparse model weights changed')
        self.index = SparseLexicalIndex({key: vector for key, (_, vector) in documents.items()})
        self.encoder = BGEM3SparseEncoder(model_path, threads=None, quantization=identity['quantization'])
        self.max_length = identity['max_length']
        self.metadata = identity

    def search(self, query, *, eligible_ids, top_k):
        vector = self.encoder.encode_batch([query], max_length=self.max_length)[0]
        return self.index.search(vector, eligible_ids=eligible_ids, top_k=top_k)
