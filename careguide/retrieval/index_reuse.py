"""Explicit reuse of real cached vectors only after complete input verification."""
from __future__ import annotations

import hashlib
import math
import uuid

from careguide.retrieval.knowledge_base import MedicalKnowledgeBase, versioned_collection_name


def reuse_verified_index(client, corpus, source_name, binding, *, model, dimensions, batch_size=256):
    """Never embed, delete, modify a source collection, or overwrite a target.

    ``binding`` must be exported from the trusted original corpus, using its
    actual embedding input construction. Stored documents alone cannot prove
    that titles and keywords used in the embedding input are unchanged.
    """
    if not 1 <= batch_size <= 512 or dimensions <= 0 or not model:
        raise ValueError('Invalid index reuse configuration')
    target_name = versioned_collection_name(corpus)
    if target_name in {item.name for item in client.list_collections()}:
        raise ValueError('Target collection exists; refusing overwrite')
    source = client.get_collection(source_name, embedding_function=None)
    metadata = source.metadata or {}
    expected = {'corpus_fingerprint': binding.get('fingerprint'), 'embedding_model': model,
                'embedding_dimensions': dimensions, 'hnsw:space': 'cosine'}
    if not binding.get('fingerprint') or any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError('Original corpus or embedding identity mismatch')
    records = {record.id: record for record in corpus.records}
    hashes = {key: hashlib.sha256(MedicalKnowledgeBase.retrieval_text(record).encode()).hexdigest()
              for key, record in records.items()}
    if binding.get('count') != len(records) or hashes != binding.get('retrieval_hashes'):
        raise ValueError('Embedding input changed; cached vectors cannot be reused')
    ids = source.get(include=[])['ids']
    if len(ids) != len(records) or set(ids) != records.keys():
        raise ValueError('Cached index records disagree with verified corpus')
    # Preflight every record and vector before creating anything. Keep only
    # the already verified vectors in memory; this is a bounded local job.
    batches = []
    for offset in range(0, len(ids), batch_size):
        rows = source.get(ids=ids[offset:offset+batch_size],
                          include=['documents', 'metadatas', 'embeddings'])
        row_ids, vectors = rows['ids'], rows['embeddings']
        if len(row_ids) != len(ids[offset:offset+batch_size]) or set(row_ids) != set(ids[offset:offset+batch_size]):
            raise ValueError('Cached batch records incomplete')
        if vectors is None or len(vectors) != len(row_ids):
            raise ValueError('Cached embeddings missing')
        for index, key in enumerate(row_ids):
            record, vector = records[key], vectors[index]
            stored = rows['metadatas'][index] or {}
            if (rows['documents'][index] != record.content or stored.get('record_id') != key
                    or stored.get('source_id') != record.source_id or stored.get('kind') != record.kind.value):
                raise ValueError('Cached record content or source mismatch')
            if len(vector) != dimensions or not all(math.isfinite(float(value)) for value in vector):
                raise ValueError('Invalid cached vector')
        batches.append((row_ids, vectors))
    target_metadata = {**expected, 'corpus_fingerprint': corpus.fingerprint,
                       'reused_from': source_name, 'reuse_complete': True}
    # The runtime cannot discover a staging name. Publish only after complete
    # verification by renaming; Chroma 0.5 rejects a metadata update containing
    # hnsw:space even when its value is unchanged.
    staging_name = target_name+'_pending_'+uuid.uuid4().hex[:12]
    target = client.create_collection(staging_name, metadata=target_metadata, embedding_function=None)
    for keys, vectors in batches:
        target.add(ids=keys, embeddings=vectors,
            documents=[records[key].content for key in keys],
            metadatas=[{'record_id': key, 'source_id': records[key].source_id,
                        'kind': records[key].kind.value} for key in keys])
    actual = target.get(include=[])['ids']
    if len(actual) != len(records) or set(actual) != records.keys():
        raise ValueError('Target verification incomplete; retained for inspection')
    target.modify(name=target_name)
    return {'source': source_name, 'target': target_name, 'records': len(records),
            'embedding_model': model, 'dimensions': dimensions, 'paid_calls': 0,
            'source_modified': False, 'target_overwritten': False}
