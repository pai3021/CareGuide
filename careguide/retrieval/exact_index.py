"""Optional exhaustive cosine search over a verified immutable collection.

Loads document vectors once. Query scoring is restricted to eligible IDs before
similarity computation; no query handler creates, edits, or re-embeds an index.
"""
from __future__ import annotations

import numpy as np


def normalize_vectors(values):
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError('Expected finite vector matrix')
    lengths = np.sqrt(np.sum(matrix*matrix, axis=1, keepdims=True))
    if not np.isfinite(lengths).all() or np.any(lengths <= 0):
        raise ValueError('Zero vector has no cosine direction')
    return matrix/lengths


class ExactCosineIndex:
    def __init__(self, ids, vectors, *, dimensions):
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate document vectors')
        self.vectors = normalize_vectors(vectors)
        if self.vectors.shape != (len(ids), dimensions):
            raise ValueError('Document IDs, vectors or dimensions disagree')
        self.ids = np.asarray(ids)
        self.positions = {key: i for i, key in enumerate(ids)}
        self.dimensions = dimensions

    def search(self, vector, *, eligible_ids, top_k):
        if not set(eligible_ids) <= set(self.positions) or top_k < 1:
            raise ValueError('Unknown eligible document or invalid depth')
        if not eligible_ids:
            return []
        query = normalize_vectors([vector])
        if query.shape[1] != self.dimensions:
            raise ValueError('Query dimensions disagree with document vectors')
        indices = np.asarray([self.positions[key] for key in sorted(eligible_ids)])
        # Filter first. Out-of-scope or expired records are never scored and
        # cannot occupy a candidate slot. C einsum avoids external BLAS DLLs.
        scores = np.clip(np.einsum('nd,qd->nq', self.vectors[indices], query, optimize=False)[:, 0], -1, 1)
        selected = np.lexsort((self.ids[indices], -scores))[:top_k]
        return [(str(self.ids[indices[i]]), float(scores[i])) for i in selected]
