"""Inverted dot-product index for real BGE-M3 lexical-weight dictionaries.

This module does not generate weights and does not approximate BGE with BM25.
The caller must supply dictionaries from a verified local model/cache.
"""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping


class SparseLexicalIndex:
    def __init__(self, documents: Mapping[str, Mapping[str, float]]):
        self.postings: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for identity, vector in documents.items():
            self._validate(vector)
            for token, weight in vector.items():
                if weight:
                    self.postings[token].append((identity, float(weight)))

    @staticmethod
    def _validate(vector):
        if any(not isinstance(token, str) or not math.isfinite(weight) or weight < 0
               for token, weight in vector.items()):
            raise ValueError('Sparse weights require string token IDs and finite nonnegative weights')

    def search(self, query: Mapping[str, float], *, top_k: int = 100,
               eligible_ids: set[str] | None = None) -> list[tuple[str, float]]:
        self._validate(query)
        if top_k < 1:
            raise ValueError('Sparse candidate depth must be positive')
        scores: dict[str, float] = defaultdict(float)
        for token, weight in query.items():
            for identity, document_weight in self.postings.get(token, ()):
                if eligible_ids is None or identity in eligible_ids:
                    scores[identity] += float(weight)*document_weight
        return sorted(((identity, score) for identity, score in scores.items() if score > 0),
                      key=lambda row: (-row[1], row[0]))[:top_k]
