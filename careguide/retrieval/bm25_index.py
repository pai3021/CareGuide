"""BM25 index for versioned CareGuide source records."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List


def tokenize_knowledge_text(text: str) -> List[str]:
    """Tokenize identifiers, Latin words, and Chinese unigrams/bigrams."""
    normalized = str(text).casefold()
    words = re.findall(r"[a-z0-9][a-z0-9._:/-]*", normalized)
    identifiers = [
        token for token in words
        if any(char.isdigit() for char in token) or "-" in token or "_" in token
    ]
    # Never create Chinese bigrams across punctuation, numbers or paragraph breaks.
    cjk_terms = [term for run in re.findall(r"[\u3400-\u9fff]+", normalized)
                 for term in ([run] if len(run) == 1 else
                              [run[i:i+2] for i in range(len(run)-1)])]
    return [
        *words,
        *identifiers,
        *cjk_terms,
    ]


@dataclass(frozen=True)
class BM25Hit:
    chunk_id: str
    score: float
    content: str
    metadata: Dict[str, Any]
    matched_term_count: int
    identifier_match: bool


class BM25Index:
    """In-process BM25 index used behind scope and source freshness filters."""

    def __init__(
        self,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        identifier_boost: float = 1.0,
    ):
        if k1 <= 0:
            raise ValueError("BM25 k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("BM25 b must be between 0 and 1")
        if identifier_boost < 1:
            raise ValueError("BM25 identifier_boost must be at least 1")
        self.k1 = float(k1)
        self.b = float(b)
        self.identifier_boost = float(identifier_boost)
        self._rows: Dict[str, Dict[str, Any]] = {}
        self._term_frequencies: Dict[str, Counter[str]] = {}
        self._document_frequencies: Counter[str] = Counter()
        self._average_length = 0.0

    def clear(self) -> None:
        self._rows.clear()
        self._term_frequencies.clear()
        self._document_frequencies.clear()
        self._average_length = 0.0

    def upsert(self, rows: Iterable[Dict[str, Any]]) -> None:
        for row in rows:
            chunk_id = str(row["chunk_id"])
            self._rows[chunk_id] = {
                "chunk_id": chunk_id,
                "content": str(row["content"]),
                "retrieval_text": str(row["retrieval_text"]),
                "metadata": dict(row["metadata"]),
            }
        self._recalculate_statistics()

    def _recalculate_statistics(self) -> None:
        self._term_frequencies = {}
        self._document_frequencies = Counter()
        total_length = 0
        for chunk_id, row in self._rows.items():
            frequencies = Counter(tokenize_knowledge_text(row["retrieval_text"]))
            self._term_frequencies[chunk_id] = frequencies
            self._document_frequencies.update(frequencies.keys())
            total_length += sum(frequencies.values())
        self._average_length = total_length / len(self._rows) if self._rows else 0.0

    def search(self, query: str, *, top_k: int = 20,
               eligible_ids: set[str] | None = None) -> List[BM25Hit]:
        query_terms = set(tokenize_knowledge_text(query))
        if not query_terms or not self._rows:
            return []
        document_count = len(self._rows)
        average_length = self._average_length or 1.0
        scored: List[BM25Hit] = []
        for chunk_id, frequencies in self._term_frequencies.items():
            if eligible_ids is not None and chunk_id not in eligible_ids:
                continue
            document_length = sum(frequencies.values())
            score = 0.0
            matched_terms = query_terms & set(frequencies)
            identifier_match = any(
                any(char.isdigit() for char in term) or "-" in term or "_" in term
                for term in matched_terms
            )
            required_matches = 1 if len(query_terms) == 1 else 2
            if len(matched_terms) < required_matches and not identifier_match:
                continue
            for term in query_terms:
                term_frequency = frequencies.get(term, 0)
                if not term_frequency:
                    continue
                document_frequency = self._document_frequencies[term]
                inverse_document_frequency = math.log(
                    1.0 + (document_count - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                term_weight = (
                    self.identifier_boost
                    if any(char.isdigit() for char in term) or "-" in term or "_" in term
                    else 1.0
                )
                denominator = term_frequency + self.k1 * (
                    1.0 - self.b + self.b * document_length / average_length
                )
                score += term_weight * inverse_document_frequency * (
                    term_frequency * (self.k1 + 1.0) / denominator
                )
            if score > 0:
                row = self._rows[chunk_id]
                scored.append(BM25Hit(
                    chunk_id=chunk_id,
                    score=score,
                    content=row["content"],
                    metadata=dict(row["metadata"]),
                    matched_term_count=len(matched_terms),
                    identifier_match=identifier_match,
                ))
        scored.sort(key=lambda hit: (-hit.score, hit.chunk_id))
        return scored[:max(1, int(top_k))]

    @property
    def count(self) -> int:
        return len(self._rows)
