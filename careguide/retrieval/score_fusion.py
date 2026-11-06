"""Query-local fusion of scored retrieval lists; scores are not probabilities."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def fuse_scores(streams: Mapping[str, Sequence[tuple[str, float]]], *,
                weights: Mapping[str, float], method: str = 'rrf',
                constant: int = 60) -> list[tuple[str, float]]:
    """Fuse a union, without inventing scores for documents absent from a leg.

    Min-Max is applied independently to each query/leg's returned scores.
    Missing scores and constant-score legs contribute zero. A zero-weight leg
    contributes neither scores nor candidates. Equal totals sort by document ID.
    RRF consumes input rank order; Min-Max consumes the original numeric scores.
    """
    if method not in {'rrf', 'minmax'}:
        raise ValueError('Unknown fusion method')
    if set(streams) != set(weights) or not weights:
        raise ValueError('Fusion weights must match the streams')
    if (any(not math.isfinite(w) or w < 0 for w in weights.values())
            or not any(weights.values())):
        raise ValueError('Fusion weights must be finite, nonnegative and not all zero')
    if constant <= 0:
        raise ValueError('RRF constant must be positive')
    totals: dict[str, float] = {}
    for name, stream in streams.items():
        ids = [identity for identity, _ in stream]
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate document within a retrieval stream')
        if any(not math.isfinite(score) for _, score in stream):
            raise ValueError('Non-finite retrieval score')
        weight = weights[name]
        if not stream or not weight:
            continue
        lo, hi = min(score for _, score in stream), max(score for _, score in stream)
        for rank, (identity, score) in enumerate(stream, 1):
            normalized = (score-lo)/(hi-lo) if hi > lo else 0.0
            contribution = 1/(constant+rank) if method == 'rrf' else normalized
            totals[identity] = totals.get(identity, 0.0) + weight*contribution
    return sorted(totals.items(), key=lambda row: (-row[1], row[0]))


def reserve_lexical_candidates(ranking: Sequence[str], lexical: Sequence[str], *,
                               base_k: int = 20, reserve_k: int = 3) -> list[str]:
    """Protect the original pool and append missed lexical leaders for reranking.

    The maximum pool is base_k+reserve_k; no score or answer-confidence gate is
    implied. This changes the reranker input pool, not the final answer order.
    """
    if base_k < 1 or reserve_k < 0:
        raise ValueError('Invalid candidate pool budget')
    selected = list(dict.fromkeys(ranking))[:base_k]
    for identity in lexical[:reserve_k]:
        if identity not in selected:
            selected.append(identity)
    return selected
