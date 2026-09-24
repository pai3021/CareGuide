"""Measure the existing lexical scorer on KUAKE-QTR development pairs."""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from careguide.external_data import load_kuake_qtr_development
from careguide.retrieval.bm25_index import tokenize_knowledge_text


ROOT = Path(__file__).resolve().parents[1]


def overlap(query: str, title: str) -> float:
    left, right = set(tokenize_knowledge_text(query)), set(tokenize_knowledge_text(title))
    return len(left & right) / math.sqrt(len(left) * len(right)) if left and right else 0.0


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + 1 + end) / 2
        for index in order[cursor:end]:
            ranks[index] = rank
        cursor = end
    return ranks


def correlation(left: list[float], right: list[float]) -> float:
    a, b = average_ranks(left), average_ranks(right)
    mean_a, mean_b = sum(a) / len(a), sum(b) / len(b)
    numerator = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    denominator = math.sqrt(sum((x - mean_a) ** 2 for x in a) *
                            sum((y - mean_b) ** 2 for y in b))
    return numerator / denominator if denominator else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "data/eval/results/cblue_qtr_lexical_v1.json"))
    args = parser.parse_args()
    cases = load_kuake_qtr_development()
    labels = [float(item["label"]) for item in cases]
    scores = [overlap(item["query"], item["title"]) for item in cases]
    result = {
        "evaluation": "KUAKE-QTR official train development diagnostic",
        "scorer": "careguide-cjk-unigram-bigram-cosine-overlap-v1",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "case_count": len(cases),
        "spearman": correlation(scores, labels),
        "mean_score_by_label": {str(label): sum(score for score, gold in zip(scores, labels) if gold == label) /
                                sum(gold == label for gold in labels) for label in range(4)},
        "data_usage": "development diagnostic; no official-dev/test claim",
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
