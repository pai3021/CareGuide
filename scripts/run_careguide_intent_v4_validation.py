"""Evaluate the train-only CareGuide-13 local baseline on validation.

This script deliberately has no option to load the frozen test split.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict

from careguide.external_data import CAREGUIDE_INTENT_LABELS
from careguide.intent_v4_dataset import load_v4_training_cases, load_v4_validation_cases
from careguide.qic_classifier import CareGuide13HybridClassifier


def safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate() -> dict:
    classifier = CareGuide13HybridClassifier(load_v4_training_cases())
    validation = load_v4_validation_cases()
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    correct = 0
    for case in validation:
        predicted, _ = classifier.predict(case.message)
        confusion[case.label][predicted] += 1
        correct += predicted == case.label

    per_label = {}
    f1_values = []
    for label in CAREGUIDE_INTENT_LABELS:
        true_positive = confusion[label][label]
        support = sum(confusion[label].values())
        predicted_count = sum(row[label] for row in confusion.values())
        recall = safe_divide(true_positive, support)
        precision = safe_divide(true_positive, predicted_count)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        f1_values.append(f1)
        per_label[label] = {
            "support": support,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "top_confusions": confusion[label].most_common(4),
        }
    return {
        "dataset": "careguide-intent-v4-13class-v1",
        "split": "validation",
        "test_split_accessed": False,
        "classifier": classifier.version,
        "count": len(validation),
        "accuracy": round(correct / len(validation), 6),
        "macro_f1": round(sum(f1_values) / len(f1_values), 6),
        "per_label": per_label,
    }


def main() -> None:
    print(json.dumps(evaluate(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
