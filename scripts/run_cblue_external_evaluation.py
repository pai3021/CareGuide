"""Run the frozen definition-only baseline on untouched KUAKE-QIC dev."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from careguide.external_data import QIC_LABELS, load_kuake_qic_evaluation
from careguide.qic_classifier import QICRuleClassifier


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "data/eval/results/cblue_qic_external_v1.json"))
    args = parser.parse_args()
    classifier = QICRuleClassifier()
    cases = load_kuake_qic_evaluation()
    pairs = [(item["label"], classifier.predict(item["query"])[0]) for item in cases]
    accuracy = sum(gold == predicted for gold, predicted in pairs) / len(pairs)
    f1 = []
    for label in QIC_LABELS:
        tp = sum(g == label and p == label for g, p in pairs)
        fp = sum(g != label and p == label for g, p in pairs)
        fn = sum(g == label and p != label for g, p in pairs)
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1.append(2 * precision * recall / (precision + recall) if precision + recall else 0)
    result = {
        "evaluation": "KUAKE-QIC official dev external evaluation",
        "classifier": classifier.version,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "case_count": len(cases),
        "accuracy": accuracy,
        "macro_f1": sum(f1) / len(f1),
        "gold_distribution": Counter(gold for gold, _ in pairs),
        "prediction_distribution": Counter(predicted for _, predicted in pairs),
        "data_usage": "evaluation_only; not used by prompts, rules, or tuning",
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
