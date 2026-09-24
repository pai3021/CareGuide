"""Evaluate the frozen QIC semantic-intent contract on the official dev split."""
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from careguide.evaluation.v3 import classification_report
from careguide.external_data import QIC_LABELS, load_kuake_qic_evaluation
from careguide.qic_intent_recognizer import QICIntentRecognizer


ROOT = Path(__file__).resolve().parents[1]


async def evaluate(recognizer: QICIntentRecognizer, cases: tuple[dict, ...]) -> list[str]:
    predictions = []
    for item in cases:
        result = await recognizer.recognize(item["query"])
        predictions.append(result.qic_intent or "其他")
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(
        ROOT / "data/eval/results/cblue_qic_runtime_local_v2.json"))
    args = parser.parse_args()
    cases = load_kuake_qic_evaluation()
    recognizer = QICIntentRecognizer(embedding_mode="local_ngram")
    predicted = asyncio.run(evaluate(recognizer, cases))
    report = classification_report(
        [item["label"] for item in cases], predicted, tuple(QIC_LABELS))
    result = {
        "evaluation": "KUAKE-QIC official dev development evaluation",
        "recognizer": recognizer.taxonomy,
        "branches": {
            "llm": "unavailable_not_scored",
            "embedding": recognizer.prototype_classifier.version,
            "rule": recognizer.rule_classifier.version,
        },
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "data_usage": (
            "official train builds prototypes; official dev has already been used for development "
            "diagnostics and is not claimed as untouched held-out"
        ),
        "clinical_validity_evaluated": False,
        **report,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
