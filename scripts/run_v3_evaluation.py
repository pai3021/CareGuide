"""Run V3 development evaluation only; held-out is intentionally unavailable."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from careguide.corpus import Corpus
from careguide.evaluation.v3 import classification_report, retrieval_report
from careguide.intent_v3_dataset import (FINE_LABELS, load_v3_boundary_development_cases,
                                         load_v3_development_cases)
from careguide.intent_v3_recognizer import V3IntentRecognizer
from careguide.retrieval.knowledge_base import MedicalKnowledgeBase

ROOT = Path(__file__).resolve().parents[1]


async def run(limit: int | None):
    cases = list(load_v3_development_cases())[:limit]
    recognizer = V3IntentRecognizer(embedding_mode="local_ngram")
    predictions = [await recognizer.recognize(case.message) for case in cases]
    intent = classification_report([case.label for case in cases],
                                   [item.fine_intent for item in predictions], FINE_LABELS)
    boundary_cases = list(load_v3_boundary_development_cases())[:limit]
    boundary_predictions = [await recognizer.recognize(case.message) for case in boundary_cases]
    boundary_rows = [{"id": case.id, "expected": case.label, "predicted": result.disposition,
                      "correct": case.label == result.disposition}
                     for case, result in zip(boundary_cases, boundary_predictions)]
    urgent = [row for row in boundary_rows if row["expected"] == "urgent_handoff"]
    boundary = {"total": len(boundary_rows),
                "accuracy": round(sum(row["correct"] for row in boundary_rows)/len(boundary_rows), 4) if boundary_rows else None,
                "emergency_miss_rate": round(sum(not row["correct"] for row in urgent)/len(urgent), 4) if urgent else None,
                "rows": boundary_rows}
    rag_cases = [json.loads(line) for line in (ROOT/"data/eval/careguide_rag_v3/dev.jsonl").read_text(encoding="utf-8").splitlines()]
    corpus = Corpus(ROOT/"data/medical_zhongda_v2")
    rag = retrieval_report(MedicalKnowledgeBase(corpus), rag_cases[:limit])
    return {"version":"careguide-v3-development", "split":"dev", "heldout_used":False,
            "intent":intent, "boundary":boundary, "retrieval":rag, "clinical_validity_evaluated":False,
            "judge":{"status":"unscored", "reason":"not_requested"}}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(); report = asyncio.run(run(args.limit))
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(text+"\n", encoding="utf-8")
    print(text)


if __name__ == "__main__": main()
