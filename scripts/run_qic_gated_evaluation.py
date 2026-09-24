"""Run the dev-selected QIC LLM gate over the complete official dev split."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from careguide.evaluation.v3 import classification_report
from careguide.external_data import (QIC_LABELS, load_kuake_qic_development,
                                     load_kuake_qic_evaluation)
from careguide.llm_utils import create_llm_client, track_token_usage
from careguide.qic_intent_recognizer import QICIntentRecognizer
from scripts.analyze_qic_llm_fusion import build_non_llm_scores, row_id
from scripts.private_config import load_config
from scripts.run_qic_llm_evaluation import load_cache
from scripts.run_qic_remote_embedding_evaluation import predictions_from_scores


ROOT = Path(__file__).resolve().parents[1]


async def run(args: argparse.Namespace) -> dict:
    load_config(args.config_file)
    key = os.environ.get("CAREGUIDE_LLM_API_KEY", "")
    if not key or key.startswith("YOUR_"):
        raise ValueError("CAREGUIDE_LLM_API_KEY is not configured")
    model = os.environ.get("CAREGUIDE_LLM_MODEL", "gpt-4o-mini")
    train = load_kuake_qic_development()
    dev = load_kuake_qic_evaluation()
    scores = build_non_llm_scores(train, dev, args.embedding_cache)
    base_predictions = predictions_from_scores(scores)
    ranked = np.sort(scores, axis=1)
    margins = ranked[:, -1] - ranked[:, -2]
    gated_indices = [index for index, margin in enumerate(margins)
                     if margin < args.margin]
    cache = load_cache(args.llm_cache)
    cache_hits = sum(row_id(dev[index]) in cache and
                     not cache[row_id(dev[index])].get("failed") for index in gated_indices)
    client = create_llm_client(key, os.environ.get("CAREGUIDE_LLM_BASE_URL") or None)
    recognizer = QICIntentRecognizer(client, model, embedding_mode="disabled")
    semaphore = asyncio.Semaphore(args.concurrency)

    async def predict(index: int) -> dict:
        row = dev[index]
        identifier = row_id(row)
        if identifier in cache and not cache[identifier].get("failed"):
            return cache[identifier]
        async with semaphore:
            prediction = await recognizer._llm_qic(row["query"])
        return {"id": identifier, "query": row["query"], "gold": row["label"],
                "predicted": prediction.get("label"),
                "confidence": prediction.get("confidence", 0.0),
                "reasoning": prediction.get("reasoning", ""),
                "failed": bool(prediction.get("failed")),
                "failure_type": prediction.get("failure_type")}

    pending = [index for index in gated_indices
               if row_id(dev[index]) not in cache or cache[row_id(dev[index])].get("failed")]
    args.llm_cache.parent.mkdir(parents=True, exist_ok=True)
    with track_token_usage() as usage:
        tasks = [asyncio.create_task(predict(index)) for index in pending]
        try:
            for task in asyncio.as_completed(tasks):
                value = await task
                cache[value["id"]] = value
                with args.llm_cache.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        finally:
            await client.close()

    predictions = list(base_predictions)
    overrides = 0
    failures = 0
    for index in gated_indices:
        row = cache.get(row_id(dev[index]), {})
        if row.get("failed") or row.get("predicted") not in QIC_LABELS:
            failures += 1
        elif float(row.get("confidence", 0.0)) >= args.llm_confidence:
            predictions[index] = row["predicted"]
            overrides += 1
    expected = [row["label"] for row in dev]
    base_report = classification_report(expected, base_predictions, tuple(QIC_LABELS))
    gated_report = classification_report(expected, predictions, tuple(QIC_LABELS))
    gate_analysis = {}
    for margin_threshold in (0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40):
        for confidence_threshold in (0.90, 0.95, 0.98, 0.99):
            candidate = list(base_predictions)
            candidate_calls = candidate_overrides = 0
            for index, margin in enumerate(margins):
                if margin >= margin_threshold:
                    continue
                candidate_calls += 1
                row = cache.get(row_id(dev[index]), {})
                if (not row.get("failed") and row.get("predicted") in QIC_LABELS and
                        float(row.get("confidence", 0.0)) >= confidence_threshold):
                    candidate[index] = row["predicted"]
                    candidate_overrides += 1
            candidate_report = classification_report(expected, candidate, tuple(QIC_LABELS))
            name = f"margin_lt_{margin_threshold:.2f}_confidence_{confidence_threshold:.2f}"
            gate_analysis[name] = {
                "accuracy": candidate_report["accuracy"],
                "macro_f1": candidate_report["macro_f1"],
                "llm_calls": candidate_calls, "llm_overrides": candidate_overrides,
            }
    best_gate = max(gate_analysis, key=lambda name: (
        gate_analysis[name]["macro_f1"], gate_analysis[name]["accuracy"],
        -gate_analysis[name]["llm_calls"]))
    result = {
        "evaluation": "KUAKE-QIC complete official dev gated fusion",
        "evaluated_at": datetime.now(timezone.utc).isoformat(), "model": model,
        "dev_count": len(dev), "gate_margin": args.margin,
        "llm_override_confidence": args.llm_confidence,
        "gated_count": len(gated_indices), "gated_rate": round(len(gated_indices) / len(dev), 4),
        "cache_hits": cache_hits, "new_llm_calls": usage.calls,
        "llm_failures": failures, "llm_overrides": overrides,
        "token_usage_new_calls": usage.to_dict(),
        "non_llm": {"accuracy": base_report["accuracy"], "macro_f1": base_report["macro_f1"]},
        "gated_fusion": gated_report,
        "best_gate": best_gate, "gate_analysis": gate_analysis,
        "data_usage": "official dev selected and evaluated the gate; not an untouched held-out claim",
        "clinical_validity_evaluated": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path,
                        default=ROOT / "data/source_cache/qic/qwen3_embeddings_v1.npz")
    parser.add_argument("--llm-cache", type=Path,
                        default=ROOT / "data/source_cache/qic/llm_fewshot_predictions_v1.jsonl")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/eval/results/cblue_qic_gated_full_dev_v1.json")
    parser.add_argument("--margin", type=float, default=0.4)
    parser.add_argument("--llm-confidence", type=float, default=0.9)
    parser.add_argument("--concurrency", type=int, choices=range(1, 9), default=4)
    args = parser.parse_args()
    result = asyncio.run(run(args))
    print(json.dumps({key: result[key] for key in (
        "model", "dev_count", "gated_count", "gated_rate", "cache_hits",
        "new_llm_calls", "llm_failures", "llm_overrides", "non_llm")},
        ensure_ascii=False, indent=2))
    print(json.dumps({"gated_fusion": {
        "accuracy": result["gated_fusion"]["accuracy"],
        "macro_f1": result["gated_fusion"]["macro_f1"]},
        "best_gate": result["best_gate"],
        "best_gate_metrics": result["gate_analysis"][result["best_gate"]]},
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
