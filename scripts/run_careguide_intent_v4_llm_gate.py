"""Evaluate a resumable MiMo gate on CareGuide-13 validation only."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from careguide.evaluation.v3 import classification_report
from careguide.external_data import CAREGUIDE_INTENT_LABELS, load_kuake_qic_development
from careguide.intent_v4_dataset import load_v4_training_cases, load_v4_validation_cases
from careguide.llm_utils import create_llm_client, track_token_usage
from careguide.qic_classifier import CareGuide13PrototypeClassifier
from careguide.qic_intent_recognizer import QICIntentRecognizer
from scripts.private_config import load_config
from scripts.run_qic_llm_evaluation import load_cache
from scripts.run_qic_remote_embedding_evaluation import cosine_matrix, normalize, row_standardize


ROOT = Path(__file__).resolve().parents[1]


def development_scores(qic_cache: Path, custom_cache: Path):
    train = load_v4_training_cases()
    validation = load_v4_validation_cases()
    qic = load_kuake_qic_development()
    qic_index = {row["id"]: index for index, row in enumerate(qic)}
    qic_vectors = np.load(qic_cache, allow_pickle=False)["train_vectors"]
    custom = np.load(custom_cache, allow_pickle=False)
    custom_train = [case for case in train if case.source_dataset != "KUAKE-QIC"]
    custom_validation = [case for case in validation if case.source_dataset != "KUAKE-QIC"]
    train_custom = dict(zip((case.id for case in custom_train), custom["train_vectors"]))
    validation_custom = dict(zip((case.id for case in custom_validation),
                                 custom["validation_vectors"]))
    train_vectors = normalize(np.asarray([
        qic_vectors[qic_index[case.source_id]] if case.source_dataset == "KUAKE-QIC"
        else train_custom[case.id] for case in train
    ], dtype=np.float32))
    validation_vectors = normalize(np.asarray([
        qic_vectors[qic_index[case.source_id]] if case.source_dataset == "KUAKE-QIC"
        else validation_custom[case.id] for case in validation
    ], dtype=np.float32))
    label_index = {label: index for index, label in enumerate(CAREGUIDE_INTENT_LABELS)}
    train_labels = np.asarray([label_index[case.label] for case in train])
    centroids = np.vstack([
        normalize(train_vectors[train_labels == index].mean(axis=0, keepdims=True))[0]
        for index in range(len(CAREGUIDE_INTENT_LABELS))
    ])
    qwen = cosine_matrix(validation_vectors, centroids, batch_size=128)
    local = CareGuide13PrototypeClassifier(train)
    local_scores = np.asarray([
        [local.score(case.message)[label] for label in CAREGUIDE_INTENT_LABELS]
        for case in validation
    ], dtype=np.float32)
    return train, validation, 0.3 * row_standardize(local_scores) + 0.7 * row_standardize(qwen)


def select_diagnostic(indices: list[int], validation, per_label: int) -> list[int]:
    grouped = defaultdict(list)
    for index in indices:
        grouped[validation[index].label].append(index)
    selected = []
    for label in CAREGUIDE_INTENT_LABELS:
        # Lowest margins are supplied first by the caller. Keep the diagnostic
        # balanced enough to expose failures in small labels.
        selected.extend(grouped[label][:per_label])
    return selected


async def run(args: argparse.Namespace) -> dict:
    load_config(args.config_file)
    key = os.environ.get("CAREGUIDE_LLM_API_KEY", "")
    if not key or key.startswith("YOUR_"):
        raise ValueError("CAREGUIDE_LLM_API_KEY is not configured")
    model = os.environ.get("CAREGUIDE_LLM_MODEL", "gpt-4o-mini")
    _, validation, scores = development_scores(args.qic_cache, args.custom_cache)
    ranked = np.sort(scores, axis=1)
    margins = ranked[:, -1] - ranked[:, -2]
    base_indices = np.argmax(scores, axis=1)
    base_predictions = [CAREGUIDE_INTENT_LABELS[index] for index in base_indices]
    gated = sorted((index for index, margin in enumerate(margins)
                    if margin < args.margin), key=lambda index: (margins[index], validation[index].id))
    selected = gated if args.full_gate else select_diagnostic(
        gated, validation, args.diagnostic_per_label)

    cache = load_cache(args.cache)
    client = create_llm_client(key, os.environ.get("CAREGUIDE_LLM_BASE_URL") or None)
    recognizer = QICIntentRecognizer(client, model, embedding_mode="disabled")
    semaphore = asyncio.Semaphore(args.concurrency)

    async def predict(index: int) -> dict:
        case = validation[index]
        if case.id in cache and not cache[case.id].get("failed"):
            return cache[case.id]
        async with semaphore:
            prediction = await recognizer._llm_careguide(case.message)
        return {"id": case.id, "query": case.message, "gold": case.label,
                "base": base_predictions[index], "margin": round(float(margins[index]), 8),
                "predicted": prediction.get("label"),
                "confidence": prediction.get("confidence", 0.0),
                "reasoning": prediction.get("reasoning", ""),
                "failed": bool(prediction.get("failed")),
                "failure_type": prediction.get("failure_type")}

    pending = [index for index in selected
               if validation[index].id not in cache or cache[validation[index].id].get("failed")]
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    with track_token_usage() as usage:
        tasks = [asyncio.create_task(predict(index)) for index in pending]
        try:
            for task in asyncio.as_completed(tasks):
                value = await task
                cache[value["id"]] = value
                with args.cache.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        finally:
            await client.close()

    expected_selected = [validation[index].label for index in selected]
    base_selected = [base_predictions[index] for index in selected]
    llm_selected = [cache.get(validation[index].id, {}).get("predicted") for index in selected]
    valid_positions = [position for position, prediction in enumerate(llm_selected)
                       if prediction in CAREGUIDE_INTENT_LABELS]
    llm_report = classification_report(
        [expected_selected[position] for position in valid_positions],
        [llm_selected[position] for position in valid_positions],
        tuple(CAREGUIDE_INTENT_LABELS))
    base_sample_report = classification_report(expected_selected, base_selected,
                                               tuple(CAREGUIDE_INTENT_LABELS))
    threshold_reports = {}
    for confidence in (0.0, 0.7, 0.8, 0.9, 0.95):
        predictions = list(base_predictions)
        overrides = 0
        for index in selected:
            row = cache.get(validation[index].id, {})
            if (row.get("predicted") in CAREGUIDE_INTENT_LABELS and
                    float(row.get("confidence", 0.0)) >= confidence):
                predictions[index] = row["predicted"]
                overrides += 1
        report = classification_report([case.label for case in validation], predictions,
                                       tuple(CAREGUIDE_INTENT_LABELS))
        threshold_reports[f"confidence_{confidence:.2f}"] = {
            "accuracy": report["accuracy"], "macro_f1": report["macro_f1"],
            "overrides": overrides, "labels": report["labels"],
        }
    production = threshold_reports["confidence_0.95"]
    result = {
        "evaluation": "CareGuide-13 MiMo gate validation development diagnostic",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "careguide-intent-v4-13class-v1", "split": "validation",
        "test_split_accessed": False, "model": model,
        "profile": "full_gate" if args.full_gate else "balanced_gate_diagnostic",
        "gate_margin": args.margin, "gate_population": len(gated),
        "selected_count": len(selected), "new_llm_calls": usage.calls,
        "token_usage_new_calls": usage.to_dict(),
        "base_full": classification_report([case.label for case in validation],
                                           base_predictions, tuple(CAREGUIDE_INTENT_LABELS)),
        "base_selected": base_sample_report, "llm_selected": llm_report,
        "valid_llm_predictions": len(valid_positions),
        "threshold_reports": threshold_reports,
        "production_candidate": {
            "policy": "margin_lt_0.40_and_confidence_gte_0.95",
            **production,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--qic-cache", type=Path, default=
                        ROOT / "data/source_cache/qic/qwen3_embeddings_v1.npz")
    parser.add_argument("--custom-cache", type=Path, default=
                        ROOT / "data/source_cache/qic/careguide13_custom_qwen3_v1.npz")
    parser.add_argument("--cache", type=Path, default=
                        ROOT / "data/source_cache/qic/careguide13_mimo_predictions_v1.jsonl")
    parser.add_argument("--output", type=Path, default=
                        ROOT / "data/eval/results/careguide13_mimo_gate_diagnostic_v1.json")
    parser.add_argument("--margin", type=float, default=0.4)
    parser.add_argument("--diagnostic-per-label", type=int, default=5)
    parser.add_argument("--full-gate", action="store_true")
    parser.add_argument("--concurrency", type=int, choices=range(1, 9), default=4)
    args = parser.parse_args()
    result = asyncio.run(run(args))
    print(json.dumps({key: result[key] for key in (
        "model", "profile", "gate_population", "selected_count", "new_llm_calls",
        "valid_llm_predictions", "base_selected", "llm_selected", "threshold_reports",
        "production_candidate")},
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
