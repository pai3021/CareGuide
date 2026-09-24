"""Fit CareGuide-13 Qwen centroids on train and evaluate validation only."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from careguide.evaluation.v3 import classification_report
from careguide.external_data import CAREGUIDE_INTENT_LABELS, load_kuake_qic_development
from careguide.intent_v4_dataset import load_v4_training_cases, load_v4_validation_cases
from careguide.qic_classifier import CareGuide13HybridClassifier, CareGuide13PrototypeClassifier
from scripts.index_kuake_ir_dense import embedding_provider
from scripts.private_config import load_config
from scripts.run_qic_remote_embedding_evaluation import cosine_matrix, normalize, row_standardize


ROOT = Path(__file__).resolve().parents[1]


def fingerprint(cases) -> str:
    payload = [[case.id, case.message, case.label] for case in cases]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def report(expected: list[str], predicted: list[str]) -> dict:
    return classification_report(expected, predicted, tuple(CAREGUIDE_INTENT_LABELS))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--dimensions", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--qic-cache", type=Path, default=
                        ROOT / "data/source_cache/qic/qwen3_embeddings_v1.npz")
    parser.add_argument("--custom-cache", type=Path, default=
                        ROOT / "data/source_cache/qic/careguide13_custom_qwen3_v1.npz")
    parser.add_argument("--centroid-output", type=Path, default=
                        ROOT / "data/external/cblue/careguide13_qwen3_centroids_v1.json")
    parser.add_argument("--output", type=Path, default=
                        ROOT / "data/eval/results/careguide13_qwen_validation_v1.json")
    args = parser.parse_args()
    load_config(args.config_file)

    train = load_v4_training_cases()
    validation = load_v4_validation_cases()
    provider = embedding_provider(args.dimensions, args.batch_size)

    qic = load_kuake_qic_development()
    qic_index = {row["id"]: index for index, row in enumerate(qic)}
    cached_qic = np.load(args.qic_cache, allow_pickle=False)
    qic_vectors = np.asarray(cached_qic["train_vectors"], dtype=np.float32)
    if qic_vectors.shape != (len(qic), args.dimensions):
        raise ValueError("Pinned QIC embedding cache shape mismatch")

    custom_train = [case for case in train if case.source_dataset != "KUAKE-QIC"]
    custom_validation = [case for case in validation if case.source_dataset != "KUAKE-QIC"]
    identity = {
        "train_fingerprint": fingerprint(custom_train),
        "validation_fingerprint": fingerprint(custom_validation),
        "model": provider.model,
        "dimensions": provider.dimensions,
    }
    custom_train_vectors = custom_validation_vectors = None
    if args.custom_cache.is_file():
        cached = np.load(args.custom_cache, allow_pickle=False)
        if json.loads(str(cached["metadata"].item())) == identity:
            custom_train_vectors = cached["train_vectors"]
            custom_validation_vectors = cached["validation_vectors"]
    if custom_train_vectors is None:
        texts = [case.message for case in custom_train + custom_validation]
        print(json.dumps({"status": "embedding_custom_development_only",
                          "train": len(custom_train), "validation": len(custom_validation),
                          "test": 0, "model": provider.model}, ensure_ascii=False), flush=True)
        vectors = np.asarray(provider.embed_documents(texts), dtype=np.float32)
        custom_train_vectors = vectors[:len(custom_train)]
        custom_validation_vectors = vectors[len(custom_train):]
        args.custom_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.custom_cache, metadata=json.dumps(identity, sort_keys=True),
                            train_vectors=custom_train_vectors,
                            validation_vectors=custom_validation_vectors)

    custom_train_by_id = dict(zip((case.id for case in custom_train), custom_train_vectors))
    custom_validation_by_id = dict(zip((case.id for case in custom_validation),
                                       custom_validation_vectors))
    train_vectors = np.asarray([
        qic_vectors[qic_index[case.source_id]] if case.source_dataset == "KUAKE-QIC"
        else custom_train_by_id[case.id]
        for case in train
    ], dtype=np.float32)
    validation_vectors = np.asarray([
        qic_vectors[qic_index[case.source_id]] if case.source_dataset == "KUAKE-QIC"
        else custom_validation_by_id[case.id]
        for case in validation
    ], dtype=np.float32)
    train_vectors = normalize(train_vectors)
    validation_vectors = normalize(validation_vectors)

    label_index = {label: index for index, label in enumerate(CAREGUIDE_INTENT_LABELS)}
    train_labels = np.asarray([label_index[case.label] for case in train])
    centroids = np.vstack([
        normalize(train_vectors[train_labels == index].mean(axis=0, keepdims=True))[0]
        for index in range(len(CAREGUIDE_INTENT_LABELS))
    ])
    artifact = {
        "version": "careguide-13-qwen3-train-centroids-v1",
        "source_dataset_version": "careguide-intent-v4-13class-v1",
        "train_fingerprint": fingerprint(train),
        "embedding_model": provider.model,
        "embedding_dimensions": provider.dimensions,
        "labels": {label: [round(float(value), 9) for value in centroids[index]]
                   for index, label in enumerate(CAREGUIDE_INTENT_LABELS)},
    }
    args.centroid_output.parent.mkdir(parents=True, exist_ok=True)
    args.centroid_output.write_text(
        json.dumps(artifact, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8")

    expected = [case.label for case in validation]
    qwen_scores = cosine_matrix(validation_vectors, centroids, batch_size=128)
    local = CareGuide13PrototypeClassifier(train)
    hybrid = CareGuide13HybridClassifier(train)
    local_scores = np.asarray([
        [local.score(case.message)[label] for label in CAREGUIDE_INTENT_LABELS]
        for case in validation
    ], dtype=np.float32)
    rule_predictions = [hybrid.predict(case.message)[0] for case in validation]
    local_predictions = [local.predict(case.message)[0] for case in validation]
    candidates = {
        "local_char_tfidf": local_predictions,
        "local_rules": rule_predictions,
        "qwen_centroid": [CAREGUIDE_INTENT_LABELS[index]
                           for index in np.argmax(qwen_scores, axis=1)],
    }
    local_scaled = row_standardize(local_scores)
    qwen_scaled = row_standardize(qwen_scores)
    for weight in (0.4, 0.5, 0.6, 0.7, 0.8):
        scores = (1 - weight) * local_scaled + weight * qwen_scaled
        name = f"local_qwen_{weight:.1f}"
        predictions = [CAREGUIDE_INTENT_LABELS[index]
                       for index in np.argmax(scores, axis=1)]
        candidates[name] = predictions
        # Apply only workflow/safety decisions made by the high-precision rules.
        candidates[f"{name}_workflow_rules"] = [
            rule if rule in {"appointment_action", "emergency"} else prediction
            for rule, prediction in zip(rule_predictions, predictions)
        ]
    reports = {name: report(expected, predictions)
               for name, predictions in candidates.items()}
    best = max(reports, key=lambda name: (reports[name]["macro_f1"],
                                          reports[name]["accuracy"]))
    result = {
        "evaluation": "CareGuide-13 train to validation Qwen development evaluation",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "careguide-intent-v4-13class-v1",
        "train_count": len(train), "validation_count": len(validation),
        "test_split_accessed": False,
        "embedding_model": provider.model,
        "embedding_dimensions": provider.dimensions,
        "custom_external_embedding_calls": len(custom_train) + len(custom_validation),
        "best_method": best,
        "methods": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"best_method": best,
                      "methods": {name: {"accuracy": value["accuracy"],
                                           "macro_f1": value["macro_f1"]}
                                  for name, value in reports.items()}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
