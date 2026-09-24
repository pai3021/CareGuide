"""Evaluate real Qwen embeddings for KUAKE-QIC before spending LLM calls."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from careguide.evaluation.v3 import classification_report
from careguide.external_data import (QIC_LABELS, load_kuake_qic_development,
                                     load_kuake_qic_evaluation)
from careguide.qic_classifier import QICPrototypeClassifier, QICRuleClassifier
from scripts.index_kuake_ir_dense import embedding_provider
from scripts.private_config import load_config


ROOT = Path(__file__).resolve().parents[1]


def fingerprint(rows: tuple[dict, ...]) -> str:
    payload = [[row["query"], row["label"]] for row in rows]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def predictions_from_scores(scores: np.ndarray) -> list[str]:
    return [QIC_LABELS[index] for index in np.argmax(scores, axis=1)]


def row_standardize(scores: np.ndarray) -> np.ndarray:
    """Put heterogeneous classifier scores on a comparable per-query scale."""
    means = scores.mean(axis=1, keepdims=True)
    standard_deviations = scores.std(axis=1, keepdims=True)
    return (scores - means) / np.maximum(standard_deviations, 1e-6)


def cosine_matrix(left: np.ndarray, right: np.ndarray, *, batch_size: int = 4) -> np.ndarray:
    """Compute normalized-vector scores without relying on Windows BLAS DLLs."""
    result = np.empty((len(left), len(right)), dtype=np.float32)
    for start in range(0, len(left), batch_size):
        batch = left[start:start + batch_size]
        result[start:start + len(batch)] = np.sum(
            batch[:, None, :] * right[None, :, :], axis=2, dtype=np.float32)
    return result


def confusion(expected: list[str], predicted: list[str]) -> dict[str, dict[str, int]]:
    matrix = {label: {target: 0 for target in QIC_LABELS} for label in QIC_LABELS}
    for gold, guess in zip(expected, predicted):
        matrix[gold][guess] += 1
    return matrix


def evaluate(expected: list[str], predicted: list[str]) -> dict:
    report = classification_report(expected, predicted, tuple(QIC_LABELS))
    report["confusion_matrix"] = confusion(expected, predicted)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--dimensions", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cache", type=Path,
                        default=ROOT / "data/source_cache/qic/qwen3_embeddings_v1.npz")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/eval/results/cblue_qic_qwen_embedding_v1.json")
    parser.add_argument("--centroid-output", type=Path,
                        default=ROOT / "data/external/cblue/qic_qwen3_centroids_v1.json")
    args = parser.parse_args()
    load_config(args.config_file)
    train = load_kuake_qic_development(); dev = load_kuake_qic_evaluation()
    provider = embedding_provider(args.dimensions, args.batch_size)
    identity = {
        "train_fingerprint": fingerprint(train), "dev_fingerprint": fingerprint(dev),
        "model": provider.model, "dimensions": provider.dimensions,
    }

    train_vectors = dev_vectors = None
    if args.cache.exists():
        cached = np.load(args.cache, allow_pickle=False)
        metadata = json.loads(str(cached["metadata"].item()))
        if metadata == identity:
            train_vectors = cached["train_vectors"]
            dev_vectors = cached["dev_vectors"]
    if train_vectors is None:
        print(json.dumps({"status": "embedding", "train": len(train), "dev": len(dev),
                          "model": provider.model, "dimensions": provider.dimensions},
                         ensure_ascii=False), flush=True)
        vectors = provider.embed_documents(
            [row["query"] for row in train] + [row["query"] for row in dev])
        matrix = np.asarray(vectors, dtype=np.float32)
        train_vectors, dev_vectors = matrix[:len(train)], matrix[len(train):]
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.cache, metadata=json.dumps(identity, sort_keys=True),
                            train_vectors=train_vectors, dev_vectors=dev_vectors)

    train_vectors = normalize(train_vectors); dev_vectors = normalize(dev_vectors)
    label_index = {label: index for index, label in enumerate(QIC_LABELS)}
    train_labels = np.asarray([label_index[row["label"]] for row in train])
    expected = [row["label"] for row in dev]

    centroids = np.vstack([normalize(train_vectors[train_labels == index].mean(axis=0, keepdims=True))[0]
                           for index in range(len(QIC_LABELS))])
    centroid_artifact = {
        "version": "qic-qwen3-all-train-centroids-v1",
        "source": "CBLUE KUAKE-QIC official train",
        "train_fingerprint": fingerprint(train),
        "embedding_model": provider.model,
        "embedding_dimensions": provider.dimensions,
        "labels": {label: [round(float(value), 9) for value in centroids[index]]
                   for index, label in enumerate(QIC_LABELS)},
    }
    args.centroid_output.parent.mkdir(parents=True, exist_ok=True)
    args.centroid_output.write_text(
        json.dumps(centroid_artifact, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8")
    centroid_scores = cosine_matrix(dev_vectors, centroids, batch_size=128)
    local_classifier = QICPrototypeClassifier(train)
    local_scores = np.asarray([
        [local_classifier.score(row["query"])[label] for label in QIC_LABELS]
        for row in dev
    ], dtype=np.float32)
    candidates: dict[str, list[str]] = {
        "local_char_tfidf_centroid": predictions_from_scores(local_scores),
        "qwen_all_train_centroid": predictions_from_scores(centroid_scores),
    }

    max_k = 15
    neighbor_indices = np.empty((len(dev), max_k), dtype=np.int32)
    neighbor_scores = np.empty((len(dev), max_k), dtype=np.float32)
    for start in range(0, len(dev), 4):
        similarities = cosine_matrix(dev_vectors[start:start + 4], train_vectors)
        indices = np.argpartition(similarities, -max_k, axis=1)[:, -max_k:]
        values = np.take_along_axis(similarities, indices, axis=1)
        order = np.argsort(-values, axis=1)
        neighbor_indices[start:start + len(indices)] = np.take_along_axis(indices, order, axis=1)
        neighbor_scores[start:start + len(indices)] = np.take_along_axis(values, order, axis=1)
    for k in (1, 3, 5, 9, 15):
        scores = np.zeros((len(dev), len(QIC_LABELS)), dtype=np.float32)
        for row in range(len(dev)):
            for rank in range(k):
                label = train_labels[neighbor_indices[row, rank]]
                scores[row, label] += max(float(neighbor_scores[row, rank]), 0.0)
        candidates[f"qwen_knn_{k}"] = predictions_from_scores(scores)

    rule = QICRuleClassifier()
    rule_labels = [rule.predict(row["query"])[0] for row in dev]
    candidates["definition_rules"] = rule_labels
    for bonus in (0.01, 0.02, 0.03, 0.05, 0.08, 0.12):
        scores = centroid_scores.copy()
        for row, label in enumerate(rule_labels):
            scores[row, label_index[label]] += bonus
        candidates[f"centroid_plus_rule_{bonus:.2f}"] = predictions_from_scores(scores)

    local_scaled = row_standardize(local_scores)
    qwen_scaled = row_standardize(centroid_scores)
    for qwen_weight in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        scores = (1.0 - qwen_weight) * local_scaled + qwen_weight * qwen_scaled
        candidates[f"local_qwen_{qwen_weight:.1f}"] = predictions_from_scores(scores)
        for rule_bonus in (0.05, 0.1, 0.15, 0.2):
            fused = scores.copy()
            for row, label in enumerate(rule_labels):
                fused[row, label_index[label]] += rule_bonus
            candidates[f"local_qwen_{qwen_weight:.1f}_rule_{rule_bonus:.2f}"] = \
                predictions_from_scores(fused)

    reports = {name: evaluate(expected, predicted) for name, predicted in candidates.items()}
    best = max(reports, key=lambda name: (reports[name]["macro_f1"], reports[name]["accuracy"]))
    result = {
        "evaluation": "KUAKE-QIC official dev real embedding evaluation",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "train_count": len(train), "dev_count": len(dev),
        "embedding_model": provider.model, "embedding_dimensions": provider.dimensions,
        "cache": str(args.cache.relative_to(ROOT)),
        "data_usage": "official train fits classifiers; official dev selects the development configuration",
        "best_method": best, "methods": reports,
        "llm_calls": 0, "clinical_validity_evaluated": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {name: {"accuracy": report["accuracy"], "macro_f1": report["macro_f1"]}
               for name, report in reports.items()}
    print(json.dumps({"best_method": best, "methods": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
