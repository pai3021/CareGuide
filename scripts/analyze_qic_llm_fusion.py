"""Measure whether the QIC few-shot branch helps on uncertain fused predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from careguide.evaluation.v3 import classification_report
from careguide.external_data import (QIC_LABELS, load_kuake_qic_development,
                                     load_kuake_qic_evaluation)
from careguide.qic_classifier import QICPrototypeClassifier, QICRuleClassifier
from scripts.run_qic_remote_embedding_evaluation import (cosine_matrix, normalize,
                                                          predictions_from_scores,
                                                          row_standardize)


ROOT = Path(__file__).resolve().parents[1]


def row_id(row: dict) -> str:
    return hashlib.sha256(f'{row["label"]}\0{row["query"]}'.encode("utf-8")).hexdigest()[:16]


def metrics(expected: list[str], predicted: list[str]) -> dict:
    result = classification_report(expected, predicted, tuple(QIC_LABELS))
    return {"accuracy": result["accuracy"], "macro_f1": result["macro_f1"]}


def build_non_llm_scores(train: tuple[dict, ...], dev: tuple[dict, ...],
                         embedding_cache: Path) -> np.ndarray:
    cached = np.load(embedding_cache, allow_pickle=False)
    train_vectors = normalize(cached["train_vectors"])
    dev_vectors = normalize(cached["dev_vectors"])
    label_index = {label: index for index, label in enumerate(QIC_LABELS)}
    train_labels = np.asarray([label_index[row["label"]] for row in train])
    centroids = np.vstack([
        normalize(train_vectors[train_labels == index].mean(axis=0, keepdims=True))[0]
        for index in range(len(QIC_LABELS))
    ])
    qwen_scores = row_standardize(cosine_matrix(dev_vectors, centroids, batch_size=128))
    local = QICPrototypeClassifier(train)
    local_scores = row_standardize(np.asarray([
        [local.score(row["query"])[label] for label in QIC_LABELS] for row in dev
    ], dtype=np.float32))
    fused_scores = 0.4 * local_scores + 0.6 * qwen_scores
    rules = QICRuleClassifier()
    for index, row in enumerate(dev):
        rule_label, _ = rules.predict(row["query"])
        fused_scores[index, label_index[rule_label]] += 0.2
    return fused_scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-cache", type=Path,
                        default=ROOT / "data/source_cache/qic/qwen3_embeddings_v1.npz")
    parser.add_argument("--llm-report", type=Path,
                        default=ROOT / "data/eval/results/cblue_qic_llm_sample_v1.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/eval/results/cblue_qic_llm_gating_analysis_v1.json")
    args = parser.parse_args()

    train = load_kuake_qic_development()
    dev = load_kuake_qic_evaluation()
    fused_scores = build_non_llm_scores(train, dev, args.embedding_cache)

    llm_report = json.loads(args.llm_report.read_text(encoding="utf-8"))
    llm_rows = {row["id"]: row for row in llm_report["rows"] if not row.get("failed")}
    selected_indices = [index for index, row in enumerate(dev) if row_id(row) in llm_rows]
    expected = [dev[index]["label"] for index in selected_indices]
    base_predictions = predictions_from_scores(fused_scores[selected_indices])
    llm_predictions = [llm_rows[row_id(dev[index])]["predicted"] for index in selected_indices]
    llm_confidences = np.asarray([
        float(llm_rows[row_id(dev[index])].get("confidence", 0.0)) for index in selected_indices])
    ranked = np.sort(fused_scores[selected_indices], axis=1)
    margins = ranked[:, -1] - ranked[:, -2]

    candidates = {
        "non_llm_fusion": {**metrics(expected, base_predictions), "llm_calls": 0},
        "llm_only": {**metrics(expected, llm_predictions), "llm_calls": len(expected)},
    }
    for threshold in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00):
        for minimum_llm_confidence in (0.0, 0.70, 0.80, 0.90):
            use_llm = (margins < threshold) & (llm_confidences >= minimum_llm_confidence)
            predicted = [llm if choose else base for llm, base, choose in
                         zip(llm_predictions, base_predictions, use_llm)]
            name = f"margin_lt_{threshold:.2f}_llm_conf_{minimum_llm_confidence:.2f}"
            candidates[name] = {**metrics(expected, predicted),
                                "llm_calls": int(use_llm.sum()),
                                "llm_call_rate": round(float(use_llm.mean()), 4)}
    best = max(candidates, key=lambda name: (
        candidates[name]["macro_f1"], candidates[name]["accuracy"],
        -candidates[name]["llm_calls"]))
    result = {
        "evaluation": "KUAKE-QIC LLM uncertainty-gating development analysis",
        "sample_count": len(expected),
        "sample_scope": llm_report["profile"],
        "llm_model": llm_report["model"],
        "non_llm_configuration": {
            "local_weight": 0.4, "qwen_weight": 0.6, "rule_bonus": 0.2},
        "best_method": best, "methods": candidates,
        "data_usage": "official dev development diagnostic; thresholds are not held-out claims",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best_method": best, "best": candidates[best],
                      "non_llm_fusion": candidates["non_llm_fusion"],
                      "llm_only": candidates["llm_only"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
