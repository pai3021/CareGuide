"""One-shot evaluation of a frozen CareGuide-13 transformer checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from careguide.external_data import CAREGUIDE_INTENT_LABELS
from careguide.transformer_intent_classifier import TransformerIntentClassifier


def load_frozen_test(data_dir: Path) -> list[dict]:
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    expected = manifest["files"]["test.jsonl"]
    raw = (data_dir / "test.jsonl").read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected["sha256"]:
        raise ValueError("Frozen test integrity mismatch")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(rows) != expected["count"] or any(row.get("split") != "test" for row in rows):
        raise ValueError("Frozen test count or split marker mismatch")
    if Counter(row["label"] for row in rows) != Counter(expected["labels"]):
        raise ValueError("Frozen test label counts mismatch")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--confirm-frozen-test", action="store_true")
    args = parser.parse_args()
    if not args.confirm_frozen_test:
        raise SystemExit("Refusing to access frozen test without --confirm-frozen-test")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen result: {args.output}")

    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    rows = load_frozen_test(args.data_dir)
    classifier = TransformerIntentClassifier(args.model_path, device=args.device)
    predictions = []
    started = time.monotonic()
    for offset in range(0, len(rows), args.batch_size):
        batch = [row["message"] for row in rows[offset:offset + args.batch_size]]
        predictions.extend(item["label"] for item in classifier.predict_many(batch))
    expected = [row["label"] for row in rows]
    labels = list(CAREGUIDE_INTENT_LABELS)
    matrix = confusion_matrix(expected, predictions, labels=labels)
    per_label = {
        label: {
            "f1": f1_score(expected, predictions, labels=[label], average="macro",
                           zero_division=0),
            "support": sum(value == label for value in expected),
        }
        for label in labels
    }
    result = {
        "evaluation": "CareGuide-13 frozen test",
        "dataset_version": "careguide-intent-v4-13class-v1",
        "model": classifier.version,
        "model_path": str(args.model_path),
        "split": "test",
        "count": len(rows),
        "accuracy": accuracy_score(expected, predictions),
        "macro_f1": f1_score(expected, predictions, average="macro", zero_division=0),
        "per_label": per_label,
        "labels": labels,
        "confusion_matrix": matrix.tolist(),
        "runtime_seconds": time.monotonic() - started,
        "device": classifier.device,
        "frozen_test_access_count_for_release": 1,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
