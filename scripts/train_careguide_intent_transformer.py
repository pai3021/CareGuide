"""Fine-tune a BERT-family encoder on CareGuide-13 train/validation only."""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path


LABELS = (
    "疾病表述", "指标解读", "医疗费用", "治疗方案", "功效作用", "病情诊断",
    "其他", "注意事项", "病因分析", "就医建议", "后果表述",
    "appointment_action", "emergency",
)


def read_jsonl(path: Path, expected_split: str) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows or any(row.get("split") != expected_split for row in rows):
        raise ValueError(f"Invalid or mixed split: {path}")
    if {row.get("label") for row in rows} != set(LABELS):
        raise ValueError(f"Missing CareGuide-13 labels: {path}")
    return rows


def class_weights(rows: list[dict], power: float = 1.0) -> list[float]:
    counts = Counter(row["label"] for row in rows)
    total = len(rows)
    return [(total / (len(LABELS) * counts[label])) ** power for label in LABELS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="hfl/chinese-roberta-wwm-ext")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--class-weight-power", type=float, default=1.0,
                        help="Exponent applied to inverse-frequency weights; 0 disables weighting.")
    args = parser.parse_args()

    random.seed(args.seed)
    train_rows = read_jsonl(args.data_dir / "train.jsonl", "train")
    validation_rows = read_jsonl(args.data_dir / "validation.jsonl", "validation")
    # The frozen test file is deliberately never opened by this program.

    import numpy as np
    import torch
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
    from torch.nn import CrossEntropyLoss
    from torch.utils.data import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments,
                              set_seed)

    set_seed(args.seed)
    label_to_id = {label: index for index, label in enumerate(LABELS)}
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    class IntentDataset(Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
            encoded = tokenizer(row["message"], truncation=True,
                                max_length=args.max_length)
            encoded["labels"] = label_to_id[row["label"]]
            return encoded

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(LABELS),
        id2label={index: label for label, index in label_to_id.items()},
        label2id=label_to_id,
    )
    weights = torch.tensor(class_weights(train_rows, args.class_weight_power),
                           dtype=torch.float32)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False,
                         num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = CrossEntropyLoss(weight=weights.to(outputs.logits.device))(
                outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(prediction):
        expected = prediction.label_ids
        predicted = np.argmax(prediction.predictions, axis=1)
        precision, recall, per_f1, support = precision_recall_fscore_support(
            expected, predicted, labels=list(range(len(LABELS))), zero_division=0)
        metrics = {
            "accuracy": accuracy_score(expected, predicted),
            "macro_f1": f1_score(expected, predicted, average="macro", zero_division=0),
        }
        for index, label in enumerate(LABELS):
            metrics[f"f1_{index:02d}"] = per_f1[index]
            metrics[f"recall_{index:02d}"] = recall[index]
        return metrics

    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=IntentDataset(train_rows),
        eval_dataset=IntentDataset(validation_rows),
        data_collator=DataCollatorWithPadding(tokenizer),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    train_result = trainer.train()
    validation_metrics = trainer.evaluate()
    trainer.save_model(str(args.output_dir / "best_model"))
    tokenizer.save_pretrained(str(args.output_dir / "best_model"))
    summary = {
        "model": args.model,
        "dataset": "careguide-intent-v4-13class-v1",
        "train_count": len(train_rows),
        "validation_count": len(validation_rows),
        "test_split_accessed": False,
        "labels": list(LABELS),
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "class_weight_power": args.class_weight_power,
        "train_runtime_seconds": train_result.metrics.get("train_runtime"),
        "validation": validation_metrics,
        "cuda_device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "max_cuda_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
