"""Local transformer inference backend for the frozen CareGuide-13 taxonomy."""
from __future__ import annotations

import threading
from pathlib import Path

from careguide.external_data import CAREGUIDE_INTENT_LABELS


class TransformerIntentClassifier:
    """Load the fine-tuned checkpoint and return uncalibrated softmax scores."""

    version = "careguide-13-chinese-roberta-wwm-ext-v1"

    def __init__(self, model_path: str | Path, *, device: str = "auto",
                 max_length: int = 128):
        path = Path(model_path).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"CareGuide intent model directory not found: {path}")
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Transformer intent backend requires torch and transformers"
            ) from exc

        resolved_device = ("cuda" if torch.cuda.is_available() else "cpu") \
            if device == "auto" else device
        if resolved_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for intent inference but is unavailable")

        self.path = path
        self.device = resolved_device
        self.max_length = max_length
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            path, local_files_only=True).to(resolved_device)
        self._model.eval()
        configured = tuple(
            self._model.config.id2label[index]
            for index in range(self._model.config.num_labels)
        )
        if configured != tuple(CAREGUIDE_INTENT_LABELS):
            raise ValueError("Transformer checkpoint label order does not match CareGuide-13")
        self._lock = threading.Lock()

    def predict_many(self, messages: list[str]) -> list[dict]:
        if not messages or any(not message.strip() for message in messages):
            raise ValueError("Intent messages must not be empty")
        with self._lock, self._torch.inference_mode():
            lengths = [len(self._tokenizer.encode(message, add_special_tokens=True)) for message in messages]
            batch = self._tokenizer(
                messages, truncation=True, max_length=self.max_length,
                padding=True, return_tensors="pt",
            )
            batch = {key: value.to(self.device) for key, value in batch.items()}
            probabilities = self._torch.softmax(
                self._model(**batch).logits.float(), dim=-1).cpu().tolist()
        results = []
        for row, length in zip(probabilities, lengths):
            scores = {
                label: float(row[index])
                for index, label in enumerate(CAREGUIDE_INTENT_LABELS)
            }
            label = max(scores, key=scores.get)
            ranked = sorted(scores.values(), reverse=True)
            results.append({
                "label": label,
                "confidence": scores[label],
                "margin": ranked[0] - ranked[1],
                "scores": scores,
                "model": self.version,
                "device": self.device,
                "input_tokens": length,
                "truncated": length > self.max_length,
            })
        return results

    def predict(self, message: str) -> dict:
        return self.predict_many([message])[0]
