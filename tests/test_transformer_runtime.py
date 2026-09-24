import pytest

from careguide import runtime
from careguide.external_data import CAREGUIDE_INTENT_LABELS
import careguide.transformer_intent_classifier as transformer_module


class RuntimeFakeTransformer:
    version = "runtime-fake-transformer"

    def __init__(self, model_path, *, device="auto"):
        self.model_path = model_path
        self.device = device

    def predict(self, message):
        scores = {label: 0.001 for label in CAREGUIDE_INTENT_LABELS}
        scores["其他"] = 0.988
        return {"label": "其他", "confidence": 0.988, "margin": 0.987,
                "scores": scores, "model": self.version, "device": self.device}


def configure_transformer(monkeypatch, model_path):
    monkeypatch.setenv("CAREGUIDE_INTENT_TAXONOMY", "qic")
    monkeypatch.setenv("CAREGUIDE_INTENT_EMBEDDING_MODE", "local_ngram")
    monkeypatch.setenv("CAREGUIDE_QIC_CLASSIFIER_BACKEND", "transformer")
    monkeypatch.setenv("CAREGUIDE_QIC_MODEL_PATH", model_path)
    monkeypatch.setenv("CAREGUIDE_QIC_MODEL_DEVICE", "cpu")


def test_runtime_transformer_backend_requires_explicit_model_path(monkeypatch):
    configure_transformer(monkeypatch, "")
    runtime.get_chat_runtime.cache_clear()
    with pytest.raises(ValueError, match="CAREGUIDE_QIC_MODEL_PATH"):
        runtime.get_chat_runtime()
    runtime.get_chat_runtime.cache_clear()


def test_runtime_wires_explicit_transformer_without_fallback(monkeypatch):
    configure_transformer(monkeypatch, "D:/models/careguide-intent/best_model")
    monkeypatch.setattr(transformer_module, "TransformerIntentClassifier",
                        RuntimeFakeTransformer)
    runtime.get_chat_runtime.cache_clear()
    try:
        engine = runtime.get_chat_runtime()[0]
        classifier = engine.recognizer.semantic_classifier
        assert isinstance(classifier, RuntimeFakeTransformer)
        assert classifier.device == "cpu"
    finally:
        runtime.get_chat_runtime.cache_clear()
