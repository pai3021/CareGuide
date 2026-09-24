from careguide.external_data import CAREGUIDE_INTENT_LABELS, QIC_LABELS
from careguide.intent_v4_dataset import load_v4_training_cases
from careguide.qic_intent_recognizer import CAREGUIDE_FEWSHOT_TEXTS
from careguide.qic_intent_recognizer import QICIntentRecognizer
from scripts.run_qic_llm_evaluation import deterministic_stratified_sample


def test_deterministic_stratified_sample_is_balanced_and_stable():
    rows = tuple({"query": f"{label}-{index}", "label": label}
                 for label in QIC_LABELS for index in range(4))
    first = deterministic_stratified_sample(rows, 2)
    second = deterministic_stratified_sample(tuple(reversed(rows)), 2)
    assert first == second
    assert len(first) == len(QIC_LABELS) * 2
    assert all(sum(row["label"] == label for row in first) == 2 for label in QIC_LABELS)


def test_careguide_fewshot_examples_come_only_from_v4_train():
    available = {(case.label, case.message) for case in load_v4_training_cases()}
    assert set(CAREGUIDE_FEWSHOT_TEXTS) == set(CAREGUIDE_INTENT_LABELS)
    assert all((label, text) in available
               for label, texts in CAREGUIDE_FEWSHOT_TEXTS.items() for text in texts)


def test_careguide_llm_override_uses_one_uniform_high_confidence_gate():
    recognizer = QICIntentRecognizer(embedding_mode="local_ngram",
                                     llm_override_confidence=0.95)
    assert recognizer._accept_careguide_llm(
        {"label": "病因分析", "confidence": 0.95}) is True
    assert recognizer._accept_careguide_llm(
        {"label": "emergency", "confidence": 0.949}) is False
    assert recognizer._accept_careguide_llm(
        {"label": "unknown", "confidence": 1.0}) is False
