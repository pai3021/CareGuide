import asyncio

from careguide.external_data import (CAREGUIDE_INTENT_LABELS, QIC_LABELS,
                                     load_kuake_qic_development)
from careguide.qic_classifier import QICPrototypeClassifier
from careguide.qic_intent_recognizer import QICIntentRecognizer


def recognize(message: str):
    return asyncio.run(QICIntentRecognizer(embedding_mode="local_ngram").recognize(message))


class FakeTransformerClassifier:
    version = "fake-transformer"

    def __init__(self, label):
        self.label = label

    def predict(self, message):
        scores = {label: 0.001 for label in CAREGUIDE_INTENT_LABELS}
        scores[self.label] = 0.988
        return {"label": self.label, "confidence": 0.988, "margin": 0.987,
                "scores": scores, "model": self.version, "device": "cpu"}


def test_qic_prototype_uses_only_official_train():
    train = load_kuake_qic_development()
    classifier = QICPrototypeClassifier(train)
    label, confidence = classifier.predict("头痛可能是什么原因")
    assert label in QIC_LABELS
    assert 0 <= confidence <= 1
    assert classifier.version == "qic-train-char-tfidf-centroid-v1"


def test_qic_semantic_intent_is_separate_from_medication_action():
    result = recognize("布洛芬有什么禁忌？")
    assert result.qic_intent in QIC_LABELS
    assert result.agent_route == "medication"
    assert result.business_action == "medication_information"
    assert result.slots["section"] == "contraindications"
    assert result.fine_intent == result.business_action


def test_qic_semantic_intent_is_separate_from_appointment_write():
    result = recognize("帮我预约模拟号源SLOT-GASTRO-001")
    assert result.qic_intent in QIC_LABELS
    assert result.agent_route == "appointment"
    assert result.business_action == "appointment_create"
    assert result.requested_action == "booking_transaction"
    assert result.entities["slot_id"] == "SLOT-GASTRO-001"
    assert result.careguide_intent == "appointment_action"


def test_qic_boundary_preempts_ordinary_agent_route():
    result = recognize("我突然胸痛而且呼吸困难")
    assert result.disposition == "urgent_handoff"
    assert result.boundary == "urgent_handoff"
    assert result.intent.value == "emergency"
    assert result.agent_route == "escalation"
    assert result.careguide_intent == "emergency"


def test_qic_entity_ids_are_not_truncated_before_chinese_text():
    result = recognize("把预约APT-DEMO-001改期到模拟号源SLOT-GASTRO-001")
    assert result.entities["appointment_id"] == "APT-DEMO-001"
    assert result.entities["slot_id"] == "SLOT-GASTRO-001"


def test_qic_rejects_individualized_dose_adjustment_wording():
    result = recognize("帮我调整药物剂量")
    assert result.boundary == "individualized_medication"
    assert result.disposition == "individualized_medication"


def test_careguide_13_training_classifies_implicit_poisoning_as_emergency():
    result = recognize("家人把有害清洗液当饮料喝下去了")
    assert result.careguide_intent == "emergency"
    assert result.agent_route == "escalation"
    assert result.disposition == "urgent_handoff"


def test_qic_compatibility_label_remains_one_of_the_official_eleven():
    result = recognize("我想查看合成医院的消化内科医生列表")
    assert result.careguide_intent == "appointment_action"
    assert result.agent_route == "appointment"
    assert result.qic_intent in QIC_LABELS


def test_transformer_semantic_backend_replaces_fusion_but_keeps_action_routing():
    recognizer = QICIntentRecognizer(
        embedding_mode="disabled",
        semantic_classifier=FakeTransformerClassifier("病因分析"),
    )
    result = asyncio.run(recognizer.recognize("为什么最近总是头痛"))
    branch = result.source_scores["qic_branches"]["careguide_13"]
    assert result.careguide_intent == "病因分析"
    assert result.qic_intent == "病因分析"
    assert result.agent_route == "triage"
    assert branch["backend"] == "transformer"
    assert branch["model"] == "fake-transformer"
    assert result.source_scores["qic_branches"]["embedding"]["skipped"] is True


def test_transformer_special_label_preserves_official_qic_compatibility():
    recognizer = QICIntentRecognizer(
        embedding_mode="disabled",
        semantic_classifier=FakeTransformerClassifier("appointment_action"),
    )
    result = asyncio.run(recognizer.recognize("查看合成医院可预约医生"))
    assert result.careguide_intent == "appointment_action"
    assert result.qic_intent == "其他"
    assert result.agent_route == "appointment"


def test_reference_question_about_red_flags_is_not_active_emergency():
    result = recognize("突然发生而且非常剧烈的头痛，公开资料建议怎么处理？")
    assert result.boundary is None
    assert result.agent_route == "triage"
    assert result.business_action == "health_information"


def test_visit_preparation_reference_is_not_appointment_action():
    result = recognize("请提供反复头痛的记录与就医准备资料")
    assert result.agent_route == "triage"
    assert result.business_action == "health_information"


def test_synthetic_hospital_department_is_bound_for_doctor_search():
    result = recognize("查询合成医院心血管内科有哪些医生")
    assert result.agent_route == "appointment"
    assert result.business_action == "doctor_search"
    assert result.entities["department"] == "心血管内科"
