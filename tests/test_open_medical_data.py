from datetime import date
from pathlib import Path

from careguide.corpus import Corpus
from careguide.external_data import (QIC_LABELS, load_huatuo_open_slice,
                                     load_kuake_qic_development,
                                     load_kuake_qic_evaluation,
                                     load_kuake_qtr_development)
from careguide.retrieval.knowledge_base import MedicalKnowledgeBase


ROOT = Path(__file__).resolve().parents[1]


def test_open_corpus_is_general_and_integrity_checked():
    corpus = Corpus(ROOT / "data" / "open_medical_v1")
    coverage = corpus.coverage(date(2026, 9, 21))
    assert coverage["version"] == "careguide-open-medical-v1"
    assert coverage["record_count"] == 1015
    assert coverage["hospitals"] == []
    assert sum(record.representation == "open_dataset_qa" for record in corpus.records) == 1000


def test_open_corpus_retrieves_real_upstream_question():
    corpus = Corpus(ROOT / "data" / "open_medical_v1")
    knowledge = MedicalKnowledgeBase(corpus)
    source = next(record for record in corpus.records if record.representation == "open_dataset_qa")
    hits = knowledge.search(source.title, list(corpus.records), today=date(2026, 9, 21), top_k=3)
    assert hits and hits[0]["record_id"] == source.id
    assert hits[0]["citation"]["license"] == "Apache-2.0"
    assert hits[0]["citation"]["review_status"] == "open_dataset_unreviewed"


def test_curated_source_wins_near_tie_over_open_qa_noise():
    corpus = Corpus(ROOT / "data" / "open_medical_v1")
    knowledge = MedicalKnowledgeBase(corpus)
    hits = knowledge.search("头痛一直不改善时，有来源的一般提示是什么？",
                            list(corpus.records), today=date(2026, 9, 21), top_k=1)
    assert hits[0]["record_id"] == "headache-diary"


def test_external_cblue_splits_keep_upstream_labels_and_counts():
    qic_train = load_kuake_qic_development()
    qic = load_kuake_qic_evaluation()
    qtr = load_kuake_qtr_development()
    assert len(qic_train) == 6931 and {item["label"] for item in qic_train} == set(QIC_LABELS)
    assert len(qic) == 1955 and {item["label"] for item in qic} == set(QIC_LABELS)
    assert len(qtr) == 24174 and {item["label"] for item in qtr} == {"0", "1", "2", "3"}
    assert all(item["upstream_split"] == "train" for item in qic_train)
    assert all(item["upstream_split"] == "dev" for item in qic)
    assert all(item["upstream_split"] == "train" for item in qtr)


def test_huatuo_slice_is_fixed_and_stratified():
    values = load_huatuo_open_slice()
    assert len(values) == 1000
    assert all(item["score"] == 5 for item in values)
    assert len({item["department"] for item in values}) == 15
