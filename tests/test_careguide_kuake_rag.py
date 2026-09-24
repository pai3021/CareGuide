import hashlib
import json
from pathlib import Path

from careguide.corpus import Corpus
from scripts.run_careguide_kuake_rag_evaluation import load_cases


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "careguide_kuake_rag_v1"
EVAL = ROOT / "data" / "eval" / "careguide_kuake_rag_v1"


def rows(name):
    return [json.loads(line) for line in (EVAL / name).read_text(encoding="utf-8").splitlines()]


def test_kuake_runtime_corpus_is_integrity_checked_and_unreviewed():
    corpus = Corpus(CORPUS)
    assert corpus.version == "careguide-kuake-rag-v1"
    assert len(corpus.records) == 10_015
    assert sum(record.representation == "open_dataset_qa" for record in corpus.records) == 10_000
    evidence = corpus.evidence(next(record for record in corpus.records
                                    if record.representation == "open_dataset_qa"))
    assert evidence["citation"]["review_status"] == "open_dataset_unreviewed"
    assert evidence["citation"]["publisher"] == "CBLUE / Alibaba KUAKE"
    open_records = [record for record in corpus.records
                    if record.representation == "open_dataset_qa"]
    assert all(record.title == f"KUAKE-IR 开放医学段落 {record.id.removeprefix('kuake-ir-')}"
               for record in open_records)
    assert all(record.keywords == ("KUAKE-IR",) for record in open_records)


def test_kuake_query_splits_are_frozen_and_document_disjoint():
    manifest = json.loads((EVAL / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["official_dev_used"] is False
    assert manifest["official_test_used"] is False
    splits = {name: rows(f"{name}.jsonl") for name in ("train", "dev", "test")}
    assert {name: len(values) for name, values in splits.items()} == {
        "train": 8000, "dev": 1000, "test": 1000}
    doc_sets = {name: {item["upstream_doc_id"] for item in values}
                for name, values in splits.items()}
    assert not (doc_sets["train"] & doc_sets["dev"])
    assert not (doc_sets["train"] & doc_sets["test"])
    assert not (doc_sets["dev"] & doc_sets["test"])
    for name, values in splits.items():
        raw = (EVAL / f"{name}.jsonl").read_bytes()
        assert hashlib.sha256(raw).hexdigest() == manifest["files"][f"{name}.jsonl"]["sha256"]
        assert all(item["split"] == name for item in values)


def test_evaluator_loads_dev_without_official_dev():
    cases, manifest = load_cases("dev")
    assert len(cases) == 1000
    assert manifest["official_dev_used"] is False
