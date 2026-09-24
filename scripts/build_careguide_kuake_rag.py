"""Build the runtime-aligned CareGuide KUAKE RAG corpus and frozen query splits.

Only upstream KUAKE-IR train query-document pairs are used.  Official dev and
test files are deliberately excluded so the project has one evaluation contract
whose corpus is identical to the runtime corpus.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from scripts.import_kuake_ir import DEFAULT_SOURCE, EXPECTED, REVISION, read_pairs, sha256


ROOT = Path(__file__).resolve().parents[1]
CORPUS_TARGET = ROOT / "data" / "careguide_kuake_rag_v1"
EVAL_TARGET = ROOT / "data" / "eval" / "careguide_kuake_rag_v1"
SEED = "careguide-kuake-rag-v1-20260922"
CORPUS_SIZE = 10_000
SPLIT_COUNTS = {"train": 8_000, "dev": 1_000, "test": 1_000}
REJECT = re.compile(
    r"(?:https?://|www\.|微信|加微|QQ|qq群|咨询热线|联系电话|电话号码|点击.{0,8}咨询)",
    re.IGNORECASE,
)


def digest_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def stable_key(query_id: str, doc_id: str) -> str:
    return hashlib.sha256(f"{SEED}\0{doc_id}\0{query_id}".encode()).hexdigest()


def eligible(query: str, passage: str) -> bool:
    query, passage = query.strip(), passage.strip()
    return (4 <= len(query) <= 80 and 40 <= len(passage) <= 512
            and not REJECT.search(query) and not REJECT.search(passage))


def validate_source(source: Path) -> None:
    for name in ("corpus.tsv", "KUAKE-IR_train_query.txt", "KUAKE-IR_train.tsv"):
        expected_count, expected_hash = EXPECTED[name]
        path = source / name
        if not path.exists() or sha256(path) != expected_hash:
            raise ValueError(f"KUAKE-IR source integrity mismatch: {name}")
        with path.open("rb") as handle:
            if sum(1 for _ in handle) != expected_count:
                raise ValueError(f"KUAKE-IR source row count mismatch: {name}")


def build(source: Path = DEFAULT_SOURCE) -> dict:
    source = Path(source).resolve()
    validate_source(source)
    queries = dict(read_pairs(source / "KUAKE-IR_train_query.txt"))
    relations = dict(read_pairs(source / "KUAKE-IR_train.tsv"))
    wanted_docs = set(relations.values())
    passages = {doc_id: text for doc_id, text in read_pairs(source / "corpus.tsv")
                if doc_id in wanted_docs}

    # Keep one deterministic representative query for every document.  This
    # prevents the same gold document from appearing in multiple query splits.
    by_document: dict[str, tuple[str, str]] = {}
    for query_id, query in queries.items():
        doc_id = relations[query_id]
        passage = passages.get(doc_id, "")
        if not eligible(query, passage):
            continue
        candidate = (query_id, query.strip())
        current = by_document.get(doc_id)
        if current is None or stable_key(query_id, doc_id) < stable_key(current[0], doc_id):
            by_document[doc_id] = candidate

    selected = sorted(
        ((stable_key(query_id, doc_id), doc_id, query_id, query)
         for doc_id, (query_id, query) in by_document.items()),
        key=lambda item: item[0],
    )[:CORPUS_SIZE]
    if len(selected) != CORPUS_SIZE:
        raise ValueError(f"Only {len(selected)} eligible unique documents; need {CORPUS_SIZE}")

    base = ROOT / "data" / "medical_v1"
    base_records = json.loads((base / "records.json").read_text(encoding="utf-8"))
    curated = [item for item in base_records if item["kind"] in {
        "care_guidance", "drug_label", "drug_interaction"}]
    source_ids = {item["source_id"] for item in curated}
    sources = [item for item in json.loads((base / "sources.json").read_text(encoding="utf-8"))
               if item["id"] in source_ids]
    sources.append({
        "id": "cblue-kuake-ir-train-slice",
        "title": "CBLUE KUAKE-IR train 固定知识切片",
        "url": "https://tianchi.aliyun.com/dataset/95414",
        "publisher": "CBLUE / Alibaba KUAKE",
        "jurisdiction": "CN",
        "scope": "KUAKE-IR开放研究数据固定切片；未经CareGuide临床审核；仅作一般健康知识检索",
        "published_at": "2022-08-01",
        "verified_at": "2026-09-22",
        "valid_until": "2035-12-31",
        "review_status": "open_dataset_unreviewed",
        "clinical_reviewed": False,
        "verification_method": "dataset_revision_and_checksum",
        "locator": f"KUAKE-IR train; transport revision {REVISION}; seed {SEED}",
        "retrieved_sha256": EXPECTED["corpus.tsv"][1],
        "evidence_anchors": ["KUAKE-IR train", "fixed seed", "source SHA-256"],
        "license": "Apache-2.0",
        "dataset_revision": REVISION,
    })

    records = list(curated)
    cases: list[dict] = []
    boundaries = (SPLIT_COUNTS["train"], SPLIT_COUNTS["train"] + SPLIT_COUNTS["dev"])
    for index, (_, doc_id, query_id, query) in enumerate(selected):
        split = "train" if index < boundaries[0] else "dev" if index < boundaries[1] else "test"
        record_id = f"kuake-ir-{doc_id}"
        passage = passages[doc_id].strip()
        records.append({
            "id": record_id,
            "source_id": "cblue-kuake-ir-train-slice",
            "kind": "care_guidance",
            # Never place the labelled query in the indexed document.  Doing
            # so would make exact-query evaluation a trivial leakage test.
            "title": f"KUAKE-IR 开放医学段落 {doc_id}",
            "content": passage,
            "keywords": ["KUAKE-IR"],
            "representation": "open_dataset_qa",
            "locator": f"KUAKE-IR train query_id={query_id}; doc_id={doc_id}",
        })
        cases.append({
            "id": f"careguide-kuake-{query_id}",
            "split": split,
            "query": query,
            "gold_record_id": record_id,
            "upstream_query_id": query_id,
            "upstream_doc_id": doc_id,
        })

    CORPUS_TARGET.mkdir(parents=True, exist_ok=True)
    source_raw = (json.dumps(sources, ensure_ascii=False, indent=2) + "\n").encode()
    record_raw = (json.dumps(records, ensure_ascii=False, indent=2) + "\n").encode()
    (CORPUS_TARGET / "sources.json").write_bytes(source_raw)
    (CORPUS_TARGET / "records.json").write_bytes(record_raw)
    corpus_manifest = {
        "version": "careguide-kuake-rag-v1",
        "source_count": len(sources),
        "record_count": len(records),
        "files": {"sources.json": digest_bytes(source_raw), "records.json": digest_bytes(record_raw)},
        "composition": {"kuake_ir_train_records": CORPUS_SIZE,
                        "retained_sourced_guidance_and_drug_records": len(curated)},
        "selection": {"seed": SEED, "unique_gold_document": True,
                      "official_dev_used": False, "official_test_used": False},
    }
    (CORPUS_TARGET / "manifest.json").write_text(
        json.dumps(corpus_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (CORPUS_TARGET / "README.md").write_text(
        "# CareGuide KUAKE RAG V1\n\n"
        "默认运行时语料由KUAKE-IR train按唯一gold文档、固定过滤规则和固定seed抽取的10,000条"
        "开放记录，以及15条既有公众健康/药品来源摘要组成。KUAKE记录未经临床审核，只用于一般"
        "健康知识检索；用药标签继续使用监管来源。官方dev/test没有参与抽取、调参或主结果。\n",
        encoding="utf-8",
    )

    EVAL_TARGET.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, count in SPLIT_COUNTS.items():
        values = [case for case in cases if case["split"] == split]
        if len(values) != count:
            raise ValueError(f"Unexpected {split} count: {len(values)}")
        raw = "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values).encode()
        name = f"{split}.jsonl"
        (EVAL_TARGET / name).write_bytes(raw)
        files[name] = {"count": len(values), "sha256": digest_bytes(raw)}
    eval_manifest = {
        "version": "careguide-kuake-rag-eval-v1",
        "source_dataset": "CBLUE KUAKE-IR train",
        "runtime_corpus_version": corpus_manifest["version"],
        "runtime_corpus_manifest_sha256": sha256(CORPUS_TARGET / "manifest.json"),
        "seed": SEED,
        "split_unit": "unique_gold_document",
        "official_dev_used": False,
        "official_test_used": False,
        "heldout_policy": "test is frozen and must not be used for tuning before version freeze",
        "files": files,
    }
    (EVAL_TARGET / "manifest.json").write_text(
        json.dumps(eval_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (EVAL_TARGET / "README.md").write_text(
        "# CareGuide KUAKE RAG V1 评测集\n\n"
        "全部查询和gold段落来自KUAKE-IR train；按唯一gold文档固定拆分为8,000 train、1,000 dev"
        "和1,000冻结test。三组查询检索同一个10,000篇运行时语料库，官方dev/test不进入主评测。"
        "冻结test在方案定版前不得用于阈值、Top-K、模型或reranker选择。\n",
        encoding="utf-8",
    )
    return {"corpus": corpus_manifest, "evaluation": eval_manifest}


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
