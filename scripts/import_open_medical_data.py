"""Build reproducible open-data slices used by CareGuide.

The large upstream Huatuo file is kept under the ignored source cache.  Only a
fixed, stratified research slice and normalized CBLUE benchmark files are
written to version control.  The importer never changes the frozen V1/V2 data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "source_cache"
EXTERNAL = ROOT / "data" / "external"
OPEN_CORPUS = ROOT / "data" / "open_medical_v1"
SEED = 20260921
HUATUO_REVISION = "90ce61699e90568db82cdce4c4035c6918d70aa6"
CBLUE_QIC_MIRROR_REVISION = "cd6be11a66b731bfe11f90cbe73a672beeaa905c"
CBLUE_QTR_MIRROR_REVISION = "330c7af04ff80dedd78b3eaa4fc4ba7525bedabd"
CBLUE_QIC_TRAIN_MIRROR_REVISION = "8f879e3624b06a184288a8770113da5fddcb61ff"

DOWNLOADS = {
    "huatuo26m_lite.jsonl": (
        "https://huggingface.co/datasets/FreedomIntelligence/Huatuo26M-Lite/resolve/"
        f"{HUATUO_REVISION}/format_data.jsonl?download=true",
        "84b849f07ab3b9517443f5efeb6c7230dcadf1f71ccaf41f3bf889e53618d4e8",
    ),
    "qic_dev.json": (
        "https://huggingface.co/datasets/wyp/CBlue-KUAKE-QIC/resolve/"
        f"{CBLUE_QIC_MIRROR_REVISION}/KUAKE-QIC_dev.json?download=true",
        "a572e0576afe8510bc9d6b25424f835e6d30e760109ac9fb48c8a77696aa2602",
    ),
    "qic_train.json": (
        "https://raw.githubusercontent.com/Sherlock-coder/CBLUE/"
        f"{CBLUE_QIC_TRAIN_MIRROR_REVISION}/data/KUAKE-QIC/KUAKE-QIC_train.json",
        "090b296e71e6ca5144a193e2fc32b0297ab662cb689539ced1272198f19bc372",
    ),
    "qtr.json": (
        "https://huggingface.co/datasets/dirtycomputer/CBLUE-KUAKE-QTR/resolve/"
        f"{CBLUE_QTR_MIRROR_REVISION}/KUAKE-QTR_train.json?download=true",
        "e6e7a9f0bd9d01a75eb90904c99ceb4bfed4dec1199632494f26e03ad96225fd",
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(name: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / name
    url, expected = DOWNLOADS[name]
    if path.exists() and sha256(path) == expected:
        return path
    error: Exception | None = None
    for attempt in range(6):
        try:
            with httpx.stream("GET", url, follow_redirects=True, timeout=180) as response:
                response.raise_for_status()
                with path.open("wb") as handle:
                    for chunk in response.iter_bytes():
                        handle.write(chunk)
            if sha256(path) != expected:
                raise ValueError(f"checksum mismatch for {name}")
            return path
        except Exception as exc:  # network retry is intentionally bounded
            error = exc
            path.unlink(missing_ok=True)
            time.sleep(2**attempt)
    raise RuntimeError(f"failed to download {name}") from error


def write_jsonl(path: Path, values: list[dict]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = ("\n".join(json.dumps(item, ensure_ascii=False, separators=(",", ":"))
                     for item in values) + "\n").encode()
    path.write_bytes(raw)
    return {"count": len(values), "sha256": hashlib.sha256(raw).hexdigest()}


def normalize_cblue() -> None:
    qic_train_path = download("qic_train.json")
    qic_dev_path, qtr_path = download("qic_dev.json"), download("qtr.json")
    qic_train = json.loads(qic_train_path.read_text(encoding="utf-8"))
    qic_dev = json.loads(qic_dev_path.read_text(encoding="utf-8"))
    qtr = json.loads(qtr_path.read_text(encoding="utf-8"))
    if len(qic_train) != 6931 or len(qic_dev) != 1955 or len(qtr) != 24174:
        raise ValueError("CBLUE mirror counts disagree with the official benchmark")
    qic_train_values = [{"id": item["id"], "query": item["query"], "label": item["label"],
                         "upstream_split": "train"} for item in qic_train]
    qic_dev_values = [{"id": item["id"], "query": item["query"], "label": item["label"],
                       "upstream_split": "dev"} for item in qic_dev]
    qtr_values = [{"id": item["id"], "query": item["query"], "title": item["title"],
                   "label": item["label"], "upstream_split": "train"} for item in qtr]
    target = EXTERNAL / "cblue"
    files = {
        "kuake_qic_train.jsonl": write_jsonl(target / "kuake_qic_train.jsonl", qic_train_values),
        "kuake_qic_dev.jsonl": write_jsonl(target / "kuake_qic_dev.jsonl", qic_dev_values),
        "kuake_qtr_train.jsonl": write_jsonl(target / "kuake_qtr_train.jsonl", qtr_values),
    }
    manifest = {
        "version": "careguide-cblue-snapshot-v2",
        "license": "Apache-2.0",
        "license_file": "../LICENSE-APACHE-2.0.txt",
        "official_source": "https://github.com/CBLUEbenchmark/CBLUE",
        "official_download_page": "https://tianchi.aliyun.com/dataset/95414",
        "transport_mirrors": {
            "kuake_qic_train": {"repository": "Sherlock-coder/CBLUE",
                                  "revision": CBLUE_QIC_TRAIN_MIRROR_REVISION,
                                  "raw_sha256": DOWNLOADS["qic_train.json"][1]},
            "kuake_qic_dev": {"repository": "wyp/CBlue-KUAKE-QIC",
                               "revision": CBLUE_QIC_MIRROR_REVISION,
                               "raw_sha256": DOWNLOADS["qic_dev.json"][1]},
            "kuake_qtr_train": {"repository": "dirtycomputer/CBLUE-KUAKE-QTR",
                                 "revision": CBLUE_QTR_MIRROR_REVISION,
                                 "raw_sha256": DOWNLOADS["qtr.json"][1]},
        },
        "usage": {"kuake_qic_train": "intent_development_only",
                  "kuake_qic_dev": "external_intent_evaluation_only",
                  "kuake_qtr_train": "retrieval_relevance_development"},
        "files": files,
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (target / "README.md").write_text(
        "# CBLUE 外部基准快照\n\n"
        "这里保留 KUAKE-QIC 官方 train/dev 与 KUAKE-QTR 官方 train 的规范化副本。"
        "标签、ID 和上游划分不做改写。QIC train 用于 few-shot、Embedding 原型和规则开发；"
        "QIC dev 只用于冻结后的外部评测；QTR 仅保留历史相关性诊断。官方数据下载页需要交互，因此 manifest 同时记录"
        "只作传输用途的 Hugging Face 镜像提交和原始文件哈希。\n",
        encoding="utf-8",
    )


def huatuo_slice(size: int = 1000) -> list[dict]:
    source = download("huatuo26m_lite.jsonl")
    groups: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("score") != 5 or not item.get("label"):
                continue
            question, answer = str(item.get("question", "")).strip(), str(item.get("answer", "")).strip()
            if not (8 <= len(question) <= 500 and 8 <= len(answer) <= 2000):
                continue
            rank = hashlib.sha256(f"{SEED}:{item['id']}".encode()).hexdigest()
            groups[item["label"]].append((rank, item))
    labels = sorted(label for label in groups if label != "急诊科")
    if not labels:
        raise ValueError("Huatuo source produced no eligible labels")
    for rows in groups.values():
        rows.sort(key=lambda pair: pair[0])
    quota, remainder = divmod(size, len(labels))
    chosen: list[dict] = []
    for index, label in enumerate(labels):
        take = quota + int(index < remainder)
        chosen.extend(item for _, item in groups[label][:take])
    if len(chosen) != size:
        raise ValueError("Huatuo stratified sample is incomplete")
    random.Random(SEED).shuffle(chosen)
    return chosen


def build_open_corpus(size: int = 1000) -> None:
    rows = huatuo_slice(size)
    sample_target = EXTERNAL / "huatuo26m_lite"
    sample = [{"id": item["id"], "question": item["question"], "answer": item["answer"],
               "score": item["score"], "department": item["label"],
               "related_diseases": item.get("related_diseases", "")}
              for item in rows]
    sample_file = write_jsonl(sample_target / "sample.jsonl", sample)
    (sample_target / "manifest.json").write_text(json.dumps({
        "version": "careguide-huatuo26m-lite-slice-v1",
        "dataset": "FreedomIntelligence/Huatuo26M-Lite",
        "revision": HUATUO_REVISION,
        "license": "Apache-2.0",
        "license_file": "../LICENSE-APACHE-2.0.txt",
        "upstream_rows": 177703,
        "upstream_sha256": DOWNLOADS["huatuo26m_lite.jsonl"][1],
        "selection": {"seed": SEED, "score": 5, "stratified_by": "label",
                      "excluded_label": "急诊科"},
        "file": sample_file,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (sample_target / "README.md").write_text(
        "# Huatuo26M-Lite 固定切片\n\n"
        "固定提交、固定 seed、按科室分层抽取的1000条研究数据。内容未经 CareGuide 临床审核，"
        "只能作为开放医疗问答检索演示语料，不能冒充医院、监管机构或医生建议。\n",
        encoding="utf-8",
    )

    base = ROOT / "data" / "medical_v1"
    base_records = json.loads((base / "records.json").read_text(encoding="utf-8"))
    keep = [item for item in base_records if item["kind"] in {
        "care_guidance", "drug_label", "drug_interaction"}]
    needed_sources = {item["source_id"] for item in keep}
    sources = [item for item in json.loads((base / "sources.json").read_text(encoding="utf-8"))
               if item["id"] in needed_sources]
    sources.append({
        "id": "huatuo26m-lite-open-slice",
        "title": "Huatuo26M-Lite 开放医学问答固定切片",
        "url": "https://huggingface.co/datasets/FreedomIntelligence/Huatuo26M-Lite",
        "publisher": "FreedomIntelligence",
        "jurisdiction": "CN",
        "scope": "Apache-2.0 开放研究数据；未经 CareGuide 临床审核；不代表医院或监管机构",
        "published_at": "2023-05-02",
        "verified_at": "2026-09-21",
        "valid_until": "2035-12-31",
        "review_status": "open_dataset_unreviewed",
        "clinical_reviewed": False,
        "verification_method": "dataset_revision_and_checksum",
        "locator": f"dataset revision {HUATUO_REVISION}; fixed seed {SEED}",
        "retrieved_sha256": sample_file["sha256"],
        "evidence_anchors": ["dataset revision", "fixed seed", "sample sha256"],
        "license": "Apache-2.0",
        "dataset_revision": HUATUO_REVISION,
    })
    records = list(keep)
    for item in sample:
        question = item["question"].strip()
        prefix = f"问题：{question}\n开放数据回答："
        answer = item["answer"].strip()[:max(0, 1000-len(prefix))]
        disease = str(item.get("related_diseases", "")).strip()
        keywords = list(dict.fromkeys(value for value in
            (item["department"], disease, question[:30]) if value))
        records.append({
            "id": f"huatuo-lite-{item['id']}",
            "source_id": "huatuo26m-lite-open-slice",
            "kind": "care_guidance",
            "title": question[:200],
            "content": prefix + answer,
            "keywords": keywords,
            "representation": "open_dataset_qa",
            "locator": f"Huatuo26M-Lite id={item['id']}; department={item['department']}",
        })
    OPEN_CORPUS.mkdir(parents=True, exist_ok=True)
    source_raw = (json.dumps(sources, ensure_ascii=False, indent=2) + "\n").encode()
    record_raw = (json.dumps(records, ensure_ascii=False, indent=2) + "\n").encode()
    (OPEN_CORPUS / "sources.json").write_bytes(source_raw)
    (OPEN_CORPUS / "records.json").write_bytes(record_raw)
    manifest = {
        "version": "careguide-open-medical-v1",
        "source_count": len(sources),
        "record_count": len(records),
        "files": {"sources.json": hashlib.sha256(source_raw).hexdigest(),
                  "records.json": hashlib.sha256(record_raw).hexdigest()},
        "composition": {"huatuo_open_qa": len(sample),
                        "retained_sourced_guidance_and_drug_records": len(keep)},
    }
    (OPEN_CORPUS / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OPEN_CORPUS / "README.md").write_text(
        "# CareGuide 开放医疗知识库 V1\n\n"
        f"默认演示语料由 {len(sample)} 条 Huatuo26M-Lite 固定切片和 {len(keep)} 条既有"
        "有来源的公众健康/药品资料组成。Huatuo记录是开放研究数据且未经临床审核，前端和引用"
        "必须展示其来源层级；既有医院专属科室及挂号记录没有进入本语料。\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=1000)
    args = parser.parse_args()
    normalize_cblue()
    build_open_corpus(args.sample_size)


if __name__ == "__main__":
    main()
