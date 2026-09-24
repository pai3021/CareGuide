"""Integrity-checked loaders for pinned third-party medical datasets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "data" / "external"
QIC_LABELS = (
    "疾病表述", "指标解读", "医疗费用", "治疗方案", "功效作用", "病情诊断",
    "其他", "注意事项", "病因分析", "就医建议", "后果表述",
)
CAREGUIDE_INTENT_LABELS = (*QIC_LABELS, "appointment_action", "emergency")


def _load_jsonl(directory: Path, filename: str, *, expected_version: str) -> tuple[dict, ...]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") != expected_version:
        raise ValueError("Unsupported external dataset manifest")
    raw = (directory / filename).read_bytes()
    expected = manifest.get("files", {}).get(filename)
    if not expected or hashlib.sha256(raw).hexdigest() != expected["sha256"]:
        raise ValueError(f"External dataset integrity mismatch: {filename}")
    values = tuple(json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip())
    if len(values) != expected["count"]:
        raise ValueError(f"External dataset count mismatch: {filename}")
    return values


def load_kuake_qic_evaluation() -> tuple[dict, ...]:
    """Return the untouched official dev split; runtime code must not call this."""
    values = _load_jsonl(ROOT / "cblue", "kuake_qic_dev.jsonl",
                         expected_version="careguide-cblue-snapshot-v2")
    if {item["label"] for item in values} != set(QIC_LABELS):
        raise ValueError("Unexpected KUAKE-QIC labels")
    return values


def load_kuake_qic_development() -> tuple[dict, ...]:
    """Return the official train split used for few-shot and embedding prototypes."""
    values = _load_jsonl(ROOT / "cblue", "kuake_qic_train.jsonl",
                         expected_version="careguide-cblue-snapshot-v2")
    if {item["label"] for item in values} != set(QIC_LABELS):
        raise ValueError("Unexpected KUAKE-QIC labels")
    return values


def load_kuake_qtr_development() -> tuple[dict, ...]:
    values = _load_jsonl(ROOT / "cblue", "kuake_qtr_train.jsonl",
                         expected_version="careguide-cblue-snapshot-v2")
    if {item["label"] for item in values} != {"0", "1", "2", "3"}:
        raise ValueError("Unexpected KUAKE-QTR labels")
    return values


def load_huatuo_open_slice() -> tuple[dict, ...]:
    directory = ROOT / "huatuo26m_lite"
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") != "careguide-huatuo26m-lite-slice-v1":
        raise ValueError("Unsupported Huatuo slice manifest")
    raw = (directory / "sample.jsonl").read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["file"]["sha256"]:
        raise ValueError("Huatuo slice integrity mismatch")
    values = tuple(json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip())
    if len(values) != manifest["file"]["count"]:
        raise ValueError("Huatuo slice count mismatch")
    return values
