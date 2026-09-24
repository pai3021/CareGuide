"""Integrity-checked access to the CareGuide-13 intent dataset.

Training and validation have separate public loaders. Access to the frozen test
split is intentionally explicit so runtime and tuning code cannot import it by
accident through a combined development loader.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from careguide.external_data import CAREGUIDE_INTENT_LABELS


DATASET_ROOT = Path(__file__).resolve().parents[1] / "data" / "eval" / "careguide_intent_v4"
DATASET_VERSION = "careguide-intent-v4-13class-v1"
Split = Literal["train", "validation", "test"]


class V4IntentCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    split: Split
    label: str
    message: str = Field(min_length=1)
    scenario_family: str = Field(min_length=1)
    source_dataset: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    upstream_label: str = Field(min_length=1)
    provenance: Literal[
        "official_train",
        "developer_synthetic",
        "hand_authored_synthetic_user_utterance",
        "developer_curated_synthetic",
    ]
    synthetic: bool
    clinical_reviewed: Literal[False]

    @field_validator("label")
    @classmethod
    def known_label(cls, value: str) -> str:
        if value not in CAREGUIDE_INTENT_LABELS:
            raise ValueError(f"Unknown CareGuide-13 label: {value}")
        return value


def normalize_message(text: str) -> str:
    """Match the normalization used by the deterministic dataset builder."""
    return "".join(character for character in text.casefold() if character.isalnum())


def _read_manifest(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") != DATASET_VERSION:
        raise ValueError("Unsupported CareGuide-13 intent manifest")
    if manifest.get("labels") != list(CAREGUIDE_INTENT_LABELS):
        raise ValueError("CareGuide-13 label contract mismatch")
    if manifest.get("official_qic_dev_used") is not False:
        raise ValueError("CareGuide-13 must not use the official QIC dev split")
    return manifest


def _read(split: Split, root: Path = DATASET_ROOT) -> tuple[V4IntentCase, ...]:
    manifest = _read_manifest(root)
    filename = f"{split}.jsonl"
    expected = manifest.get("files", {}).get(filename)
    if not expected:
        raise ValueError(f"CareGuide-13 manifest is missing {filename}")

    raw = (root / filename).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected.get("sha256"):
        raise ValueError(f"CareGuide-13 dataset integrity mismatch: {filename}")

    cases = tuple(
        V4IntentCase.model_validate_json(line)
        for line in raw.decode("utf-8").splitlines()
        if line.strip()
    )
    if len(cases) != expected.get("count"):
        raise ValueError(f"CareGuide-13 dataset count mismatch: {filename}")
    if any(case.split != split for case in cases):
        raise ValueError(f"CareGuide-13 split marker mismatch: {filename}")
    if len({case.id for case in cases}) != len(cases):
        raise ValueError(f"CareGuide-13 duplicate ids: {filename}")
    if len({normalize_message(case.message) for case in cases}) != len(cases):
        raise ValueError(f"CareGuide-13 normalized duplicate messages: {filename}")
    if Counter(case.label for case in cases) != Counter(expected.get("labels", {})):
        raise ValueError(f"CareGuide-13 label counts mismatch: {filename}")
    return cases


def load_v4_training_cases(root: Path = DATASET_ROOT) -> tuple[V4IntentCase, ...]:
    return _read("train", root)


def load_v4_validation_cases(root: Path = DATASET_ROOT) -> tuple[V4IntentCase, ...]:
    return _read("validation", root)


def load_v4_test_cases(root: Path = DATASET_ROOT) -> tuple[V4IntentCase, ...]:
    """Load the frozen test split explicitly; do not use this during tuning."""
    return _read("test", root)
