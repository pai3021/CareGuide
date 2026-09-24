"""Versioned V3 fine-intent dataset contract. Held-out access is explicit."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DATASET_ROOT = Path(__file__).resolve().parents[1] / "data" / "eval" / "medical_intent_v3"
DOMAIN_BY_FINE = {
    **{label: "triage" for label in ("symptom_information", "department_navigation", "care_preparation",
                                      "report_process_guidance", "followup_guidance")},
    **{label: "medication" for label in ("drug_label", "drug_dosage_information",
                                          "drug_contraindications", "drug_precautions", "drug_interactions")},
    **{label: "appointment" for label in ("hospital_information", "appointment_policy",
                                           "appointment_materials", "doctor_search", "slot_search",
                                           "appointment_create", "appointment_reschedule",
                                           "appointment_cancel", "appointment_status")},
    "handoff_request": "local_workflow", "handoff_query": "local_workflow",
}
FINE_LABELS = tuple(DOMAIN_BY_FINE)
BOUNDARY_LABELS = ("urgent_handoff", "diagnosis_request", "individualized_medication", "out_of_scope")


class V3IntentCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str = Field(min_length=1)
    split: Literal["dev", "heldout"]
    label: str
    domain: str | None
    message: str = Field(min_length=1)
    provenance: Literal["developer_synthetic"]
    clinical_reviewed: Literal[False]


class V3BoundaryCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str = Field(min_length=1)
    split: Literal["dev", "heldout"]
    label: str
    message: str = Field(min_length=1)
    expected_handling: str = Field(min_length=1)
    provenance: Literal["developer_synthetic"]
    clinical_reviewed: Literal[False]


def _read(split: Literal["dev", "heldout"], root: Path = DATASET_ROOT) -> tuple[V3IntentCase, ...]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") != "careguide-medical-intent-v3" or manifest.get("labels") != list(FINE_LABELS):
        raise ValueError("Unsupported V3 intent manifest")
    filename = f"{split}.jsonl"; raw = (root / filename).read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["files"][filename]["sha256"]:
        raise ValueError(f"V3 intent dataset integrity mismatch: {filename}")
    cases = tuple(V3IntentCase.model_validate_json(line) for line in raw.decode().splitlines() if line.strip())
    expected_count = 30 if split == "dev" else 10
    counts = Counter(case.label for case in cases)
    if counts != Counter({label: expected_count for label in FINE_LABELS}):
        raise ValueError("V3 label balance mismatch")
    if len(cases) != manifest["files"][filename]["count"] or len({c.id for c in cases}) != len(cases):
        raise ValueError("V3 split count or identities mismatch")
    return cases


def load_v3_development_cases():
    return _read("dev")


def load_v3_evaluation_cases():
    return _read("heldout")


def _read_boundaries(split: Literal["dev", "heldout"], root: Path = DATASET_ROOT) -> tuple[V3BoundaryCase, ...]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    filename = f"boundary_{split}.jsonl"; raw = (root / filename).read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["files"][filename]["sha256"]:
        raise ValueError(f"V3 boundary dataset integrity mismatch: {filename}")
    cases = tuple(V3BoundaryCase.model_validate_json(line) for line in raw.decode().splitlines() if line.strip())
    expected_count = 20 if split == "dev" else 10
    if Counter(case.label for case in cases) != Counter({label: expected_count for label in BOUNDARY_LABELS}):
        raise ValueError("V3 boundary label balance mismatch")
    return cases


def load_v3_boundary_development_cases():
    return _read_boundaries("dev")


def load_v3_boundary_evaluation_cases():
    return _read_boundaries("heldout")
