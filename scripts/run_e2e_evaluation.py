"""Run CareGuide's workflow benchmark with an explicit held-out gate."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from careguide.action_store import ActionStore
from careguide.agent_orchestrator import AgentOrchestrator
from careguide.appointment_service import AppointmentService
from careguide.corpus import Corpus
from careguide.evaluator import LLMJudge
from careguide.hospital_workflow import InMemoryHospitalBackend
from careguide.knowledge_backends import EmbeddedKnowledgeBackend
from careguide.llm_utils import create_llm_client
from careguide.qic_intent_recognizer import QICIntentRecognizer
from careguide.retrieval.knowledge_base import MedicalKnowledgeBase
from careguide.retrieval.knowledge_base import versioned_collection_name
from careguide.runtime import chroma_client, semantic_embedding, semantic_reranker
from careguide.tools import MedicalToolRegistry
from careguide.trace_store import validate_trace_completeness


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "eval" / "careguide_e2e_v1"


def load_cases(split: str, *, allow_heldout: bool = False) -> list[dict]:
    if split not in {"dev", "heldout"}:
        raise ValueError("E2E split must be dev or heldout")
    if split == "heldout" and not allow_heldout:
        raise PermissionError("heldout requires the explicit --run-heldout gate")
    manifest = json.loads((DATASET / "manifest.json").read_text(encoding="utf-8"))
    name = f"{split}.jsonl"
    expected = manifest["files"][name]
    raw = (DATASET / name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected["sha256"]:
        raise ValueError(f"E2E {split} dataset integrity mismatch")
    cases = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if len(cases) != expected["count"] or any(case["split"] != split for case in cases):
        raise ValueError(f"E2E {split} split is malformed")
    return cases


def load_development_cases() -> list[dict]:
    return load_cases("dev")


def seed_appointment(backend: InMemoryHospitalBackend) -> None:
    backend.slots["SLOT-CARD-001"]["status"] = "busy"
    backend.appointments["APT-DEMO-001"] = {
        "id": "APT-DEMO-001", "patient_id": "e2e-user", "slot_id": "SLOT-CARD-001",
        "status": "booked", "department": "心血管内科",
        "start": backend.slots["SLOT-CARD-001"]["start"],
    }


def ratio(rows: list[dict], key: str) -> float | None:
    values = [bool(row[key]) for row in rows if key in row]
    return round(sum(values) / len(values), 4) if values else None


async def run_case(case: dict, recognizer: QICIntentRecognizer, registry: MedicalToolRegistry,
                   judge: LLMJudge, directory: Path, *, agent_client=None,
                   agent_model: str = "gpt-4o-mini") -> dict:
    backend = InMemoryHospitalBackend()
    if case.get("seed_appointment"):
        seed_appointment(backend)
    service = AppointmentService(backend, ActionStore(directory / f"{case['id']}.sqlite3"))
    engine = AgentOrchestrator(
        recognizer, registry,
        client=agent_client, model=agent_model,
        knowledge_backend=EmbeddedKnowledgeBackend(registry.knowledge, list(registry.corpus.records)),
        appointment_service=service,
    )
    before = await backend.list_appointments("e2e-user")
    payload = await engine.process(case["question"], "e2e-user", case["id"])
    after_chat = await backend.list_appointments("e2e-user")
    trace = engine.traces.get(payload["request_id"])
    expected = case["expected"]
    row = {
        "id": case["id"], "category": case["category"], "operation": case["operation"],
        "question": case["question"], "response": payload["response"], "expected": expected,
        "tools_used": [call["tool_name"] for call in trace["execution"]["tool_calls"]],
        "evidence_ids": [item.get("record_id") for item in trace.get("evidence", [])],
        "route_correct": (expected.get("agent_route") is None or
                          payload["agent_route"] == expected["agent_route"]),
        "action_correct": (expected.get("business_action") is None or
                           payload["business_action"] == expected["business_action"]),
        "boundary_correct": payload["boundary"] == expected.get("boundary"),
        "tool_correct": (expected.get("required_tool") is None or expected["required_tool"] in
                         [call["tool_name"] for call in trace["execution"]["tool_calls"]]),
        "trace_complete": validate_trace_completeness(trace)["complete"],
        "zero_write_before_confirmation": len(after_chat) == len(before),
        "workflow_status": payload["workflow_status"],
        "qic_intent": payload["qic_intent"],
    }
    if case["category"] == "knowledge":
        target_ids = set(case.get("target_evidence_ids", []))
        actual_ids = {identity for identity in row["evidence_ids"] if identity}
        answerable = bool(case.get("answerable", True))
        evidence_correct = bool(target_ids & actual_ids) if answerable else not actual_ids
        allowed_statuses = ({"completed"} if answerable else
                            {"needs_clarification", "not_covered"})
        row.update({
            "answerable": answerable,
            "target_evidence_ids": sorted(target_ids),
            "evidence_correct": evidence_correct,
            "answerability_behavior_correct": (
                evidence_correct and payload["workflow_status"] in allowed_statuses),
        })

    operation = case["operation"]
    if operation in {"confirm", "reject", "pending"}:
        actions = payload["action_requests"]
        row["confirmation_prepared"] = len(actions) == 1 and actions[0]["status"] == "pending_confirmation"
        if actions:
            action_id = actions[0]["id"]
            if operation == "confirm":
                completed = await service.confirm(action_id, "e2e-user", case["id"])
                first_state = await backend.list_appointments("e2e-user")
                replay = await service.confirm(action_id, "e2e-user", case["id"])
                second_state = await backend.list_appointments("e2e-user")
                row["state_consistent"] = completed["status"] == "completed" and first_state == second_state
                row["idempotent"] = replay["status"] == "completed" and first_state == second_state
            elif operation == "reject":
                rejected = service.reject(action_id, "e2e-user", case["id"])
                row["rejection_zero_write"] = (rejected["status"] == "rejected" and
                                                await backend.list_appointments("e2e-user") == before)
            else:
                row["pending_zero_write"] = await backend.list_appointments("e2e-user") == before

    scores = await judge.judge(case["question"], payload["response"], evidence={
        "case_contract": {
            "answerable": case.get("answerable"),
            "target_evidence_ids": case.get("target_evidence_ids", []),
            "allowed_outcome": case.get("allowed_outcome"),
            "expected": case["expected"],
        },
        "task_understanding": trace["task_understanding"], "routing": trace["routing"],
        "tool_calls": trace["execution"]["tool_calls"], "side_effects": trace["side_effects"],
        "actual_evidence_ids": row["evidence_ids"],
        "final_status": trace["final"]["workflow_status"],
    })
    row["judge"] = ({"status": "unscored", "error": scores.error} if scores.judge_failed else
                    {"status": "scored", "relevance": scores.relevance,
                     "accuracy": scores.accuracy, "completeness": scores.completeness,
                     "helpfulness": scores.helpfulness, "overall": scores.overall})
    deterministic = [value for key, value in row.items()
                     if key.endswith("_correct") or key in {"trace_complete", "zero_write_before_confirmation",
                         "confirmation_prepared", "state_consistent", "idempotent",
                         "rejection_zero_write", "pending_zero_write"}]
    row["deterministic_pass"] = all(deterministic)
    return row


async def run(judge_enabled: bool, limit: int | None, *, split: str = "dev",
              allow_heldout: bool = False, judge_provider: str = "dedicated",
              agent_provider: str = "offline") -> dict:
    cases = load_cases(split, allow_heldout=allow_heldout)
    if limit is not None:
        cases = cases[:limit]
    corpus = Corpus(ROOT / "data" / "careguide_kuake_rag_v1")
    embedding = semantic_embedding()
    collection = chroma_client().get_collection(
        versioned_collection_name(corpus), embedding_function=None)
    knowledge = MedicalKnowledgeBase(
        corpus, collection=collection, embedding=embedding, mode="hybrid",
        min_dense_score=-1.0, dense_weight=4.0, lexical_weight=1.0,
        hybrid_candidates=100, reranker=semantic_reranker(), rerank_candidates=20)
    registry = MedicalToolRegistry(knowledge)
    recognizer = QICIntentRecognizer(embedding_mode="local_ngram")
    prefix = "CAREGUIDE_LLM" if judge_provider == "chat" else "CAREGUIDE_JUDGE"
    key = os.environ.get(prefix + "_API_KEY", "") if judge_enabled else ""
    base_url = os.environ.get(prefix + "_BASE_URL") or None
    model = os.environ.get(prefix + "_MODEL", "gpt-4o-mini")
    client = create_llm_client(key, base_url) if key else None
    judge = LLMJudge(
        client, model, thinking_enabled=False,
        json_mode=os.environ.get("CAREGUIDE_JUDGE_JSON_MODE", "false").lower() == "true")
    agent_key = os.environ.get("CAREGUIDE_LLM_API_KEY", "") if agent_provider == "chat" else ""
    agent_model = os.environ.get("CAREGUIDE_LLM_MODEL", "gpt-4o-mini")
    if agent_provider == "chat" and not agent_key:
        raise RuntimeError("Chat agent provider requires CAREGUIDE_LLM_API_KEY")
    agent_client = create_llm_client(
        agent_key, os.environ.get("CAREGUIDE_LLM_BASE_URL") or None) if agent_key else None
    try:
        with tempfile.TemporaryDirectory(prefix="careguide-e2e-") as temporary:
            rows = []
            for index, case in enumerate(cases, 1):
                rows.append(await run_case(
                    case, recognizer, registry, judge, Path(temporary),
                    agent_client=agent_client, agent_model=agent_model))
                if index % 5 == 0 or index == len(cases):
                    print(f"evaluated {index}/{len(cases)}", flush=True)
    finally:
        if client is not None:
            await client.close()
        if agent_client is not None:
            await agent_client.close()
    scored = [row["judge"] for row in rows if row["judge"]["status"] == "scored"]
    judge_summary = {
        "status": "scored" if scored else "unscored",
        "configured": judge.configured, "model": judge.model,
        "provider_scope": "shared_chat_provider" if judge_provider == "chat" else "dedicated_judge_provider",
        "scored": len(scored), "unscored": len(rows) - len(scored),
    }
    if scored:
        for metric in ("relevance", "accuracy", "completeness", "helpfulness", "overall"):
            judge_summary[f"mean_{metric}"] = round(sum(item[metric] for item in scored) / len(scored), 4)
        judge_summary["pass_threshold"] = 0.75
        judge_summary["passed"] = sum(item["overall"] >= 0.75 for item in scored)
        judge_summary["pass_rate"] = round(judge_summary["passed"] / len(scored), 4)
        judge_summary["by_category"] = {
            category: {
                "count": len(category_rows),
                "mean_overall": round(sum(row["judge"]["overall"] for row in category_rows) /
                                      len(category_rows), 4),
                "passed": sum(row["judge"]["overall"] >= 0.75 for row in category_rows),
            }
            for category in ("knowledge", "appointment", "safety")
            if (category_rows := [row for row in rows if row["category"] == category])
        }
    deterministic_keys = ["route_correct", "action_correct", "boundary_correct", "tool_correct",
                          "trace_complete", "zero_write_before_confirmation", "confirmation_prepared",
                          "state_consistent", "idempotent", "rejection_zero_write", "pending_zero_write"]
    return {
        "version": "careguide-e2e-workflow-v3", "split": split,
        "heldout_used": split == "heldout",
        "evaluated_at": datetime.now(timezone.utc).isoformat(), "case_count": len(rows),
        "execution_profile": {
            "intent_semantic_backend": "local_ngram_workflow_isolation",
            "released_intent_model_evaluated_separately": "careguide-intent-roberta-v1",
            "retrieval": "hybrid_rrf_dense4_bm25_1_top100_rerank20",
            "hospital_backend": "in_memory_synthetic_contract",
            "agent_execution": ("provider_tool_loop" if agent_provider == "chat" else
                                "offline_deterministic_tool_fallback"),
            "agent_model": agent_model if agent_provider == "chat" else None,
        },
        "category_counts": dict(Counter(row["category"] for row in rows)),
        "deterministic": {**{key + "_rate": ratio(rows, key) for key in deterministic_keys},
                          "evidence_correct_rate": ratio(rows, "evidence_correct"),
                          "answerability_behavior_correct_rate": ratio(rows, "answerability_behavior_correct"),
                          "task_pass_rate": ratio(rows, "deterministic_pass")},
        "llm_judge": judge_summary, "clinical_validity_evaluated": False, "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--judge-provider", choices=("dedicated", "chat"), default="dedicated",
                        help="Use dedicated judge config, or explicitly reuse the chat provider")
    parser.add_argument("--agent-provider", choices=("offline", "chat"), default="offline",
                        help="Run deterministic fallback or the configured chat-model tool loop")
    parser.add_argument("--split", choices=("dev", "heldout"), default="dev")
    parser.add_argument("--run-heldout", action="store_true",
                        help="Required explicit gate for reading the frozen held-out split")
    parser.add_argument("--config-file", type=Path,
                        help="Explicit private CareGuide config; values are never printed")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.config_file:
        from scripts.private_config import load_config
        load_config(args.config_file)
    if args.split == "heldout" and not args.run_heldout:
        parser.error("--split heldout requires --run-heldout")
    report = asyncio.run(run(
        args.judge, args.limit, split=args.split, allow_heldout=args.run_heldout,
        judge_provider=args.judge_provider, agent_provider=args.agent_provider))
    if args.output is None:
        args.output = ROOT / f"data/eval/results/careguide_e2e_{args.split}_v3.json"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
