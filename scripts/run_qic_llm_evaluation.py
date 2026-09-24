"""Evaluate the configured chat model as a KUAKE-QIC few-shot branch.

The default profile selects 20 deterministic official-dev rows per label.  It
is intentionally a development diagnostic: the official dev split has already
been used for classifier selection and is not described as untouched held-out.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from careguide.evaluation.v3 import classification_report
from careguide.external_data import QIC_LABELS, load_kuake_qic_evaluation
from careguide.llm_utils import create_llm_client, track_token_usage
from careguide.qic_intent_recognizer import QICIntentRecognizer
from scripts.private_config import load_config


ROOT = Path(__file__).resolve().parents[1]


def deterministic_stratified_sample(rows: tuple[dict, ...], per_label: int) -> tuple[dict, ...]:
    selected = []
    for label in QIC_LABELS:
        matching = [row for row in rows if row["label"] == label]
        matching.sort(key=lambda row: hashlib.sha256(
            f'{label}\0{row["query"]}'.encode("utf-8")).hexdigest())
        selected.extend(matching[:per_label])
    return tuple(selected)


def load_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    cached = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            cached[row["id"]] = row
    return cached


async def run(args: argparse.Namespace) -> dict:
    load_config(args.config_file)
    key = os.environ.get("CAREGUIDE_LLM_API_KEY", "")
    if not key or key.startswith("YOUR_"):
        raise ValueError("CAREGUIDE_LLM_API_KEY is not configured")
    model = os.environ.get("CAREGUIDE_LLM_MODEL", "gpt-4o-mini")
    client = create_llm_client(key, os.environ.get("CAREGUIDE_LLM_BASE_URL") or None)
    recognizer = QICIntentRecognizer(client, model, embedding_mode="disabled")
    all_rows = load_kuake_qic_evaluation()
    rows = all_rows if args.full else deterministic_stratified_sample(all_rows, args.per_label)
    indexed = tuple({**row, "id": hashlib.sha256(
        f'{row["label"]}\0{row["query"]}'.encode("utf-8")).hexdigest()[:16]} for row in rows)
    cache = load_cache(args.cache)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def predict(row: dict) -> dict:
        if row["id"] in cache and not cache[row["id"]].get("failed"):
            return cache[row["id"]]
        async with semaphore:
            prediction = await recognizer._llm_qic(row["query"])
        return {"id": row["id"], "query": row["query"], "gold": row["label"],
                "predicted": prediction.get("label"),
                "confidence": prediction.get("confidence", 0.0),
                "reasoning": prediction.get("reasoning", ""),
                "failed": bool(prediction.get("failed")),
                "failure_type": prediction.get("failure_type")}

    args.cache.parent.mkdir(parents=True, exist_ok=True)
    pending = [row for row in indexed if row["id"] not in cache or cache[row["id"]].get("failed")]
    with track_token_usage() as usage:
        tasks = [asyncio.create_task(predict(row)) for row in pending]
        try:
            for task in asyncio.as_completed(tasks):
                value = await task
                cache[value["id"]] = value
                with args.cache.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        finally:
            await client.close()

    results = [cache[row["id"]] for row in indexed]
    scored = [row for row in results if not row.get("failed") and row.get("predicted") in QIC_LABELS]
    report = classification_report(
        [row["gold"] for row in scored], [row["predicted"] for row in scored], tuple(QIC_LABELS))
    result = {
        "evaluation": "KUAKE-QIC configured LLM few-shot development diagnostic",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "profile": "full_official_dev" if args.full else f"deterministic_stratified_{args.per_label}_per_label",
        "model": model, "requested_count": len(indexed), "scored_count": len(scored),
        "failure_count": len(indexed) - len(scored), "token_usage_new_calls": usage.to_dict(),
        "data_usage": "official dev development diagnostic; not untouched held-out",
        "clinical_validity_evaluated": False, **report, "rows": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--per-label", type=int, default=20)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--concurrency", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--cache", type=Path,
                        default=ROOT / "data/source_cache/qic/llm_fewshot_predictions_v1.jsonl")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/eval/results/cblue_qic_llm_sample_v1.json")
    args = parser.parse_args()
    result = asyncio.run(run(args))
    print(json.dumps({key: result[key] for key in (
        "model", "profile", "requested_count", "scored_count", "failure_count",
        "accuracy", "macro_f1", "token_usage_new_calls")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
