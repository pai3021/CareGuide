"""Build the full KUAKE-IR FTS index and evaluate the official dev split."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from careguide.evaluation.kuake_ir import KuakeIRFTSIndex, evaluate_queries
from scripts.import_kuake_ir import EXPECTED, DEFAULT_SOURCE, sha256


ROOT = Path(__file__).resolve().parents[1]


def pairs(path: Path) -> dict[str, str]:
    values = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            key, value = line.rstrip("\r\n").split("\t", 1)
            values[key] = value
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--index", type=Path,
                        default=DEFAULT_SOURCE / "kuake_ir_fts.sqlite3")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/eval/results/kuake_ir_fts_dev_v1.json")
    args = parser.parse_args()
    source = args.source_dir.resolve()
    for name in ("corpus.tsv", "KUAKE-IR_dev_query.txt", "KUAKE-IR_dev.tsv"):
        if sha256(source / name) != EXPECTED[name][1]:
            raise ValueError(f"KUAKE-IR source integrity mismatch: {name}")

    index = KuakeIRFTSIndex(args.index.resolve())
    if args.rebuild or not index.path.exists():
        count = index.build(source / "corpus.tsv")
        if count != EXPECTED["corpus.tsv"][0]:
            raise ValueError("KUAKE-IR index did not consume the full corpus")
    if index.document_count() != EXPECTED["corpus.tsv"][0]:
        raise ValueError("KUAKE-IR index document count mismatch")

    report = evaluate_queries(
        index, pairs(source / "KUAKE-IR_dev_query.txt"), pairs(source / "KUAKE-IR_dev.tsv"))
    report = {
        "evaluation": "KUAKE-IR official dev full-corpus lexical retrieval",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "retriever": index.version,
        "corpus_count": index.document_count(),
        "data_usage": "official dev external evaluation; test labels are unavailable",
        **report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {key: value for key, value in report.items() if key != "rows"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
