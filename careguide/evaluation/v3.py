"""Deterministic V3 routing, RAG and workflow metrics."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from math import log2


def classification_report(expected: list[str], predicted: list[str], labels: tuple[str, ...]) -> dict:
    rows = {}; total_correct = 0
    for label in labels:
        tp = sum(e == p == label for e, p in zip(expected, predicted))
        fp = sum(e != label and p == label for e, p in zip(expected, predicted))
        fn = sum(e == label and p != label for e, p in zip(expected, predicted))
        precision = tp/(tp+fp) if tp+fp else 0; recall = tp/(tp+fn) if tp+fn else 0
        f1 = 2*precision*recall/(precision+recall) if precision+recall else 0
        rows[label] = {"precision": round(precision, 4), "recall": round(recall, 4),
                       "f1": round(f1, 4), "support": sum(e == label for e in expected)}
        total_correct += tp
    return {"total": len(expected), "accuracy": round(total_correct/len(expected), 4) if expected else None,
            "macro_f1": round(sum(row["f1"] for row in rows.values())/len(rows), 4), "labels": rows}


def retrieval_report(knowledge, cases: list[dict], top_k: int = 5) -> dict:
    records = list(knowledge.corpus.records); rows = []
    for case in cases:
        hits = knowledge.search(case["query"], records,
                                today=datetime.now(timezone.utc).date(), top_k=top_k)
        returned = [item["record_id"] for item in hits]; relevant = set(case["gold_record_ids"])
        if not case.get("answerable", True):
            rows.append({"id": case["id"], "returned": returned,
                         "answerable": False, "correctly_rejected": not returned})
            continue
        rank = next((index for index, item in enumerate(returned, 1) if item in relevant), None)
        dcg = sum((1/log2(index+1)) for index, item in enumerate(returned, 1) if item in relevant)
        ideal = sum(1/log2(index+1) for index in range(1, min(len(relevant), top_k)+1)) or 1
        rows.append({"id": case["id"], "returned": returned, "answerable": True,
                     "recall_at_k": len(relevant & set(returned))/len(relevant),
                     "reciprocal_rank": 1/rank if rank else 0, "ndcg": dcg/ideal})
    positive = [row for row in rows if row["answerable"]]
    negative = [row for row in rows if not row["answerable"]]
    mean = lambda key: round(sum(row[key] for row in positive)/len(positive), 4) if positive else None
    return {"total": len(rows), "recall_at_k": mean("recall_at_k"), "mrr": mean("reciprocal_rank"),
            "ndcg": mean("ndcg"),
            "unanswerable_rejection_rate": round(sum(row["correctly_rejected"] for row in negative)/len(negative), 4) if negative else None,
            "answerable_count": len(positive), "unanswerable_count": len(negative),
            "rows": rows, "clinical_validity_evaluated": False}


def workflow_report(rows: list[dict]) -> dict:
    metrics = Counter()
    for row in rows:
        metrics["total"] += 1
        metrics["task_completed"] += bool(row.get("task_completed"))
        metrics["workflow_accurate"] += bool(row.get("workflow_accurate"))
        metrics["write_before_confirmation"] += bool(row.get("write_before_confirmation"))
        metrics["idempotent"] += bool(row.get("idempotent"))
    total = metrics["total"] or 1
    return {"total": metrics["total"], "task_completion_rate": round(metrics["task_completed"]/total, 4),
            "workflow_accuracy": round(metrics["workflow_accurate"]/total, 4),
            "unauthorized_write_count": metrics["write_before_confirmation"],
            "idempotency_rate": round(metrics["idempotent"]/total, 4)}
