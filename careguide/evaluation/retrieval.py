"""Scoped record-level retrieval metrics; no answer or clinical scoring."""
from datetime import date
from typing import Any, Mapping


def evaluate_retrieval_cases(knowledge, cases: list[Mapping[str, Any]], *, today: date) -> dict[str, Any]:
    by_id = {record.id: record for record in knowledge.corpus.records}
    rows = []
    for case in cases:
        candidates = case['candidate_ids']
        unknown = (set(candidates) | set(case['relevant_ids'])) - set(by_id)
        if unknown:
            raise ValueError(f"{case['case_id']}: unknown corpus IDs {sorted(unknown)}")
        hits = knowledge.search(case['query'], [by_id[item] for item in candidates], today=today, top_k=5)
        ids = [item['record_id'] for item in hits]
        relevant = set(case['relevant_ids'])
        if not relevant and case.get('hard_negative') is not True:
            raise ValueError(f"{case['case_id']}: negatives must be explicitly marked")
        if relevant and not relevant <= set(candidates):
            raise ValueError(f"{case['case_id']}: relevant records must be eligible candidates")
        reciprocal_rank = next((1/rank for rank, item in enumerate(ids, 1) if item in relevant), 0.0)
        rows.append({'case_id': case['case_id'], 'hard_negative': not relevant,
            'expected': sorted(relevant), 'returned': ids,
            'recall_at_1': len(relevant & set(ids[:1]))/len(relevant) if relevant else None,
            'recall_at_3': len(relevant & set(ids[:3]))/len(relevant) if relevant else None,
            'recall_at_5': len(relevant & set(ids[:5]))/len(relevant) if relevant else None,
            'reciprocal_rank': reciprocal_rank if relevant else None,
            'negative_rejected': not ids if not relevant else None,
            'scope_violation': any(item not in candidates for item in ids)})
    positives = [row for row in rows if not row['hard_negative']]
    negatives = [row for row in rows if row['hard_negative']]
    def mean(key):
        return round(sum(row[key] for row in positives)/len(positives), 4) if positives else None
    return {'mode': knowledge.mode, 'total': len(rows), 'positive_count': len(positives),
            'hard_negative_count': len(negatives), 'recall_at_1': mean('recall_at_1'),
            'recall_at_3': mean('recall_at_3'), 'recall_at_5': mean('recall_at_5'),
            'mrr': mean('reciprocal_rank'),
            'hard_negative_accuracy': round(sum(row['negative_rejected'] for row in negatives)/len(negatives), 4)
            if negatives else None, 'scope_violations': sum(row['scope_violation'] for row in rows),
            'cases': rows, 'clinical_validity_evaluated': False}
