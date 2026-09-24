"""Agent-independent deterministic Skill selection evaluation."""
from types import SimpleNamespace
from typing import Any, Mapping


def evaluate_skill_cases(registry, cases: list[Mapping[str, Any]]) -> dict[str, Any]:
    rows = []
    for case in cases:
        context = SimpleNamespace(task_type=case['task_type'],
            requested_action=case.get('requested_action', 'sourced_information'))
        selection = registry.select_for_agent(case['agent'], context)
        predicted = selection.definition.skill_id if selection else None
        rows.append({'case_id': case['case_id'], 'expected': case.get('expected_skill'),
                     'predicted': predicted, 'correct': predicted == case.get('expected_skill')})
    no_skill = [row for row in rows if row['expected'] is None]
    return {'total': len(rows), 'correct': sum(row['correct'] for row in rows),
            'exact_selection_accuracy': round(sum(row['correct'] for row in rows)/len(rows), 4) if rows else None,
            'no_skill_accuracy': round(sum(row['correct'] for row in no_skill)/len(no_skill), 4)
            if no_skill else None, 'external_llm_calls': 0, 'cases': rows}
