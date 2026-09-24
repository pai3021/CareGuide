"""Audit dataset contracts and split isolation; never run a recognizer."""
import hashlib
import json
import re
from collections import Counter

from careguide.domain import AGENTS, ROUTES, IntentCategory
from careguide.intent_dataset import DATASET_ROOT, load_development_cases, load_evaluation_cases


def normalized(text):
    return re.sub(r'[^\w]', '', text.casefold())


def trigrams(text):
    value = normalized(text)
    return {value[i:i+3] for i in range(max(0, len(value)-2))}


def audit():
    dev, heldout = load_development_cases(), load_evaluation_cases()
    assert (len(dev), len(heldout)) == (240, 120)
    assert len({c.id for c in dev + heldout}) == 360
    assert not ({c.scenario_family for c in dev} & {c.scenario_family for c in heldout})
    assert not ({normalized(c.message) for c in dev} & {normalized(c.message) for c in heldout})
    for cases, domain_count, boundary_count in ((dev, 50, 20), (heldout, 25, 10)):
        counts = Counter(c.primary_intent.value if c.primary_intent else c.disposition.value for c in cases)
        assert counts == dict(symptom_query=domain_count, medication_query=domain_count,
                              appointment=domain_count, emergency=domain_count,
                              needs_clarification=boundary_count, out_of_scope=boundary_count)
        for case in cases:
            # V1 is an immutable historical contract: emergency used to
            # short-circuit with no formal agent. CareGuide-13 now routes the
            # same class to EscalationAgent and is validated by runtime tests.
            agents = (() if IntentCategory.EMERGENCY in case.expected_intents else
                      tuple(dict.fromkeys(AGENTS[ROUTES[i]].name
                                          for i in case.expected_intents if i in ROUTES)))
            assert case.expected_agents == agents, case.id
    flags = []
    for a in dev:
        left = trigrams(a.message)
        for b in heldout:
            right = trigrams(b.message)
            score = len(left & right)/len(left | right) if left | right else 0
            if score >= 0.65:
                flags.append({'dev_id': a.id, 'heldout_id': b.id, 'trigram_jaccard': round(score, 4)})
    return {'dataset_manifest_sha256': hashlib.sha256((DATASET_ROOT/'manifest.json').read_bytes()).hexdigest(),
            'case_count': 360, 'schema_and_labels_valid': True, 'unique_ids': True,
            'disjoint_scenario_families': True, 'no_cross_split_normalized_message_duplicates': True,
            'cross_split_trigram_threshold': 0.65, 'near_duplicate_flags': flags,
            'recognition_evaluation_run': False,
            'limitations': 'String checks cannot establish semantic independence or clinical validity.'}


def main():
    result = audit()
    (DATASET_ROOT/'validation.json').write_bytes((json.dumps(result, ensure_ascii=False, indent=2)+'\n').encode('utf-8'))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
