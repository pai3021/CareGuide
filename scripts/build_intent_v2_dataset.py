"""One-time synthetic V2 intent dataset builder; refuses to replace a split."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter

from careguide.domain import MedicalTask
from careguide.intent_dataset import Disposition, HistoryTurn
from careguide.intent_v2_dataset import (BOUNDARY_LABELS, DATASET_ROOT, DOMAIN_BY_FINE,
                                         FINE_LABELS, FineIntentCase)
from scripts.intent_v2_seed_cases import SEEDS


def parse_stem(stem: str):
    if not stem.startswith('HISTORY:'):
        return stem, ()
    match = re.fullmatch(r'HISTORY:(.*?)\|ASSISTANT:(.*?)\|USER:(.*)', stem)
    if not match:
        raise ValueError(f'Invalid history stem: {stem}')
    return match.group(3), (HistoryTurn(role='user', content=match.group(1)),
                            HistoryTurn(role='assistant', content=match.group(2)))


def build():
    if set(SEEDS) != set(FINE_LABELS) | set(BOUNDARY_LABELS):
        raise ValueError('Seed labels do not match the V2 taxonomy')
    if DATASET_ROOT.exists():
        raise FileExistsError(f'Refusing to replace V2 dataset: {DATASET_ROOT}')
    cases = {'dev': [], 'heldout': []}
    for label, by_split in SEEDS.items():
        for split, stems in by_split.items():
            for index, stem in enumerate(stems):
                message, history = parse_stem(stem)
                for wording in range(2):
                    variant = message if wording == 0 else f'请问：{message}'
                    if label in FINE_LABELS:
                        domain = DOMAIN_BY_FINE[label]
                        action = ('local_handoff_lookup' if label == 'handoff_query' else
                                  'local_handoff_only' if label == 'handoff_request' else
                                  'guidance_only_no_transaction' if domain == 'appointment' else
                                  'redirect_to_professional' if label == 'drug_dosage_information' else
                                  'sourced_information_only')
                        disposition = Disposition.ROUTE
                    else:
                        domain = None
                        action = {'urgent_handoff': 'urgent_handoff',
                                  'needs_clarification': 'clarify_task',
                                  'out_of_scope': 'out_of_scope'}[label]
                        disposition = Disposition(label)
                    case = FineIntentCase(
                        id=f'V2-{split.upper()}-{label}-{index:02d}-{wording}', split=split,
                        scenario_family=f'{label}-{split}-{index:02d}', message=variant,
                        history=history, disposition=disposition,
                        primary_fine_intent=MedicalTask(label) if domain else None,
                        fine_intents=(MedicalTask(label),) if domain else (),
                        primary_domain=domain, policy_action=action, missing_entities=(),
                        source_coverage='not_assessed',
                        annotation_rationale=f'开发者合成任务表达，标注为 {label}；仅评价任务路由。',
                        provenance='developer_synthetic', clinical_reviewed=False)
                    cases[split].append(case)
    if len(cases['dev']) != 360 or len(cases['heldout']) != 180:
        raise ValueError('V2 split size mismatch')
    if set(c.scenario_family for c in cases['dev']) & set(c.scenario_family for c in cases['heldout']):
        raise ValueError('Scenario family leaked across splits')
    if set((tuple(t.content for t in c.history), c.message) for c in cases['dev']) & set(
            (tuple(t.content for t in c.history), c.message) for c in cases['heldout']):
        raise ValueError('Message leaked across splits')
    DATASET_ROOT.mkdir(parents=True)
    files = {}
    for split, rows in cases.items():
        filename = f'{split}.jsonl'
        raw = (''.join(json.dumps(c.model_dump(mode='json'), ensure_ascii=False, sort_keys=True)
                       + '\n' for c in rows)).encode('utf-8')
        (DATASET_ROOT / filename).write_bytes(raw)
        files[filename] = {'count': len(rows), 'sha256': hashlib.sha256(raw).hexdigest(),
                           'label_counts': dict(sorted(Counter(
                               c.primary_fine_intent.value if c.primary_fine_intent else c.disposition.value
                               for c in rows).items()))}
    manifest = {'version': 'careguide-medical-intent-v2', 'labels': list(FINE_LABELS),
                'boundary_labels': list(BOUNDARY_LABELS), 'files': files,
                'provenance': 'developer_synthetic', 'clinical_reviewed': False,
                'split_policy': 'Distinct scenario stems by split; two wordings stay in the same family.',
                'heldout_policy': 'Never load into prompts, templates, rules or tuning. Run once after freezing V2.'}
    (DATASET_ROOT / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(f'Created {len(cases["dev"])} dev and {len(cases["heldout"])} heldout cases')


if __name__ == '__main__':
    build()
