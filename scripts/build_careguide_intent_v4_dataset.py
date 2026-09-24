"""Build the frozen CareGuide-13 train/validation/test dataset.

KUAKE-QIC contributes all 6,931 official-train rows. Appointment and emergency
rows reuse versioned CareGuide scenario families and add bounded, explicitly
marked synthetic families for workflow coverage. Historical source files are
read only and are never rewritten.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from careguide.external_data import CAREGUIDE_INTENT_LABELS, load_kuake_qic_development
from careguide.intent_dataset import load_development_cases, load_evaluation_cases
from careguide.intent_v2_dataset import (load_v2_development_cases,
                                         load_v2_evaluation_cases)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/eval/careguide_intent_v4"
SPLITS = ("train", "validation", "test")
SEED = "careguide-intent-v4-2026-09-22"


APPOINTMENT_BASES = {
    "doctor_search": (
        "合成医院消化内科有哪些可以选择的医生",
        "我想查看合成医院的消化内科医生列表",
        "请帮我找合成医院消化内科的模拟医生",
    ),
    "slot_search": (
        "合成医院消化内科最近有哪些可预约号源",
        "我想查询合成医院下周的模拟号源",
        "请列出合成医院消化内科可选的就诊时间",
    ),
    "appointment_create": (
        "我选好了模拟号源，想创建预约",
        "请为我准备这个合成医院号源的预约",
        "确认信息后帮我提交模拟预约",
    ),
    "appointment_reschedule": (
        "我想把现有模拟预约改到另一个号源",
        "请帮我准备预约改期",
        "这个合成医院预约需要更换就诊时间",
    ),
    "appointment_cancel": (
        "我想取消现有的模拟预约",
        "请帮我准备退掉这个合成医院预约",
        "这个预约不需要了，确认后取消",
    ),
    "appointment_status": (
        "我想查询模拟预约现在是什么状态",
        "请查看这个合成医院预约是否成功",
        "帮我确认预约记录目前的处理状态",
    ),
}

EMERGENCY_BASES = (
    "突然胸口剧烈疼痛而且呼吸困难",
    "胸痛同时冒冷汗并且越来越喘",
    "怎么叫都叫不醒而且呼吸很弱",
    "突然失去意识倒在地上",
    "正在持续抽搐并且没有恢复意识",
    "突然一侧手脚无力而且说话含糊",
    "突然嘴歪并且无法正常说话",
    "吃东西后喉咙迅速肿起并且喘不上气",
    "全身起疹子的同时出现严重呼吸困难",
    "伤口大量出血一直止不住而且开始发晕",
    "从高处摔下后意识不清",
    "头部受伤后反复呕吐并且越来越嗜睡",
    "误服大量药物后意识逐渐模糊",
    "接触不明物质后呼吸困难并且意识异常",
    "孕期突然大量出血并伴有剧烈腹痛",
    "高烧的同时出现意识混乱和叫不醒",
    "突然出现从未有过的剧烈头痛随后昏倒",
    "婴儿嘴唇发紫而且呼吸非常微弱",
    "严重喘息导致无法说出完整句子",
)


def stable_key(value: str) -> str:
    return hashlib.sha256(f"{SEED}\0{value}".encode("utf-8")).hexdigest()


def normalized(text: str) -> str:
    return "".join(character for character in str(text).casefold() if character.isalnum())


def source_record(*, message: str, family: str, source_dataset: str,
                  source_id: str, upstream_label: str, provenance: str,
                  synthetic: bool) -> dict:
    return {
        "message": message.strip(), "scenario_family": family,
        "source_dataset": source_dataset, "source_id": source_id,
        "upstream_label": upstream_label, "provenance": provenance,
        "synthetic": synthetic, "clinical_reviewed": False,
    }


def qic_rows() -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in load_kuake_qic_development():
        grouped[row["label"]].append(source_record(
            message=row["query"], family=f'qic:{row["id"]}',
            source_dataset="KUAKE-QIC", source_id=row["id"],
            upstream_label=row["label"], provenance="official_train", synthetic=False))
    result = {split: [] for split in SPLITS}
    for label, rows in grouped.items():
        rows.sort(key=lambda row: stable_key(row["source_id"]))
        train_end = round(len(rows) * 0.70)
        validation_end = train_end + round(len(rows) * 0.15)
        for split, selected in (
            ("train", rows[:train_end]),
            ("validation", rows[train_end:validation_end]),
            ("test", rows[validation_end:]),
        ):
            result[split].extend({**row, "label": label} for row in selected)
    return result


def legacy_groups(kind: str) -> tuple[list[list[dict]], list[list[dict]]]:
    v1_groups: dict[str, list[dict]] = defaultdict(list)
    for case in (*load_development_cases(), *load_evaluation_cases()):
        is_match = ((kind == "appointment_action" and
                     getattr(case.primary_intent, "value", None) == "appointment") or
                    (kind == "emergency" and case.disposition.value == "urgent_handoff"))
        if not is_match:
            continue
        family = f"v1:{case.scenario_family}"
        v1_groups[family].append(source_record(
            message=case.message, family=family, source_dataset="careguide-medical-intent-v1",
            source_id=case.id, upstream_label=(getattr(case.primary_intent, "value", None)
                                                or case.disposition.value),
            provenance=case.provenance, synthetic=True))

    v2_groups: dict[str, list[dict]] = defaultdict(list)
    for case in (*load_v2_development_cases(), *load_v2_evaluation_cases()):
        is_match = ((kind == "appointment_action" and case.primary_domain == "appointment") or
                    (kind == "emergency" and case.disposition.value == "urgent_handoff"))
        if not is_match:
            continue
        family = f"v2:{case.scenario_family}"
        v2_groups[family].append(source_record(
            message=case.message, family=family, source_dataset="careguide-medical-intent-v2",
            source_id=case.id, upstream_label=(getattr(case.primary_fine_intent, "value", None)
                                                or case.disposition.value),
            provenance=case.provenance, synthetic=True))
    first = sorted(v1_groups.values(), key=lambda group: stable_key(group[0]["scenario_family"]))
    second = sorted(v2_groups.values(), key=lambda group: stable_key(group[0]["scenario_family"]))
    return first, second


def appointment_curated_groups() -> list[list[dict]]:
    prefixes = ("", "请帮我", "我现在想")
    groups = []
    for action, bases in APPOINTMENT_BASES.items():
        for family_index, base in enumerate(bases, start=1):
            family = f"appointment-curated:{action}:{family_index}"
            group = []
            for expression_index, prefix in enumerate(prefixes, start=1):
                message = f"{prefix}{base}" if prefix else base
                group.append(source_record(
                    message=message, family=family, source_dataset="careguide-intent-v4-curated",
                    source_id=f"appointment-{action}-{family_index}-{expression_index}",
                    upstream_label=action, provenance="developer_curated_synthetic",
                    synthetic=True))
            groups.append(group)
    return groups


def emergency_curated_groups() -> list[list[dict]]:
    prefixes = ("", "现在", "家里人", "我这里有人", "情况很急，")
    groups = []
    for family_index, base in enumerate(EMERGENCY_BASES, start=1):
        family = f"emergency-curated:{family_index}"
        group = []
        for expression_index, prefix in enumerate(prefixes, start=1):
            message = f"{prefix}{base}" if prefix else base
            group.append(source_record(
                message=message, family=family, source_dataset="careguide-intent-v4-curated",
                source_id=f"emergency-{family_index}-{expression_index}",
                upstream_label="urgent_handoff", provenance="developer_curated_synthetic",
                synthetic=True))
        groups.append(group)
    return groups


def add_groups(target: dict[str, list[dict]], split: str, label: str,
               groups: list[list[dict]]) -> None:
    for group in groups:
        target[split].extend({**row, "label": label} for row in group)


def custom_rows() -> dict[str, list[dict]]:
    result = {split: [] for split in SPLITS}

    appointment_v1, appointment_v2 = legacy_groups("appointment_action")
    appointment_curated = sorted(appointment_curated_groups(),
                                 key=lambda group: stable_key(group[0]["scenario_family"]))
    # Each split receives all six appointment action families through one
    # three-expression family per action. Legacy families provide broader
    # process/material/follow-up language.
    curated_by_action: dict[str, list[list[dict]]] = defaultdict(list)
    for group in appointment_curated:
        curated_by_action[group[0]["upstream_label"]].append(group)
    curated_split = {split: [] for split in SPLITS}
    for groups in curated_by_action.values():
        groups.sort(key=lambda group: stable_key(group[0]["scenario_family"]))
        for split, group in zip(SPLITS, groups):
            curated_split[split].append(group)
    add_groups(result, "train", "appointment_action",
               appointment_v1[:6] + appointment_v2[:46] + curated_split["train"])
    add_groups(result, "validation", "appointment_action",
               appointment_v1[6:8] + appointment_v2[46:47] + curated_split["validation"])
    add_groups(result, "test", "appointment_action",
               appointment_v1[8:10] + appointment_v2[47:48] + curated_split["test"])

    emergency_v1, emergency_v2 = legacy_groups("emergency")
    emergency_curated = sorted(emergency_curated_groups(),
                               key=lambda group: stable_key(group[0]["scenario_family"]))
    add_groups(result, "train", "emergency",
               emergency_v1[:9] + emergency_v2[:5] + emergency_curated[:17])
    add_groups(result, "validation", "emergency",
               emergency_v1[9:12] + emergency_v2[5:10] + emergency_curated[17:18])
    add_groups(result, "test", "emergency",
               emergency_v1[12:15] + emergency_v2[10:15] + emergency_curated[18:19])
    return result


def build(output: Path = OUTPUT) -> dict:
    qic = qic_rows()
    custom = custom_rows()
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    all_families = {}
    all_messages = {}
    for split in SPLITS:
        rows = qic[split] + custom[split]
        rows.sort(key=lambda row: (list(CAREGUIDE_INTENT_LABELS).index(row["label"]),
                                   stable_key(row["source_id"])))
        for index, row in enumerate(rows, start=1):
            row["id"] = f"CG13-{split.upper()}-{index:05d}"
            row["split"] = split
        families = {row["scenario_family"] for row in rows}
        messages = {normalized(row["message"]) for row in rows}
        if len(messages) != len(rows):
            raise ValueError(f"Normalized duplicate inside {split}")
        for previous_split, previous in all_families.items():
            if families & previous:
                raise ValueError(f"Scenario family leakage: {previous_split}/{split}")
        for previous_split, previous in all_messages.items():
            if messages & previous:
                raise ValueError(f"Normalized message leakage: {previous_split}/{split}")
        all_families[split], all_messages[split] = families, messages
        raw = ("\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                         for row in rows) + "\n").encode("utf-8")
        filename = f"{split}.jsonl"
        (output / filename).write_bytes(raw)
        files[filename] = {
            "count": len(rows), "sha256": hashlib.sha256(raw).hexdigest(),
            "labels": dict(Counter(row["label"] for row in rows)),
        }
    manifest = {
        "version": "careguide-intent-v4-13class-v1",
        "labels": list(CAREGUIDE_INTENT_LABELS), "split_seed": SEED,
        "split_policy": "stratified QIC rows; custom scenario families never cross splits",
        "official_qic_dev_used": False,
        "clinical_reviewed": False,
        "files": files,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# CareGuide-13 意图数据集\n\n"
        "KUAKE-QIC 官方 train 的 6931 条样本，加 200 条预约与 200 条急症表达。"
        "按类别和场景族拆分为 train、validation、test；同一场景族不跨 split。\n\n"
        "自建数据为开发者整理的合成表达，未经临床审核。官方 QIC dev 不参与本版本切分。"
        "冻结 test 在版本定型前不得进入 prompt、centroid、规则、阈值或错误分析。\n",
        encoding="utf-8")
    return manifest


def main() -> None:
    manifest = build()
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
