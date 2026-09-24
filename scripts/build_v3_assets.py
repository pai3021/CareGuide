"""Build deterministic RAG evaluation cases and synthetic FHIR fixtures."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, values: list[dict]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = ("\n".join(json.dumps(v, ensure_ascii=False, separators=(",", ":")) for v in values)+"\n").encode()
    path.write_bytes(raw)
    return {"count": len(values), "sha256": hashlib.sha256(raw).hexdigest()}


def build_rag():
    records = json.loads((ROOT/"data/medical_zhongda_v2/records.json").read_text(encoding="utf-8"))
    target = ROOT/"data/eval/careguide_rag_v3"
    files = {}
    for split, count, offset in (("dev", 240, 0), ("heldout", 120, 1000)):
        values = []
        for index in range(count):
            if index % 8 == 7:
                values.append({"id": f"rag-v3-{split}-{index+1:03d}", "split": split,
                    "query": f"请查询江北院区眼科宠物疫苗停车收费资料（问法{index+1}）",
                    "gold_record_ids": [], "gold_source_ids": [], "answerable": False,
                    "scope": {"hospital_id": "not_in_corpus"},
                    "provenance": "developer_synthetic_hard_negative", "clinical_reviewed": False})
                continue
            record = records[(index * 7 + offset) % len(records)]
            keyword = (record.get("keywords") or [record["title"]])[index % len(record.get("keywords") or [record["title"]])]
            values.append({"id": f"rag-v3-{split}-{index+1:03d}", "split": split,
                "query": f"请查询{keyword}相关的有来源资料（问法{index+1}）",
                "gold_record_ids": [record["id"]], "gold_source_ids": [record["source_id"]],
                "answerable": True,
                "scope": {key: record.get(key) for key in ("hospital_id", "campus", "department") if record.get(key)},
                "provenance": "developer_synthetic_from_curated_record", "clinical_reviewed": False})
        files[f"{split}.jsonl"] = write_jsonl(target/f"{split}.jsonl", values)
    manifest = {"version": "careguide-rag-v3", "heldout_policy": "never used for tuning before freeze",
                "source_corpus": "medical_zhongda_v2", "files": files}
    (target/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (target/"README.md").write_text("""# CareGuide RAG V3 评测集\n\n240 条开发查询与 120 条隔离评测查询，由已核验语料标题和关键词确定性生成，并按固定比例加入不可回答的范围外查询。每条包含目标记录、来源、适用范围和可回答性。它评测资料定位与拒答，不评测临床正确性；held-out 在版本冻结前不得用于调参。\n""", encoding="utf-8")


def build_fhir():
    resources = [
      {"resourceType":"Organization","id":"careguide-synthetic-hospital","name":"CareGuide 合成医院"},
      {"resourceType":"Location","id":"demo-campus","name":"模拟院区","managingOrganization":{"reference":"Organization/careguide-synthetic-hospital"}},
      {"resourceType":"Patient","id":"u001","name":[{"text":"合成患者 u001"}]},
      {"resourceType":"Practitioner","id":"PRA-CARD-01","name":[{"text":"合成医生甲"}]},
      {"resourceType":"Practitioner","id":"PRA-GASTRO-01","name":[{"text":"合成医生乙"}]},
      {"resourceType":"PractitionerRole","id":"ROLE-CARD-01","practitioner":{"reference":"Practitioner/PRA-CARD-01"},"organization":{"reference":"Organization/careguide-synthetic-hospital"},"specialty":[{"coding":[{"system":"urn:careguide:department","code":"cardiology","display":"心血管内科"}]}]},
      {"resourceType":"PractitionerRole","id":"ROLE-GASTRO-01","practitioner":{"reference":"Practitioner/PRA-GASTRO-01"},"organization":{"reference":"Organization/careguide-synthetic-hospital"},"specialty":[{"coding":[{"system":"urn:careguide:department","code":"gastroenterology","display":"消化内科"}]}]},
      {"resourceType":"Schedule","id":"SCH-CARD-01","active":True,"actor":[{"reference":"Practitioner/PRA-CARD-01"}]},
      {"resourceType":"Schedule","id":"SCH-GASTRO-01","active":True,"actor":[{"reference":"Practitioner/PRA-GASTRO-01"}]},
    ]
    for dept, schedule, prefix, hour in (("心血管内科","SCH-CARD-01","CARD",9),("消化内科","SCH-GASTRO-01","GASTRO",10)):
        for index in range(1, 3):
            resources.append({"resourceType":"Slot","id":f"SLOT-{prefix}-{index:03d}",
                "schedule":{"reference":f"Schedule/{schedule}"},"status":"free",
                "start":f"2030-01-02T{hour:02d}:{(index-1)*15:02d}:00+08:00",
                "end":f"2030-01-02T{hour:02d}:{index*15:02d}:00+08:00",
                "serviceType":[{"coding":[{"system":"urn:careguide:department","display":dept}]}]})
    bundle = {"resourceType":"Bundle","type":"transaction","entry":[
        {"resource": resource, "request":{"method":"PUT","url":f"{resource['resourceType']}/{resource['id']}"}}
        for resource in resources]}
    target = ROOT/"data/synthetic_hospital"; target.mkdir(parents=True, exist_ok=True)
    (target/"fhir_bundle.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    digest = hashlib.sha256((target/"fhir_bundle.json").read_bytes()).hexdigest()
    (target/"manifest.json").write_text(json.dumps({"version":"careguide-synthetic-hospital-v1","seed":9999,
        "source_model":"H-AdminSim-inspired administrative schema","resource_count":len(resources),
        "sha256":digest,"real_patient_data":False}, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (target/"README.md").write_text("""# CareGuide 合成医院\n\n固定 seed 9999 的 FHIR R4 演示数据，包含虚构医院、院区、患者、医生、角色、排班和号源。所有姓名、标识和时间均为合成数据，不对应中大医院或任何真实医疗机构。\n\n导入：`python -m scripts.load_synthetic_fhir --apply`。\n""", encoding="utf-8")


if __name__ == "__main__":
    build_rag(); build_fhir()
