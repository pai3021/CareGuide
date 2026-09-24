"""Export the frozen Zhongda V2 corpus as attributable WeKnora Markdown."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    source_root = ROOT/"data/medical_zhongda_v2"
    target = ROOT/"data/weknora_import/medical_zhongda_v2"
    target.mkdir(parents=True, exist_ok=True)
    records = json.loads((source_root/"records.json").read_text(encoding="utf-8"))
    sources = {item["id"]: item for item in json.loads((source_root/"sources.json").read_text(encoding="utf-8"))}
    grouped = defaultdict(list)
    for record in records: grouped[record["source_id"]].append(record)
    files = []
    for source_id, items in sorted(grouped.items()):
        source = sources[source_id]
        lines = [f"# {source.get('title', source_id)}", "",
                 f"- source_id: `{source_id}`", f"- url: {source.get('url', '')}",
                 f"- checked_at: {source.get('checked_at', '')}", ""]
        for record in items:
            lines += [f"## {record['title']}", "", record["content"], "",
                      f"- record_id: `{record['id']}`",
                      f"- hospital_id: `{record.get('hospital_id', '')}`",
                      f"- campus: `{record.get('campus', '')}`",
                      f"- department: `{record.get('department', '')}`", ""]
        path = target/f"{source_id}.md"; path.write_text("\n".join(lines), encoding="utf-8")
        files.append({"path": path.name, "source_id": source_id,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "record_count": len(items)})
    (target/"manifest.json").write_text(json.dumps({"version":"weknora-import-medical-zhongda-v2",
        "source_corpus_fingerprint": json.loads((source_root/"manifest.json").read_text(encoding="utf-8"))["files"]["records.json"],
        "files":files}, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (target/"README.md").write_text("""# WeKnora 导入包\n\n由 `medical_zhongda_v2` 确定性导出。每个 Markdown 文件保留 source_id、原文 URL、核验日期及记录适用范围。导入 WeKnora 后必须把知识库限制为只读 MCP 检索工具；不要开放写入工具给 CareGuide。\n""", encoding="utf-8")


if __name__ == "__main__": main()
