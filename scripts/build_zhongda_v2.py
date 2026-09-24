"""Build the second Zhongda pilot snapshot without modifying either prior corpus."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'data' / 'medical_zhongda_v1'
TARGET = ROOT / 'data' / 'medical_zhongda_v2'
HOSPITAL = {'hospital_id': 'zhongda', 'hospital_name': '东南大学附属中大医院', 'campus': '本部'}


def source(identity, title, path, digest, anchor):
    return {
        'id': identity, 'title': title, 'url': 'https://www.njzdyy.com/' + path,
        'publisher': HOSPITAL['hospital_name'], 'jurisdiction': 'CN',
        'scope': '官网科室介绍；仅摘录明确提及本部的信息', 'published_at': None,
        'verified_at': '2026-09-19', 'valid_until': '2026-12-18',
        'review_status': 'developer_source_verified', 'clinical_reviewed': False,
        'verification_method': 'developer_html_source_verification',
        'locator': '浏览器呈现的 HTML DOM 正文；本部与其他院区分别核对',
        'retrieved_sha256': digest, 'evidence_anchors': [anchor],
    }


def record(identity, source_id, department, content, keywords):
    return {
        'id': identity, 'source_id': source_id, 'kind': 'department',
        'title': f'中大医院本部{department}资料范围', 'content': content,
        'keywords': keywords, 'representation': 'curated_factual_summary',
        'department': department, **HOSPITAL,
    }


def main():
    if TARGET.exists():
        raise FileExistsError(f'Refusing to overwrite corpus: {TARGET}')
    sources = json.loads((BASE / 'sources.json').read_text(encoding='utf-8'))
    records = json.loads((BASE / 'records.json').read_text(encoding='utf-8'))
    sources += [
        source('zhongda-gastroenterology', '中大医院消化内科介绍', 'ksdh1/nkks/xhnk.htm',
               'dc0295219f6a9ea5b3c883daf46dbf6f96637c7eaa7293b98f5ce7f85b60e33a',
               '消化内镜中心两个，本部面积800平米'),
        source('zhongda-orthopedics', '中大医院骨科介绍', 'ksdh1/wkks/gk.htm',
               '944df0a13a2a4856fe01300cd217312bc09abb829e15aeeb6772e7d071d04b4b',
               '本部丁家桥院区开放床位129张'),
    ]
    records += [
        record('zhongda-gastroenterology-main-campus', 'zhongda-gastroenterology', '消化内科',
               '官网消化内科介绍明确列出本部设有消化内镜中心；科室介绍涉及消化道内镜及炎症性肠病等诊疗方向。页面也涉及江北院区，不能据此推断本部具体门诊项目、医生排班或个人应挂科室。',
               ['消化内科', '本部', '消化内镜', '炎症性肠病']),
        record('zhongda-orthopedics-main-campus', 'zhongda-orthopedics', '骨科',
               '官网骨科介绍明确提到本部丁家桥院区的骨科床位，并介绍脊柱外科、关节外科与运动医学、创伤骨科等亚专业。该介绍也涉及其他院区，具体本部门诊项目、排班和就诊地点须以官方挂号信息为准。',
               ['骨科', '本部', '丁家桥', '脊柱外科', '运动医学']),
        {
            'id': 'zhongda-main-campus-phone', 'source_id': 'zhongda-outpatient',
            'kind': 'hospital_guide', 'title': '中大医院本部咨询与预约电话',
            'content': '官网门诊指南页页脚将本部咨询、预约挂号电话列为025-83262114、025-83272114、025-83262001、025-83262003，并另列江北院区电话。此处只提供本部公开联系方式，不代拨、不保证实时接通或号源。',
            'keywords': ['本部', '咨询电话', '预约电话', '挂号电话', '025-83262114'],
            'representation': 'curated_factual_summary', 'section': 'channels', **HOSPITAL,
        },
    ]
    TARGET.mkdir(parents=True)
    for name, value in [('sources.json', sources), ('records.json', records)]:
        (TARGET / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    manifest = {
        'version': 'careguide-medical-zhongda-v2', 'verified_at': '2026-09-19',
        'clinical_reviewed': False, 'source_count': len(sources), 'record_count': len(records),
        'files': {name: hashlib.sha256((TARGET / name).read_bytes()).hexdigest()
                  for name in ('sources.json', 'records.json')},
    }
    (TARGET / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(f'Built {len(sources)} sources and {len(records)} records in {TARGET}')


if __name__ == '__main__':
    main()
