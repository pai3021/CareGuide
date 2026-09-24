"""Build the independent Zhongda main-campus pilot corpus once.

The source hashes are SHA-256 of UTF-8 browser-rendered HTML DOM verified on
2026-09-19. This does not assert a stable server response or clinical review.
The frozen medical_v1 input is read only; an existing output is never replaced.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'data' / 'medical_v1'
TARGET = ROOT / 'data' / 'medical_zhongda_v1'
HOSPITAL = {'hospital_id': 'zhongda', 'hospital_name': '东南大学附属中大医院', 'campus': '本部'}


def source(identity, title, path, digest, scope, anchors, *, days=30):
    return {
        'id': identity, 'title': title, 'url': 'https://www.njzdyy.com/' + path,
        'publisher': HOSPITAL['hospital_name'], 'jurisdiction': 'CN',
        'scope': scope, 'published_at': None, 'verified_at': '2026-09-19',
        'valid_until': '2026-10-19' if days == 30 else '2026-12-18',
        'review_status': 'developer_source_verified', 'clinical_reviewed': False,
        'verification_method': 'developer_html_source_verification',
        'locator': '浏览器呈现的 HTML DOM 正文；摘要按本部适用范围摘录',
        'retrieved_sha256': digest, 'evidence_anchors': anchors,
    }


def guide(identity, source_id, section, title, content, keywords):
    return {
        'id': identity, 'source_id': source_id, 'kind': 'hospital_guide',
        'title': title, 'content': content, 'keywords': keywords,
        'representation': 'curated_factual_summary', 'section': section,
        **HOSPITAL,
    }


def main():
    if TARGET.exists():
        raise FileExistsError(f'Refusing to overwrite corpus: {TARGET}')
    old_sources = json.loads((BASE / 'sources.json').read_text(encoding='utf-8'))
    old_records = json.loads((BASE / 'records.json').read_text(encoding='utf-8'))
    sources = [s for s in old_sources if s['id'] in {'nmpa-otc-labels', 'nhs-headaches'}]
    records = [r for r in old_records if r['source_id'] in {'nmpa-otc-labels', 'nhs-headaches'}]
    sources += [
        source('zhongda-outpatient', '中大医院门诊指南', 'hzfw/mzzn.htm',
               'e5963b80e9a10ca89255df12f1eed6288d5aa2d58824b337cfc062207c7e4893',
               '门诊流程与本部咨询入口', ['开诊时间', '挂号须知', '丁家桥87号']),
        source('zhongda-card', '中大医院就诊卡办理', 'hzfw/mzzn/jzkbl.htm',
               '46c4586ad991b2386a3326a510268d1fa735eef9f8642f32bdb4e380b85fa63d',
               '本部门诊挂号身份材料', ['市民卡A卡', '身份证原件', 'C卡']),
        source('zhongda-pharmacy', '中大医院取药须知', 'hzfw/mzzn/qyxz.htm',
               '0d4cf182b81e942105a9207a4c062500d786cc67c964429dd9f114876d5bf32a',
               '门诊普通患者取药流程', ['普通患者', '取药报到机']),
        source('zhongda-respiratory', '中大医院呼吸内科介绍', 'ksdh1/nkks/hxnk1.htm',
               '93b05c4bb0c987ce00204667888442f249c8848cebacdd18b97d62aede9ef340',
               '呼吸内科介绍；仅摘录明确提及本部的信息', ['本部现有编制床位', '慢性咳嗽'], days=90),
    ]
    records += [
        guide('zhongda-booking-channel', 'zhongda-outpatient', 'channels', '中大医院本部预约入口',
              '官网门诊指南提供“预约挂号”入口，链接至医院互联网医院页面。这里只指引官方入口，不查询号源，也不完成挂号；请在页面核对本部及具体就诊地点。',
              ['中大医院', '本部', '预约挂号', '官网', '入口']),
        guide('zhongda-opening-hours', 'zhongda-outpatient', 'process', '中大医院门诊开诊与挂号时间',
              '官网门诊指南列出：周一至周六上午8:00—12:00、下午14:00—17:30，周日上午8:00—12:00；挂号时间提前半小时。页面另提示丁家桥与湖南路门诊可能涉及不同就诊地址，须以挂号单为准。',
              ['门诊时间', '开诊', '挂号时间', '丁家桥', '湖南路']),
        guide('zhongda-visit-flow', 'zhongda-outpatient', 'process', '中大医院门诊就诊流程',
              '官网门诊指南列出挂号、按挂号单楼层候诊及叫号、医生看诊后缴费，再到相应部门检查、取药或治疗的流程。请按挂号单上的实际就诊地址与现场指引办理。',
              ['门诊流程', '候诊', '缴费', '挂号单']),
        guide('zhongda-insurance-materials', 'zhongda-outpatient', 'materials', '中大医院门诊挂号材料',
              '官网门诊指南写明：医保患者凭医保卡或医保码，非医保患者携身份证至挂号处或自助机办理挂号。',
              ['医保卡', '医保码', '身份证', '挂号材料']),
        guide('zhongda-special-card', 'zhongda-card', 'materials', '中大医院特殊人群就诊卡办理',
              '官网就诊卡办理页写明：外籍人士及无身份证号码的新生儿，可到丁家桥门诊二楼预约中心或湖南路一楼综合服务中心办理C卡；自费患者可携身份证原件至窗口或自助机直接挂号。页面提及C卡工本费，金额请以医院当前公示为准。',
              ['C卡', '外籍', '新生儿', '自费', '身份证']),
        guide('zhongda-medicine-pickup', 'zhongda-pharmacy', 'process', '中大医院普通门诊取药流程',
              '官网取药须知写明：普通患者缴费后在药房门口取药报到机报到，在相应窗口等候叫号取药，并听取药师用药交代。这里只说明取药流程，不给个人用药方案。',
              ['取药', '药房', '报到机', '叫号']),
        {
            'id': 'zhongda-respiratory-main-campus', 'source_id': 'zhongda-respiratory',
            'kind': 'department', 'title': '中大医院本部呼吸内科资料范围',
            'content': '官网呼吸内科介绍明确提到本部设有该科床位，并介绍肺结节、呼吸道疾病及慢性咳嗽等诊疗方向。页面同时涉及江北院区；本摘要不代表这些服务均在本部开设门诊，具体出诊和地点须查官方挂号页。',
            'keywords': ['呼吸内科', '本部', '慢性咳嗽', '肺结节'],
            'representation': 'curated_factual_summary', 'department': '呼吸内科', **HOSPITAL,
        },
    ]
    TARGET.mkdir(parents=True)
    for name, value in [('sources.json', sources), ('records.json', records)]:
        (TARGET / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    manifest = {
        'version': 'careguide-medical-zhongda-v1', 'verified_at': '2026-09-19',
        'clinical_reviewed': False, 'source_count': len(sources), 'record_count': len(records),
        'files': {name: hashlib.sha256((TARGET / name).read_bytes()).hexdigest()
                  for name in ('sources.json', 'records.json')},
    }
    (TARGET / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(f'Built {len(sources)} sources and {len(records)} records in {TARGET}')


if __name__ == '__main__':
    main()
