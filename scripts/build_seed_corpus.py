"""Build developer-verified factual source notes, never synthetic clinical content.

Uses previously fetched official source cache. PDF passages were visually checked
against PDF pages 32-36 (printed pages 33-37); no OCR text is silently accepted.
"""
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

from scripts.inspect_source_pages import extract

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'data' / 'source_cache'
TARGET = ROOT / 'data' / 'medical_v1'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    if (TARGET / 'manifest.json').exists():
        raise SystemExit('Seed corpus exists; refusing implicit overwrite. Review a new corpus version explicitly.')
    verified_at = date(2026, 9, 17)
    sources = []
    records = []

    def source(key, url, publisher, file, anchors, *, published=None, region='CN',
               scope='general_information', valid_days=180, locator='正文', pdf=False):
        raw = (CACHE / file).read_bytes()
        if pdf:
            assert raw.startswith(b'%PDF')
            method = 'developer_visual_pdf_verification'
        else:
            text = extract(raw)
            for anchor in anchors:
                if anchor not in text:
                    raise ValueError(f'Source anchor missing: {key}: {anchor}')
            method = 'developer_html_source_verification'
        sources.append({
            'id': key, 'title': publisher + ' · ' + scope, 'url': url,
            'publisher': publisher, 'jurisdiction': region, 'scope': scope,
            'published_at': published, 'verified_at': verified_at.isoformat(),
            'valid_until': (verified_at + timedelta(days=valid_days)).isoformat(),
            'review_status': 'developer_source_verified', 'clinical_reviewed': False,
            'verification_method': method, 'locator': locator,
            'retrieved_sha256': sha(raw), 'evidence_anchors': anchors,
        })
        return key

    def note(key, source_id, kind, title, content, keywords, **fields):
        records.append({'id': key, 'source_id': source_id, 'kind': kind,
                        'title': title, 'content': content, 'keywords': keywords,
                        'representation': 'curated_factual_summary', **fields})

    nmpa = source('nmpa-otc-labels',
                  'https://www.nmpa.gov.cn/directory/web/nmpa/images/1672728602691034318.pdf',
                  '国家药品监督管理局', 'nmpa-otc.pdf',
                  ['精氨酸布洛芬片', '富马酸酮替芬鼻喷雾剂'], pdf=True,
                  scope='OTC说明书范本；不绑定商品名或生产企业',
                  locator='PDF页32-36；印刷页33-37')
    for specification, frequency, page in [('0.2g', '一日3次', 33), ('0.4g', '一日2次', 35)]:
        product = dict(drug_name='精氨酸布洛芬片', dosage_form='片剂',
                       specification=specification, label_type='regulatory_template',
                       locator=f'PDF页{page}；印刷页{page+1}', section='dosage')
        prefix = 'ibuprofen-arginine-' + specification.replace('.', '-')
        note(prefix+'-dosage', nmpa, 'drug_label', f'精氨酸布洛芬片 {specification} 说明书用法用量',
             f'该规格以布洛芬计。说明书列出口服，成人和12岁以上患者一次{specification}（1片），{frequency}。这是说明书范本信息，不是针对个人的用药方案。',
             ['精氨酸布洛芬片', '用法用量', '说明书', specification], **product)
        product['section'] = 'contraindications'
        note(prefix+'-contraindications', nmpa, 'drug_label', f'精氨酸布洛芬片 {specification} 禁忌',
             '范本禁忌包括其他非甾体抗炎药过敏、孕期或哺乳期、胃或十二指肠溃疡、支气管哮喘、肾脏疾病、血液凝固及血细胞生成障碍。该清单仅适用于本范本，个人情况需由医师或药师核实。',
             ['精氨酸布洛芬片', '禁忌', '过敏', '孕妇', '哺乳'], **product)
        product['section'] = 'precautions'
        note(prefix+'-precautions', nmpa, 'drug_label', f'精氨酸布洛芬片 {specification} 注意事项',
             '范本限制对症止痛连续使用不超过5天、解热不超过3天；未缓解需咨询医师或药师。提示避免其他含解热镇痛药的药品及酒精，正在使用其他药品时先咨询医师或药师。完整注意事项应查看原文。',
             ['精氨酸布洛芬片', '注意事项', '感冒药', '酒精'], **product)
        product['section'] = 'interactions'
        note(prefix+'-heparin', nmpa, 'drug_interaction', f'精氨酸布洛芬片 {specification} 与肝素',
             '该说明书范本记载：与肝素等抗凝药同用可能延长凝血酶原时间、增加出血倾向。需要医师或药师核对，工具不作个人能否合用的决定。',
             ['精氨酸布洛芬片', '肝素', '相互作用', '抗凝药'],
             other_drug='肝素', **product)
        note(prefix+'-methotrexate', nmpa, 'drug_interaction', f'精氨酸布洛芬片 {specification} 与甲氨蝶呤',
             '该说明书范本列出与甲氨蝶呤同用会使其血药浓度增高，提示不宜同用。这里只提供已记录的说明书信息，具体用药由医师或药师核实。',
             ['精氨酸布洛芬片', '甲氨蝶呤', '相互作用'], other_drug='甲氨蝶呤', **product)
    nasal = dict(drug_name='富马酸酮替芬鼻喷雾剂', dosage_form='鼻喷雾剂',
                 specification='15ml:16.7mg', label_type='regulatory_template',
                 locator='PDF页32；印刷页33')
    note('ketotifen-nasal-dosage', nmpa, 'drug_label', '富马酸酮替芬鼻喷雾剂 说明书用法用量',
         '范本规格为15毫升含16.7毫克（以酮替芬计），每喷0.15毫克。说明书列出鼻腔喷雾一次1至2喷，一日1至3次。这是原文用法信息，不适用于个人剂量决定。',
         ['富马酸酮替芬鼻喷雾剂', '说明书', '用法用量'], section='dosage', **nasal)
    note('ketotifen-nasal-contraindications', nmpa, 'drug_label', '富马酸酮替芬鼻喷雾剂 禁忌与注意事项',
         '范本禁忌列出服用降糖药的糖尿病患者；注意事项要求孕妇慎用、哺乳期在医师指导下使用，对本品过敏者禁用，并提示驾驶及操作精密仪器等工作风险。完整说明应查看原文。',
         ['富马酸酮替芬鼻喷雾剂', '禁忌', '注意事项'], section='contraindications', **nasal)
    note('ketotifen-nasal-alcohol', nmpa, 'drug_interaction', '富马酸酮替芬鼻喷雾剂 与酒精',
         '说明书范本记载与中枢神经抑制剂、酒精或抗组胺药并用可能增强镇静作用。当前只收录酒精这一明确条目，不是完整相互作用数据库。',
         ['富马酸酮替芬鼻喷雾剂', '酒精', '相互作用'], section='interactions', other_drug='酒精', **nasal)

    pkuih = source('pkuih-appointment', 'https://www.pkuih.edu.cn/notice/details_940_1487.html',
                   '北京大学国际医院', 'pkuih-appointment.html',
                   ['2025/11/25', '放号时间', '预约识别码', '取消订单'],
                   published='2025-11-25', scope='本院门诊预约挂号', valid_days=30)
    hospital = dict(hospital_id='pkuih', hospital_name='北京大学国际医院', campus='本院')
    note('pkuih-channels', pkuih, 'hospital_guide', '北京大学国际医院 挂号渠道',
         '官网指南列出微信公众号、114、自助机、电话、支付宝和官网预约渠道，门诊实行全预约制。此处提供流程资料，不提供实时号源。',
         ['挂号', '预约', '渠道', '公众号', '114'], section='channels', **hospital)
    note('pkuih-process', pkuih, 'hospital_guide', '北京大学国际医院 微信挂号流程',
         '指南流程是先注册并实名建档、添加就诊人，再在预约模块选择科室、医生日期和时段、就诊人以及支付选项。操作须由用户在官方渠道完成。',
         ['挂号', '流程', '微信', '初诊', '建档'], section='process', **hospital)
    note('pkuih-materials', pkuih, 'hospital_guide', '北京大学国际医院 到院取号材料',
         '该院指南要求就诊当日携带本人有效证件及预约识别码，在诊区自助机验证预约信息并取号。预约登记与证件信息需一致。',
         ['挂号', '材料', '身份证', '取号', '预约识别码'], section='materials', **hospital)
    note('pkuih-cancellation', pkuih, 'hospital_guide', '北京大学国际医院 取消预约流程',
         '微信渠道可在就医服务中的挂号记录打开号源信息，选择取消订单。取消时限和爽约规则请核对官方指南及当前渠道提示；本工具不会代为取消。',
         ['取消', '退号', '预约', '挂号记录'], section='cancellation', **hospital)

    pkufh = source('pkufh-daxing-appointment', 'https://www.pkufh.com/Html/News/Articles/61679.html',
                   '北京大学第一医院', 'pkufh-daxing-appointment.html',
                   ['2025-05-06', '健康大兴APP', '10分钟内', '同一个就诊单元'],
                   published='2025-05-06', scope='大兴院区挂号指南', valid_days=30)
    daxing = dict(hospital_id='pkufh', hospital_name='北京大学第一医院', campus='大兴院区')
    note('pkufh-daxing-channels', pkufh, 'hospital_guide', '北京大学第一医院 大兴院区挂号渠道',
         '大兴院区指南列出微信服务号、支付宝生活号、抖音小程序、114、健康大兴APP等渠道；自助机限当日挂号。不能将这份指南直接套用其他院区。',
         ['挂号', '渠道', '大兴', '公众号', '114'], section='channels', **daxing)
    note('pkufh-daxing-process', pkufh, 'hospital_guide', '北京大学第一医院 大兴院区挂号流程',
         '指南要求绑定或添加就诊卡，阅读预约须知，选择院区、科室、日期、医师和就诊人，确认后缴费；页面列出的缴费期限为预约后10分钟内。规则可能更新，须核对官方渠道。',
         ['挂号', '流程', '就诊卡', '缴费', '10分钟'], section='process', **daxing)
    note('pkufh-daxing-followup', pkufh, 'hospital_guide', '北京大学第一医院 大兴院区复诊挂号',
         '该院区指南区分就诊单元：同一个就诊单元看结果不需重复挂号，复诊报到即可；不同单元需重新挂号。页面以上午、下午分别作为一个单元。适用情况需向院方核实。',
         ['复诊', '结果', '重新挂号', '就诊单元'], section='followup', **daxing)

    for key, department, filename, url, anchors, content in [
        ('pkuih-neurology', '神经内科', 'pkuih-neurology.html',
         'https://www.pkuih.edu.cn/department/details_931_3308.html',
         ['头晕头痛', '周围神经病变'],
         '官网科室介绍列出神经内科诊疗范围包含头晕头痛、脑血管病、癫痫、认知障碍及周围神经病变等。这是科室服务范围，不是对当前症状的诊断或紧急程度判断。'),
        ('pkuih-gastroenterology', '消化内科', 'pkuih-gastroenterology.html',
         'https://www.pkuih.edu.cn/department/details_931_3301.html',
         ['内镜中心', '炎症性肠病'],
         '官网介绍该科包括病房、门诊及内镜中心，开展消化道疾病内镜诊疗，并设炎症性肠病等专业门诊。资料仅描述科室服务范围，不能据此判断个人病因。'),
    ]:
        sid = source(key, url, '北京大学国际医院', filename, anchors,
                     scope=department+'科室介绍', valid_days=90)
        note(key+'-scope', sid, 'department', department+' 服务范围', content,
             [department, *anchors], department=department, **hospital)

    nhs = source('nhs-headaches', 'https://www.nhs.uk/symptoms/headaches/',
                 '英国 NHS', 'nhs-headaches.html',
                 ['Keeping a headache diary', 'it started suddenly and is extremely painful'],
                 region='GB', scope='英国公众头痛就医信息；中文摘要非官方译文')
    note('headache-care-warning', nhs, 'care_guidance', '头痛 就医风险信息',
         'NHS 公众资料将突然发生且极度疼痛的头痛、伴麻木无力或意识混乱等列为需要立即求助的情况。中文摘要只用于资料查询，不提供个人风险排除；急救联系方式应以所在地为准。',
         ['头痛', '突然', '剧烈', '无力', '意识', '就医'])
    note('headache-diary', nhs, 'care_guidance', '头痛 记录与就医准备',
         'NHS 资料提到记录头痛日记有助于了解诱因；反复头痛或自行处理后未改善、加重时应寻求医护帮助。这里提供一般信息，不判断个人病因。',
         ['头痛', '日记', '记录', '反复', '就医准备'])

    TARGET.mkdir(parents=True, exist_ok=True)
    for name, value in [('sources.json', sources), ('records.json', records)]:
        (TARGET / name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')
    manifest = {'version': 'careguide-medical-corpus-v1',
                'verified_at': verified_at.isoformat(), 'clinical_reviewed': False,
                'source_count': len(sources), 'record_count': len(records),
                'files': {name: sha((TARGET/name).read_bytes()) for name in ['sources.json','records.json']}}
    (TARGET/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(f'Built {len(sources)} official sources, {len(records)} factual notes')


if __name__ == '__main__':
    main()
