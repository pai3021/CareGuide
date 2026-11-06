"""One deterministic parser for supported single read-only query expressions."""
from datetime import datetime, timedelta, timezone
import re


PREFIX = r'(?:请|麻烦|劳驾)?(?:(?:能否|可以|能)(?:帮我)?|帮我)?(?:(?:我)?想(?:要)?|我需要)?'
SUFFIX = r'(?:吗|呢|可以吗|好吗)?'
VERB = r'(?:查询|查一下|查查|查|看看|看一下|了解一下|了解)'
SPEC = (r'(?:\d+(?:\.\d+)?\s*(?:ml|毫升)\s*[:：/]\s*\d+(?:\.\d+)?\s*(?:mg|g|毫克|克)'
        r'|\d+(?:\.\d+)?\s*(?:mg|g|ml|μg|ug|毫克|克|毫升))')
DATE = r'(?:今天|明天|后天|20\d{2}[-年/]\d{1,2}[-月/]\d{1,2}日?)'


def parse_time_scope(text, at):
    """Shared calendar validation for single queries and doctor/slot templates."""
    args = {}
    for word, value in [('上午', 'morning'), ('下午', 'afternoon'), ('晚上', 'evening')]:
        if word in text:
            args['time_period'] = value
    if matched := re.search(DATE, text):
        value = matched.group()
        offsets = {'今天': 0, '明天': 1, '后天': 2}
        if value in offsets:
            source_at = datetime.fromisoformat(at)
            if source_at.tzinfo is None:
                return None
            day = source_at.astimezone(timezone(timedelta(hours=8))).date()
            args['date'] = (day+timedelta(days=offsets[value])).isoformat()
        else:
            parts = re.fullmatch(r'(20\d{2})[-年/](\d{1,2})[-月/](\d{1,2})日?', value)
            try:
                args['date'] = datetime(*(int(part) for part in parts.groups())).date().isoformat()
            except ValueError:
                return None
    return args


def parse_query(text, local, at):
    """Consume all meaning in a clause; unknown modifiers stay with Planner."""
    body = text.strip().rstrip('。！？?!').strip()
    department = local._department
    doctor = r'(?:医生|医师|大夫)'
    doctor_forms = (
        local._prefix+rf'(?P<department>{department})?'+doctor,
        PREFIX+rf'(?:找|找一下|找找)(?:一位|一个|个)?(?:看)?(?P<department>{department})的?'+doctor+SUFFIX,
        PREFIX+VERB+rf'(?P<department>{department})的?(?:有哪些|有什么)?'+doctor+r'(?:名单|列表)?'+SUFFIX,
        rf'(?P<department>{department})(?:有(?:哪些|什么))'+doctor+r'(?:可以选择|可选)?'+SUFFIX,
    )
    for pattern in doctor_forms:
        if matched := re.fullmatch(pattern, body):
            value = matched['department']
            return 'search_doctors', {'department': value.replace('心内科', '心血管内科')} if value else {}
    slot_pattern = (rf'(?:{PREFIX}{VERB}|{local._prefix})(?P<department>{department})?'
                    rf'(?P<date>{DATE})?(?P<period>上午|下午|晚上)?的?'
                    r'(?:还有(?:哪些|什么|没有)|有没有|有哪些|有什么)?'
                    r'(?:可预约的?时间|能约的时间|可约时间|号源)'+SUFFIX)
    if matched := re.fullmatch(slot_pattern, body):
        args = parse_time_scope(body, at)
        if args is None:
            return None
        if matched['department']:
            args['department'] = matched['department'].replace('心内科', '心血管内科')
        return 'list_available_slots', args
    names = [name for name in local.drugs if name in body]
    if len(names) == 1:
        label_pattern = (rf'(?:{PREFIX}{VERB}|{local._prefix})'+re.escape(names[0])+rf'(?:\s*{SPEC})?的?'
                         r'(?P<section>说明书|禁忌|注意事项|用法|用量)(?:资料)?'+SUFFIX)
        if matched := re.fullmatch(label_pattern, body, re.I):
            entities = local.entities._explicit_entities(text, [])
            args = {key: value for key, value in entities.items()
                    if key in {'drug_name', 'dosage_form', 'specification'}}
            args['drug_name'] = names[0]
            args['section'] = {'禁忌': 'contraindications', '注意事项': 'precautions',
                               '用法': 'dosage', '用量': 'dosage'}.get(matched['section'], 'all')
            return 'lookup_drug_label', args
    return None
