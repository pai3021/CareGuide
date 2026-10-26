"""General-information answers: no tool claims, citations or workflow mutations."""
from __future__ import annotations

import json
import re
import uuid

from careguide.errors import ErrorCode, Failure, model_failure
from careguide.intent_recognizer import IntentRecognizer
from careguide.qic_intent_recognizer import QICIntentRecognizer
from careguide.llm_utils import stream_answer_completion
from careguide.orchestration.budget import BudgetExhausted, use_budget
from careguide.streaming import emit, emit_answer
from careguide.timing import phase


# Conservative guardrails complement the model's semantic decision. They do not
# establish factual correctness; concrete source/business requests use tools.
TOOL_REQUIRED = re.compile(
    r'预约|挂号|号源|医生名单|有哪些医生|医生排班|取消|改期|查订单|查记录|费用|价格|报销|医保政策|'
    r'官方|来源|引用|查证|核实|说明书|药品|药物|药剂|剂量|服用|布洛芬|阿司匹林|'
    r'院区|本部|本院|该院|这家医院|那家医院|中大医院|协和医院|北大医院|'
    r'最新|实时|现在开放|营业时间|具体医院|医院规定|医院要求|医院地址|就诊卡|'
    r'我(?:是否|是不是|得了|患有)|帮我诊断|给我诊断|开药|开处方|检查报告解读|'
    r'呼吸困难|胸痛|昏迷|大出血|自杀|自伤|'
    r'我(?:现在|最近|一直|这几天)?(?:头痛|发烧|发热|咳嗽|腹痛|呕吐)')
NOTICE = '一般信息：由模型生成，未查询知识库或医院系统；具体要求请以就诊医院为准。'


def direct_allowed(message, *, corpus=None):
    if TOOL_REQUIRED.search(message):
        return False
    if IntentRecognizer._pattern(message).get('urgent') or QICIntentRecognizer._boundary(message):
        return False
    if corpus is not None:
        for record in corpus.records:
            if record.hospital_name and record.hospital_name in message:
                return False
    return True


def local_general_request(message, *, corpus=None):
    """A narrow fast path; unknown requests still go through the planner."""
    if not direct_allowed(message, corpus=corpus):
        return False
    text = re.sub(r'[，,。？！?!\s]', '', message)
    if text in {'你好', '您好', '谢谢', '谢谢你'}:
        return True
    return bool(re.fullmatch(
        r'(?:请|帮我|麻烦)?(?:整理(?:一下)?|列(?:一下|出)?|介绍(?:一下)?|告诉我)?'
        r'(?:普通门诊|一般性的?|通用的?)?(?:初诊|复诊|就诊|看病|门诊|去医院)前'
        r'(?:通常|一般)?(?:(?:需要|要|应该|应)?(?:准备|携带|带|记录)(?:哪些|什么|的)?'
        r'(?:材料|资料|东西|信息|内容)|(?:需要|要)什么准备)'
        r'(?:(?:请)?(?:用|以)(?:列表|清单|Markdown|表格)(?:形式)?回答)?', text))


def validate_general_answer(text):
    if not text.strip() or re.search(
            r'\[S\d+\]|https?://|已(?:经)?(?:查询|检索|预约|挂号|提交|取消成功)|'
            r'查询结果(?:显示|如下)|根据.{0,20}(?:官方|医院规定)|<\|', text):
        raise ValueError('Ungrounded source or action claim in general answer')


SYSTEM = '''你是 CareGuide，就用户的问题直接提供清楚、有帮助的一般信息。
本次没有调用知识库、检索或医院业务工具。不得编造来源、引用编号、链接、医院具体规定、实时状态或已执行操作。
适合回答通用就诊准备、普通概念说明、病情表达整理、问候。按用户要求使用列表或其他 Markdown 格式。
通用就诊材料问题可直接整理证件、已有病历和检查资料、当前用药及过敏记录、症状经过等一般准备项；不要强求用户提供医院。
不要自行安排检查、要求所有人空腹停药或作个人诊断、处方、剂量调整。具体医疗安排留给就诊医生。
上下文只用于理解指代，历史中的医院要求和操作状态未经本次核实，不能当作已核实事实。
如果需要个人诊疗决策或当前医院规定，应简要说明需要专业判断或核实，不凭空给出结论。
直接回答，避免模板化开场；不要声称正在查询工具或替用户执行任何业务。'''


async def generate_general_answer(client, model, message, history, budget):
    identity, offset = uuid.uuid4().hex, 0
    async def delta(text):
        nonlocal offset
        await emit_answer('delta', answer_id=identity, offset=offset, text=text, provisional=True)
        offset += len(text)
    if client is None:
        return None, {'status': 'failed', 'mode': 'llm_direct',
                      'failure': Failure(ErrorCode.MODEL_UNAVAILABLE, 'direct_answering').payload()}
    recent = [{'role': item['role'], 'content': item['content'][:2000]}
              for item in history[-4:] if item.get('role') in {'user', 'assistant'}
              and isinstance(item.get('content'), str)]
    emit('stage', stage='answering', message='正在生成一般信息回答')
    try:
        with phase('answering'):
            async with budget.call('model', 20) as ticket:
                with use_budget(None):
                    text = await stream_answer_completion(client, model=model,
                        messages=[{'role': 'system', 'content': SYSTEM},
                                  {'role': 'user', 'content': json.dumps(
                                      {'recent_conversation': recent, 'request': message}, ensure_ascii=False)}],
                        on_delta=delta, on_usage=ticket.record_response, max_tokens=1536)
        validate_general_answer(text)
        return text+'\n\n'+NOTICE, {'status': 'generated', 'mode': 'llm_direct',
            'answer_id': identity, 'sources': [], 'knowledge_used': False}
    except Exception as exc:
        await emit_answer('answer_reset', answer_id=identity, reason='direct_answer_failed')
        failure = (Failure(ErrorCode.BUDGET_EXHAUSTED, 'direct_answering').payload()
                   if isinstance(exc, BudgetExhausted) else
                   Failure(ErrorCode.MODEL_FORMAT, 'direct_answering').payload()
                   if isinstance(exc, ValueError) else model_failure(exc, 'direct_answering'))
        return None, {'status': 'failed', 'mode': 'llm_direct', 'failure': failure}
