import asyncio
import json

import httpx
from openai import AsyncOpenAI
import pytest

from careguide.evaluation.faithfulness import quantities, quantity_risks, judge_answer
from scripts.evaluate_answer_faithfulness import evaluate, probe_case, CASES
from scripts.evaluate_intent_ablation import score
from careguide.orchestration.answering import validate_answer


def test_quantities_convert_units_and_reject_invented_values_with_valid_citations():
    sources = [{'id': 'S1', 'task_id': 'drug', 'text': '某片剂规格为0.2g。'}]
    validate_answer('资料规格为200mg。[S1]', sources)
    with pytest.raises(ValueError, match='quantitative'):
        validate_answer('资料规格为0.4g。[S1]', sources)
    assert quantities('规格为0.2g') == quantities('规格为200毫克')
    validate_answer('某药0.2g的资料请见来源。[S1]', [dict(id='S1', task_id='drug',
        text='该药禁忌见原文。', metadata_text='某药0.2g说明书')])
    # A user-derived display title must not authorize a quantity absent from evidence.
    with pytest.raises(ValueError, match='quantitative'):
        validate_answer('该药0.4g。[S1]', [dict(id='S1', task_id='drug', text='该药规格0.2g。', title='能用0.4g吗？')])
    assert quantity_risks('300mg', [{'text': '200～400mg'}])['status'] == 'abstained'


def test_labeled_probes_do_not_claim_semantic_validation_from_citation_membership():
    report = asyncio.run(evaluate('offline'))
    assert all(row['known_good']['contract_accepted'] for row in report['rows'])
    assert not report['rows'][0]['known_bad']['contract_accepted']
    assert report['rows'][1]['known_bad']['contract_accepted']  # negation needs semantic review
    assert report['usage']['calls'] == 0


def test_intent_metrics_score_goals_arguments_and_forbidden_actions_separately():
    from careguide.orchestration.contracts import PlanProposal
    plan = PlanProposal(goal='查医生', tasks=[dict(id='d', fragment='查医生', action='search_doctors',
        agent='appointment', completion='items', arguments={'department': '消化内科'})])
    case = {'actions': ['search_doctors'], 'arguments': {'search_doctors': {'department': '消化内科'}}}
    assert score(case, plan)['complete'] is True
    case['arguments']['search_doctors']['department'] = '心血管内科'
    assert score(case, plan)['action_f1'] == 1 and score(case, plan)['argument_match'] is False
    case['forbidden_actions'] = ['search_doctors']
    assert score(case, plan)['forbidden_actions'] == ['search_doctors']


def test_judge_must_quote_actual_answer_and_actual_evidence():
    def handler(request):
        result = {'claims': [{'answer_quote': '可以饮酒', 'verdict': 'supported',
                              'source_id': 'S1', 'source_quote': '资料说可以饮酒'}]}
        return httpx.Response(200, json={'id': 'offline', 'object': 'chat.completion', 'created': 1,
            'model': 'script', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': json.dumps(result)}}]})
    async def run():
        async with AsyncOpenAI(api_key='offline', max_retries=0,
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))) as client:
            with pytest.raises(ValueError, match='evidence quotation'):
                await judge_answer(client, 'script', '可以饮酒', [{'id': 'S1', 'text': '不可饮酒'}])
    asyncio.run(run())
