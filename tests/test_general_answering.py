import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from careguide.orchestration.decisions import RequestDecision
from careguide.orchestration.general_answering import direct_allowed, local_general_request, NOTICE
from careguide.orchestration.understanding_planner import validate_decision, build_context
from careguide.streaming import progress_events
from tests.test_answer_streaming import prepared, sdk, frame


QUESTION = '帮我整理一下就诊前需要准备哪些材料，用列表回答。'


@pytest.mark.parametrize('text', [QUESTION, '就诊前要准备什么材料？', '你好'])
def test_narrow_general_fast_path(text):
    assert local_general_request(text)


@pytest.mark.parametrize('text', [
    '帮我预约心内科', '中大医院本部就诊前需要准备哪些材料', '查布洛芬说明书',
    '医院官方要求准备什么材料', '查一下明天有哪些医生', '帮我取消预约',
    '我是不是得了肺炎', '我现在呼吸困难，准备什么材料',
    '一般就诊准备清单，并帮我预约医生',
])
def test_source_business_and_personal_decisions_cannot_take_direct_route(text):
    assert not direct_allowed(text)
    assert not local_general_request(text)
    context = build_context(text, 'm', '2026-10-26T10:00:00+00:00')
    with pytest.raises(PermissionError):
        validate_decision(RequestDecision(kind='direct'), context)


def test_direct_stream_preserves_existing_workflow_and_does_not_reset_its_budget(tmp_path):
    async def run():
        runtime, state, old_budget = await prepared(tmp_path)
        for _ in range(20):
            ticket = old_budget.reserve('model')
            old_budget.finish(ticket, 0)
        old_state = runtime.store.get(state.id, 'u', 'c').model_dump()
        old_used = old_budget.summary()['used']
        runtime.worker.run = AsyncMock(side_effect=AssertionError('No worker for direct answer'))
        runtime.planner.decide = AsyncMock(side_effect=AssertionError('Narrow route needs no planner'))
        calls, events = [], []
        async with sdk([frame('可按下面的清单准备：\n\n- 有效证件\n- 已有检查报告'), frame(finish='stop')], calls) as client:
            runtime.engine.client = client
            with progress_events(lambda event, data: events.append((event, data))):
                result = await runtime.process(QUESTION, 'u', 'c')
        assert result['workflow_status'] == 'completed'
        assert result['execution_mode'] == 'llm_direct'
        assert result['workflow'] is None
        assert not result['knowledge_used'] and not result['grounded']
        assert not result['tools_used'] and not result['evidence']
        assert result['answer']['sources'] == []
        assert NOTICE in result['answer']['body']
        assert len(calls) == 1 and calls[0]['stream']
        assert 'tools' not in calls[0]
        assert any(e == 'delta' for e, _ in events)
        assert old_budget.summary()['used'] == old_used
        assert runtime.store.latest('u', 'c').id == state.id
        assert runtime.store.get(state.id, 'u', 'c').model_dump() == old_state
        trace = runtime.traces.get(result['request_id'])
        assert trace['status'] == 'completed'
        assert trace['execution']['tool_calls'] == []
    asyncio.run(run())


@pytest.mark.parametrize('text', ['根据官方医院规定，已查询到了资料。', '资料如下。[S1]', '详见 https://example.com'])
def test_direct_generation_rejects_fake_sources_and_resets_provisional_content(tmp_path, text):
    async def run():
        runtime, _, _ = await prepared(tmp_path)
        calls, events = [], []
        async with sdk([frame(text), frame(finish='stop')], calls) as client:
            runtime.engine.client = client
            with progress_events(lambda event, data: events.append((event, data))):
                result = await runtime.process(QUESTION, 'u', 'c')
        assert result['workflow_status'] == 'failed'
        assert result['errors'][0]['code'] == 'model_format_error'
        assert text not in result['response']
        assert any(e == 'answer_reset' for e, _ in events)
    asyncio.run(run())


def test_planner_can_choose_direct_for_general_explanation(tmp_path):
    async def run():
        runtime, state, _ = await prepared(tmp_path)
        runtime.planner.decide = AsyncMock(return_value=(RequestDecision(kind='direct'), []))
        calls = []
        async with sdk([frame('症状日记用于整理症状发生的时间和变化。'), frame(finish='stop')], calls) as client:
            runtime.engine.client = client
            result = await runtime.process('解释什么是症状日记', 'u', 'c')
        assert result['execution_mode'] == 'llm_direct'
        assert result['workflow_status'] == 'completed'
        assert runtime.store.latest('u', 'c').id == state.id
    asyncio.run(run())
