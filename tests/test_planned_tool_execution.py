import asyncio
import json

from careguide.domain import AgentType
from careguide.task_context import TaskContext
from tests.test_chat_runtime import ScriptedClient, tool_reply
from tests.test_runtime_failures import medication_engine


def task():
    return TaskContext(task_type='symptom_query', requested_action='sourced_information',
        planned_action='search_care_guidance', task_id='care',
        entities={'query': '就诊前需要准备哪些材料'})


def test_planned_worker_requires_first_tool_and_uses_bound_query_instead_of_brief():
    client = ScriptedClient([tool_reply('search_care_guidance', {'query': '模型改写的检索词'}),
                             {'content': 'done'}])
    engine = medication_engine(client)
    brief = json.dumps({'objective': '就诊前准备', 'allowed_actions': ['search_care_guidance']})
    result = asyncio.run(engine._run_agent(AgentType.TRIAGE, brief, [], task()))
    assert client.calls[0]['tool_choice'] == {
        'type': 'function', 'function': {'name': 'search_care_guidance'}}
    # A genuine not-covered result stops immediately; a successful result may
    # be followed by an optional model turn that must not force a duplicate call.
    if len(client.calls) > 1:
        assert 'tool_choice' not in client.calls[1]
    assert result['tool_calls'][0]['input']['query'] == '就诊前需要准备哪些材料'
    assert result['tool_calls'][0]['input']['query'] != brief


def test_model_ignoring_required_tool_executes_validated_plan_through_dispatcher():
    client = ScriptedClient([{'content': '直接回答而未查询'}])
    context = TaskContext(task_type='appointment', requested_action='sourced_information',
        planned_action='lookup_hospital_guide', task_id='materials', entities={'section': 'materials'})
    result = asyncio.run(medication_engine(client)._run_agent(
        AgentType.APPOINTMENT, 'task brief', [], context))
    assert len(result['tool_calls']) == 1
    call = result['tool_calls'][0]
    assert call['tool_name'] == 'lookup_hospital_guide'
    assert call['output']['status'] == 'needs_clarification'
    assert 'hospital_id' in call['output']['missing_fields']
    assert call['failure']['code'] == 'missing_user_input'
    assert len(client.calls) == 1
