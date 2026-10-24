"""Verify the selected framework and SQLite adapter in the actual environment."""
import asyncio
import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt


class WaitingState(TypedDict, total=False):
    completed_result: str
    answer: dict


def waiting_graph(saver, calls):
    async def read(state):
        calls.append('read')
        return {'completed_result': 'source-already-read'}

    def wait_for_user(state):
        answer = interrupt({'task_id': 'drug', 'missing_fields': ['specification']})
        return {'answer': answer}

    builder = StateGraph(WaitingState)
    builder.add_node('read', read)
    builder.add_node('wait_for_user', wait_for_user)
    builder.add_edge(START, 'read')
    builder.add_edge('read', 'wait_for_user')
    builder.add_edge('wait_for_user', END)
    return builder.compile(checkpointer=saver)


def test_sqlite_wait_survives_new_graph_and_connection_without_replaying_read(tmp_path):
    path = str(tmp_path/'checkpoints.sqlite3')
    calls = []
    config = {'configurable': {'thread_id': 'synthetic-workflow'}}

    async def run():
        async with AsyncSqliteSaver.from_conn_string(path) as saver:
            first = await waiting_graph(saver, calls).ainvoke({}, config)
            assert first['__interrupt__'][0].value['task_id'] == 'drug'
            assert calls == ['read']
        # Reopen both objects: no in-memory graph/checkpointer state is reused.
        async with AsyncSqliteSaver.from_conn_string(path) as saver:
            resumed = await waiting_graph(saver, calls).ainvoke(
                Command(resume={'specification': '0.2g'}), config)
            assert resumed['completed_result'] == 'source-already-read'
            assert resumed['answer']['specification'] == '0.2g'
            assert calls == ['read']
    asyncio.run(run())


class DispatchState(TypedDict, total=False):
    items: list[int]
    value: int
    results: Annotated[list[int], operator.add]


def test_dynamic_send_fanout_executes_independent_nodes_concurrently():
    async def run():
        started = []
        both_started = asyncio.Event()

        def dispatch(state):
            return [Send('worker', {'value': value}) for value in state['items']]

        async def worker(state):
            started.append(state['value'])
            if len(started) == 2:
                both_started.set()
            await asyncio.wait_for(both_started.wait(), timeout=2)
            return {'results': [state['value']*2]}

        builder = StateGraph(DispatchState)
        builder.add_node('worker', worker)
        builder.add_conditional_edges(START, dispatch, ['worker'])
        builder.add_edge('worker', END)
        result = await builder.compile().ainvoke({'items': [2, 3], 'results': []})
        assert sorted(result['results']) == [4, 6]
    asyncio.run(run())
