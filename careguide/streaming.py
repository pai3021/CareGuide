"""Request-scoped progress events. Streams observe execution, never authorize it."""
from contextlib import contextmanager
from contextvars import ContextVar
import json
import inspect

_sink = ContextVar('careguide_progress_sink', default=None)


@contextmanager
def progress_events(sink):
    token = _sink.set(sink)
    try:
        yield
    finally:
        _sink.reset(token)


def emit(event, **data):
    sink = _sink.get()
    if sink is not None:
        sink(event, data)


async def emit_answer(event, **data):
    """Answer deltas/reset are ordered and backpressured, unlike snapshots."""
    sink = _sink.get()
    if sink is not None:
        pending = sink(event, data)
        if inspect.isawaitable(pending):
            await pending


def workflow_progress(state, *, stage='executing'):
    # Read-only, bounded presentation fields. No raw model output, tool inputs,
    # patient arguments, confirmation capabilities, or hidden reasoning.
    emit('workflow', stage=stage, workflow={
        'id': state.id, 'version': state.version, 'revision': state.revision,
        'status': state.status, 'tasks': [
            {'id': task.spec.id, 'revision': task.revision,
             'label': task.spec.fragment, 'status': task.status.value,
             'agent': task.spec.agent.value, 'depends_on': task.spec.depends_on}
            for task in state.tasks.values()]})


def encode_event(event, data, sequence):
    return f'id: {sequence}\nevent: {event}\ndata: {json.dumps(data, ensure_ascii=False, allow_nan=False)}\n\n'
