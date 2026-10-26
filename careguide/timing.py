"""Request-local timings. Concurrent/nested spans overlap; never add them as wall time."""
from contextlib import contextmanager
from contextvars import ContextVar
import time

_current = ContextVar('careguide_request_timing', default=None)


class RequestTiming:
    def __init__(self):
        self.started = time.monotonic()
        self.spans = []
        self.first_content_ms = None

    def snapshot(self):
        totals = {}
        for name, start, end in self.spans:
            if end is not None:
                totals[name] = totals.get(name, 0)+(end-start)*1000
        return {'total_ms': round((time.monotonic()-self.started)*1000, 3),
                'first_content_ms': self.first_content_ms,
                'phases_ms': {key: round(value, 3) for key, value in totals.items()},
                'scope': 'request-local; nested/concurrent phases overlap, not additive'}


@contextmanager
def request_timing():
    recorder = RequestTiming()
    token = _current.set(recorder)
    try:
        yield recorder
    finally:
        _current.reset(token)


def start_span(name):
    recorder = _current.get()
    if recorder is None or len(recorder.spans) >= 256:
        return None
    span = [name, time.monotonic(), None]
    recorder.spans.append(span)
    return span


def end_span(span):
    if span is not None:
        span[2] = time.monotonic()


@contextmanager
def phase(name):
    span = start_span(name)
    try:
        yield
    finally:
        end_span(span)


def timing_snapshot():
    recorder = _current.get()
    return recorder.snapshot() if recorder else None


def first_content():
    recorder = _current.get()
    if recorder is not None and recorder.first_content_ms is None:
        recorder.first_content_ms = round((time.monotonic()-recorder.started)*1000, 3)
