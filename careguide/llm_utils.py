"""Shared helpers for OpenAI Chat Completions-compatible providers.

The production runtime uses the official ``openai`` async client.  A small
legacy fallback is kept in ``create_chat_completion`` so the deterministic
offline tests can continue to inject their scripted ``messages.create``
clients without making network requests.
"""

import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

from openai import AsyncOpenAI

from careguide.orchestration.budget import budgeted_call


@dataclass
class TokenUsageRecorder:
    """Accumulate provider-reported chat token usage for one evaluation stage."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls_without_usage: int = 0
    by_model: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def record(self, response: Any, model: str) -> None:
        self.calls += 1
        usage = _field(response, "usage")
        if usage is None:
            self.calls_without_usage += 1
            return
        prompt = int(_field(usage, "prompt_tokens", 0) or 0)
        completion = int(_field(usage, "completion_tokens", 0) or 0)
        total = int(_field(usage, "total_tokens", prompt + completion) or 0)
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += total
        model_usage = self.by_model.setdefault(model, {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        })
        model_usage["calls"] += 1
        model_usage["prompt_tokens"] += prompt
        model_usage["completion_tokens"] += completion
        model_usage["total_tokens"] += total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "calls_without_usage": self.calls_without_usage,
            "by_model": {key: dict(value) for key, value in self.by_model.items()},
        }


_ACTIVE_TOKEN_RECORDER: ContextVar[Optional[TokenUsageRecorder]] = ContextVar(
    "careguide_active_token_recorder", default=None
)


@contextmanager
def track_token_usage() -> Iterator[TokenUsageRecorder]:
    """Track chat completion usage inside the current async context."""
    recorder = TokenUsageRecorder()
    token = _ACTIVE_TOKEN_RECORDER.set(recorder)
    try:
        yield recorder
    finally:
        _ACTIVE_TOKEN_RECORDER.reset(token)


def _record_token_usage(response: Any, model: str) -> None:
    recorder = _ACTIVE_TOKEN_RECORDER.get()
    if recorder is not None:
        recorder.record(response, model)


def create_llm_client(api_key: str, base_url: Optional[str] = None) -> AsyncOpenAI:
    """Build one async OpenAI client for an official or compatible endpoint."""
    # Retries belong to the workflow budget, not an invisible SDK retry loop.
    kwargs: Dict[str, Any] = {"api_key": api_key, "max_retries": 0}
    if base_url:
        kwargs["base_url"] = base_url
    return AsyncOpenAI(**kwargs)


def is_openai_chat_client(client: Any) -> bool:
    """Return whether ``client`` exposes ``chat.completions.create``."""
    chat = getattr(client, "chat", None)
    completions = getattr(chat, "completions", None)
    return callable(getattr(completions, "create", None))


async def create_chat_completion(
    client: Any,
    *,
    model: str,
    messages: Sequence[Dict[str, Any]],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    system: Optional[str] = None,
    tools: Optional[Sequence[Dict[str, Any]]] = None,
    tool_choice: Optional[Dict[str, Any]] = None,
    response_format: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
) -> Any:
    """Call an OpenAI-compatible chat endpoint.

    ``tools`` accepts the internal CareGuide schema
    ``{name, description, input_schema}`` and converts it to the OpenAI
    function-tool schema.  The fallback branch only exists for injected test
    doubles that still implement the former Anthropic-shaped ``messages.create``
    method.
    """
    # Server-owned memory provenance is available to the planner, but is not
    # an OpenAI/legacy message field. Its readable content remains in context.
    messages = [{key: value for key, value in message.items() if key != 'durable_preferences'}
                for message in messages]
    if is_openai_chat_client(client):
        # MiMo enables thinking by default. Restore the source recognizer's
        # disabled setting for bounded JSON/tool calls throughout the runtime.
        # Other providers keep their original request schema.
        if extra_body is None and model.casefold().startswith('mimo-'):
            extra_body = {'thinking': {'type': 'disabled'}}
        payload_messages = list(messages)
        if system:
            if payload_messages and payload_messages[0].get("role") == "system":
                payload_messages[0] = {
                    "role": "system",
                    "content": system,
                }
            else:
                payload_messages.insert(0, {"role": "system", "content": system})

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": payload_messages,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature
        if tools:
            kwargs["tools"] = [_to_openai_tool(tool) for tool in tools]
            if tool_choice is not None:
                kwargs['tool_choice'] = tool_choice
        if response_format is not None:
            kwargs["response_format"] = response_format
        if extra_body is not None:
            kwargs["extra_body"] = extra_body
        async with budgeted_call('model') as ticket:
            response = await client.chat.completions.create(**kwargs)
            if ticket is not None:
                ticket.record_response(response)
        _record_token_usage(response, model)
        return response

    # Compatibility path for deterministic tests and externally injected
    # clients that still implement the former Anthropic-shaped interface.
    messages_api = getattr(client, "messages", None)
    create = getattr(messages_api, "create", None)
    if not callable(create):
        raise TypeError("LLM client must expose chat.completions.create")

    kwargs = {
        "model": model,
        "messages": list(messages),
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if temperature is not None:
        kwargs["temperature"] = temperature
    if system:
        kwargs["system"] = system
    if tools:
        kwargs["tools"] = [_to_legacy_tool(tool) for tool in tools]
    async with budgeted_call('model') as ticket:
        response = await create(**kwargs)
        if ticket is not None:
            ticket.record_response(response)
    _record_token_usage(response, model)
    return response


async def stream_answer_completion(client, *, model, messages, on_delta, on_usage=None, max_tokens=2048):
    """Stream only public answer content, with one quota for the entire stream.

    Tool calls and reasoning are never forwarded. A clean stop is required;
    callers must discard the provisional answer on any exception.
    """
    if not is_openai_chat_client(client):
        raise TypeError('Answer streaming requires a chat-completions client')
    kwargs = dict(model=model, messages=messages, max_tokens=max_tokens,
                  stream=True, stream_options={'include_usage': True})
    if model.casefold().startswith('mimo-'):
        kwargs['extra_body'] = {'thinking': {'type': 'disabled'}}
    stream, chunks, finished, usage, length = None, [], False, None, 0
    try:
        async with budgeted_call('model') as ticket:
            stream = await client.chat.completions.create(**kwargs)
            async for chunk in stream:
                if _field(chunk, 'usage') is not None:
                    usage = chunk
                    if ticket is not None:
                        ticket.record_response(chunk)
                    if on_usage is not None:
                        on_usage(chunk)
                for choice in _field(chunk, 'choices', []) or []:
                    if _field(choice, 'index', 0) != 0:
                        raise ValueError('Unexpected answer choice')
                    delta = _field(choice, 'delta')
                    if _field(delta, 'tool_calls') or _field(delta, 'function_call'):
                        raise ValueError('Answer generation cannot call tools')
                    text = _field(delta, 'content')
                    if text:
                        if finished or not isinstance(text, str):
                            raise ValueError('Invalid answer chunk')
                        chunks.append(text)
                        length += len(text)
                        if length > 12000:
                            raise ValueError('Answer exceeds output limit')
                        await on_delta(text)
                    reason = _field(choice, 'finish_reason')
                    if reason is not None:
                        if reason != 'stop':
                            raise ValueError('Incomplete or rejected answer')
                        finished = True
            if not finished:
                raise ValueError('Answer stream ended without a clean stop')
        _record_token_usage(usage or {}, model)
        return ''.join(chunks)
    finally:
        if stream is not None:
            await stream.close()


def extract_text_content(value: Any) -> str:
    """Extract text from an OpenAI response or an older content-block list."""
    message = _completion_message(value)
    if message is not None:
        return _extract_text_value(_field(message, "content"))

    # Legacy response objects expose ``response.content`` directly.
    response_content = _field(value, "content", _MISSING)
    if response_content is not _MISSING:
        return _extract_text_value(response_content)
    return _extract_text_value(value)


def extract_tool_calls(value: Any) -> List[Dict[str, Any]]:
    """Normalize OpenAI function calls and legacy tool-use blocks."""
    message = _completion_message(value)
    if message is not None:
        calls = _field(message, "tool_calls") or []
        normalized: List[Dict[str, Any]] = []
        for call in calls:
            function = _field(call, "function")
            raw_arguments = _field(function, "arguments")
            arguments, parse_error = _parse_arguments(raw_arguments)
            item = {
                "type": "tool_use",
                "id": str(_field(call, "id", "")),
                "name": str(_field(function, "name", "")),
                "input": arguments,
            }
            if parse_error:
                item['parse_error'] = parse_error
                item['_raw_arguments'] = raw_arguments if isinstance(raw_arguments, str) else ''
            normalized.append(item)
        return normalized

    content = _field(value, "content", value)
    normalized = []
    for block in content or []:
        if _field(block, "type") != "tool_use":
            continue
        arguments, parse_error = _parse_arguments(_field(block, 'input'))
        item = {
            "type": "tool_use",
            "id": _field(block, "id", ""),
            "name": _field(block, "name", ""),
            "input": arguments,
        }
        if parse_error:
            item['parse_error'] = parse_error
        normalized.append(item)
    return normalized


def _parse_arguments(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return {}, 'invalid_json'
    if not isinstance(value, dict):
        return {}, 'expected_object'
    return value, None


def build_openai_tool_followup(
    response: Any,
    tool_results: Sequence[tuple[str, Any]],
) -> List[Dict[str, Any]]:
    """Build the assistant/tool messages required for the next tool round."""
    message = _completion_message(response)
    if message is None:
        raise TypeError("OpenAI response does not contain a completion message")

    assistant_tool_calls = []
    for call in extract_tool_calls(response):
        assistant_tool_calls.append({
            "id": call["id"],
            "type": "function",
            "function": {
                "name": call["name"],
                "arguments": call.get('_raw_arguments', json.dumps(call["input"], ensure_ascii=False)),
            },
        })

    messages: List[Dict[str, Any]] = [{
        "role": "assistant",
        "content": _field(message, "content"),
        "tool_calls": assistant_tool_calls,
    }]
    for tool_call_id, result in tool_results:
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps(result, ensure_ascii=False),
        })
    return messages


def _to_openai_tool(tool: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a CareGuide tool schema to an OpenAI function tool."""
    if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
        return dict(tool)
    return {
        "type": "function",
        "function": {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema") or tool.get("parameters") or {
                "type": "object",
                "properties": {},
            },
        },
    }


def _to_legacy_tool(tool: Dict[str, Any]) -> Dict[str, Any]:
    """Keep the old shape for injected legacy test doubles."""
    if "input_schema" in tool:
        return dict(tool)
    function = tool.get("function") or {}
    return {
        "name": function.get("name", tool.get("name", "")),
        "description": function.get("description", tool.get("description", "")),
        "input_schema": function.get("parameters") or {"type": "object"},
    }


def _completion_message(value: Any) -> Any:
    choices = _field(value, "choices")
    if choices:
        return _field(choices[0], "message")
    return None


def _extract_text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        text = value.get("text")
        if isinstance(text, str):
            return text
        nested = value.get("content")
        return _extract_text_value(nested) if nested is not None else ""

    texts: List[str] = []
    if isinstance(value, Iterable):
        for block in value:
            if isinstance(block, str):
                texts.append(block)
                continue
            block_type = _field(block, "type")
            text = _field(block, "text")
            if isinstance(text, str) and block_type in (None, "text", "output_text"):
                texts.append(text)
            elif text is None:
                nested = _field(block, "content")
                if nested is not None:
                    nested_text = _extract_text_value(nested)
                    if nested_text:
                        texts.append(nested_text)
    return "\n".join(text for text in texts if text)


_MISSING = object()


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)
