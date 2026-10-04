from __future__ import annotations

import importlib
import json
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import httpx
import pytest

if TYPE_CHECKING:
    from gptmock.core.settings import Settings


@pytest.fixture(autouse=True)
def reject_external_http(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missed mock must fail locally, including after other tests reload modules."""
    async def reject_async(self: Any, request: httpx.Request) -> httpx.Response:
        raise AssertionError("Reasoning replay tests must not make external HTTP requests")

    def reject_sync(self: Any, request: httpx.Request) -> httpx.Response:
        raise AssertionError("Reasoning replay tests must not make external HTTP requests")

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", reject_async)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", reject_sync)


def _settings(**options: Any) -> Settings:
    return importlib.import_module("gptmock.core.settings").Settings(**options)


def _reasoning(item_id: str | None, encrypted: str) -> dict[str, Any]:
    item: dict[str, Any] = {
        "type": "reasoning", "summary": [], "encrypted_content": encrypted,
        "future_field": {"opaque": ["unchanged", 1]},
    }
    if item_id is not None:
        item["id"] = item_id
    return item


def _terminal(output: list[dict[str, Any]] | None = None, **extra: Any) -> dict[str, Any]:
    response: dict[str, Any] = {"id": "resp_replay", "status": "completed", **extra}
    if output is not None:
        response["output"] = output
    return {"type": "response.completed", "response": response}


def _message(text: str) -> dict[str, Any]:
    return {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}


def _payload(**extra: Any) -> dict[str, Any]:
    return {"model": "gpt-5.6-sol", "messages": [{"role": "user", "content": "hello"}], **extra}


async def _post(
    monkeypatch: pytest.MonkeyPatch,
    events: list[dict[str, Any]],
    payload: dict[str, Any],
    *,
    settings: Settings | None = None,
) -> tuple[httpx.Response, list[dict[str, Any]]]:
    chat = importlib.import_module("gptmock.services.chat")
    app_module = importlib.import_module("gptmock.app")
    captured: list[dict[str, Any]] = []

    async def fake_auth() -> tuple[str, str]:
        return "test-token", "test-account"

    async def fake_send(upstream_payload: dict[str, Any], *args: Any, **kwargs: Any) -> httpx.Response:
        captured.append(deepcopy(upstream_payload))
        body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
        return httpx.Response(200, content=body.encode())

    monkeypatch.setattr(chat, "get_effective_chatgpt_auth", fake_auth)
    monkeypatch.setattr(chat, "send_upstream_request", fake_send)
    app = app_module.create_app(settings=settings or _settings())
    async with httpx.AsyncClient() as upstream_client:
        app.state.http_client = upstream_client
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            result = await client.post("/v1/chat/completions", json=payload)
    return result, captured


def _chunks(response: httpx.Response) -> list[dict[str, Any]]:
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: {")]


def _replay(response: httpx.Response, stream: bool) -> list[dict[str, Any]]:
    assert response.status_code == 200, response.text
    if not stream:
        return response.json()["choices"][0]["message"].get("reasoning_items", [])
    replay_chunks = [
        choice for chunk in _chunks(response) for choice in chunk.get("choices", [])
        if "reasoning_items" in choice["delta"]
    ]
    assert len(replay_chunks) <= 1
    if not replay_chunks:
        return []
    assert replay_chunks[0]["finish_reason"] in ("stop", "tool_calls", "length", "content_filter")
    return replay_chunks[0]["delta"]["reasoning_items"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_final_snapshot_preserves_multiple_opaque_items_once(
    monkeypatch: pytest.MonkeyPatch, stream: bool,
) -> None:
    first = _reasoning("rs_first", " first ==\nopaque/+")
    second = _reasoning(None, "opaque-second")
    events = [
        {"type": "response.output_item.done", "output_index": 2, "item": second},
        {"type": "response.output_item.done", "output_index": 0, "item": _reasoning("rs_first", "stale")},
        {"type": "response.output_item.done", "output_index": 2, "item": second},
        _terminal([first, _message("answer"), second]),
    ]
    result, captured = await _post(monkeypatch, events, _payload(stream=stream, reasoning_replay=True))
    assert _replay(result, stream) == [first, second]
    assert captured[0]["include"] == ["reasoning.encrypted_content"]
    assert "reasoning_replay" not in captured[0]
    if stream:
        assert "".join(
            choice["delta"].get("content", "") for chunk in _chunks(result) for choice in chunk["choices"]
        ) == "answer"
    else:
        assert result.json()["choices"][0]["message"]["content"] == "answer"


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("final_output", [None, []])
async def test_item_done_fallback_orders_and_deduplicates_when_terminal_snapshot_is_elided(
    monkeypatch: pytest.MonkeyPatch, stream: bool, final_output: list[dict[str, Any]] | None,
) -> None:
    first, second = _reasoning("rs_first", "first"), _reasoning(None, "second")
    events = [
        {"type": "response.output_item.done", "output_index": 2, "item": second},
        {"type": "response.output_item.done", "output_index": 0, "item": first},
        {"type": "response.output_item.done", "output_index": 2, "item": second},
        _terminal(final_output),
    ]
    result, _ = await _post(monkeypatch, events, _payload(stream=stream, reasoning_replay=True))
    assert _replay(result, stream) == [first, second]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("reasoning_replay", [False, True])
async def test_codex_elided_terminal_preserves_completed_message_tools_and_reasoning(
    monkeypatch: pytest.MonkeyPatch, stream: bool, reasoning_replay: bool,
) -> None:
    reasoning = [_reasoning("rs_first", "first"), _reasoning("rs_second", "second")]
    tool = {"type": "function_call", "id": "fc_lookup", "call_id": "call_lookup", "name": "lookup", "arguments": "{}"}
    message = {"id": "msg_answer", "status": "completed", **_message("complete answer")}
    events = [
        {"type": "response.output_item.done", "output_index": 3, "item": message},
        {"type": "response.output_item.done", "output_index": 0, "item": reasoning[0]},
        {"type": "response.output_item.done", "output_index": 1, "item": tool},
        {"type": "response.output_item.done", "output_index": 2, "item": reasoning[1]},
        {"type": "response.output_item.done", "output_index": 3, "item": message},
        _terminal([], usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}),
    ]
    result, _ = await _post(monkeypatch, events, _payload(
        stream=stream, reasoning_replay=reasoning_replay, stream_options={"include_usage": True},
    ))
    assert _replay(result, stream) == (reasoning if reasoning_replay else [])
    if stream:
        chunks = _chunks(result)
        deltas = [choice["delta"] for chunk in chunks for choice in chunk["choices"]]
        assert "".join(delta.get("content", "") for delta in deltas) == "complete answer"
        calls = [call for delta in deltas for call in delta.get("tool_calls", [])]
        assert len(calls) == 1
        assert calls[0]["id"] == "call_lookup"
        assert chunks[-1]["usage"]["total_tokens"] == 15
    else:
        body = result.json()
        answer = body["choices"][0]["message"]
        assert answer["content"] == "complete answer"
        assert answer["tool_calls"][0]["id"] == "call_lookup"
        assert len(answer["tool_calls"]) == 1
        assert body["usage"]["total_tokens"] == 15


@pytest.mark.asyncio
async def test_nonempty_terminal_snapshot_replaces_stale_completed_chat_output(monkeypatch: pytest.MonkeyPatch) -> None:
    old_tool = {"type": "function_call", "call_id": "old_call", "name": "old_tool", "arguments": "{}"}
    events = [
        {"type": "response.output_item.done", "item": _message("old answer")},
        {"type": "response.output_item.done", "item": old_tool},
        {"type": "response.output_item.done", "item": _reasoning("old_reasoning", "stale")},
        _terminal([_message("final answer")]),
    ]
    result, _ = await _post(monkeypatch, events, _payload(reasoning_replay=True))
    message = result.json()["choices"][0]["message"]
    assert message == {"role": "assistant", "content": "final answer"}


@pytest.mark.asyncio
async def test_empty_terminal_without_completed_items_does_not_recover_partial_chat_text(monkeypatch: pytest.MonkeyPatch) -> None:
    events = [
        {"type": "response.output_text.delta", "delta": "unfinished"},
        {"type": "response.output_item.added", "item": _reasoning("rs_partial", "not completed")},
        _terminal([]),
    ]
    result, _ = await _post(monkeypatch, events, _payload(reasoning_replay=True))
    assert result.json()["choices"][0]["message"] == {"role": "assistant", "content": None}


def test_output_replay_preserves_all_completed_item_types_and_never_invents_delta_items() -> None:
    replay_module = importlib.import_module("gptmock.services.replay")
    completed = replay_module.OutputReplay()
    reasoning = _reasoning("rs", "opaque")
    unknown = {"type": "future_output_kind", "id": "future", "opaque_fields": {"data": [1, 2]}}
    message = {"id": "msg", **_message("answer")}
    originals = [reasoning, unknown, message]
    for index in (2, 0, 1, 2):
        completed.observe({"type": "response.output_item.done", "output_index": index, "item": originals[index]})
    completed.observe(_terminal([]))
    assert completed.items() == originals
    copy = completed.items()
    copy[1]["opaque_fields"]["data"].append(3)
    assert completed.items() == originals
    completed.observe(_terminal([_message("authoritative")]))
    assert completed.items() == [_message("authoritative")]

    incomplete = replay_module.OutputReplay()
    incomplete.observe({"type": "response.output_text.delta", "delta": "unfinished"})
    incomplete.observe({"type": "response.output_item.added", "item": reasoning})
    incomplete.observe(_terminal([]))
    assert incomplete.items() == []


def test_reasoning_replay_facade_does_not_restore_stale_reasoning_over_nonempty_final_output() -> None:
    replay = importlib.import_module("gptmock.services.replay").ReasoningReplay()
    replay.observe({"type": "response.output_item.done", "item": _reasoning("rs", "old")})
    replay.observe(_terminal([_message("final")]))
    assert replay.items() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_terminal_only_reasoning_and_tool_calls_survive_round_trip(
    monkeypatch: pytest.MonkeyPatch, stream: bool,
) -> None:
    reasoning = [_reasoning("rs_a", "A"), _reasoning("rs_b", "B")]
    tool = {"type": "function_call", "call_id": "call_lookup", "name": "lookup", "arguments": '{"x":1}'}
    result, _ = await _post(
        monkeypatch, [_terminal([*reasoning, tool])], _payload(stream=stream, reasoning_replay=True),
    )
    assert _replay(result, stream) == reasoning
    if stream:
        choices = [choice for chunk in _chunks(result) for choice in chunk["choices"]]
        assert choices[0]["delta"]["role"] == "assistant"
        calls = [call for choice in choices for call in choice["delta"].get("tool_calls", [])]
        assert len(calls) == 1
        assert calls[0]["id"] == "call_lookup"
        assert choices[-1]["finish_reason"] == "tool_calls"
        assistant = {"role": "assistant", "content": None, "reasoning_items": reasoning, "tool_calls": calls}
    else:
        assistant = result.json()["choices"][0]["message"]
        assert result.json()["choices"][0]["finish_reason"] == "tool_calls"
    original = deepcopy(assistant)
    followup = _payload(
        reasoning_replay=False,
        messages=[
            {"role": "user", "content": "lookup"}, assistant,
            {"role": "tool", "tool_call_id": "call_lookup", "content": "found"},
            {"role": "user", "content": "continue"},
        ],
    )
    second, captured = await _post(monkeypatch, [_terminal([_message("done")])], followup)
    assert second.status_code == 200
    assert captured[0]["input"][1:3] == reasoning
    assert [item["type"] for item in captured[0]["input"]] == [
        "message", "reasoning", "reasoning", "function_call", "function_call_output", "message",
    ]
    replayed_call = captured[0]["input"][3]
    assert {key: value for key, value in replayed_call.items() if key != "arguments"} == {
        key: value for key, value in tool.items() if key != "arguments"
    }
    assert json.loads(replayed_call["arguments"]) == json.loads(tool["arguments"])
    assert assistant == original


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(("setting", "override", "enabled"), [(False, None, False), (True, None, True), (True, False, False)])
async def test_replay_output_opt_in_and_request_override(
    monkeypatch: pytest.MonkeyPatch, stream: bool, setting: bool, override: bool | None, enabled: bool,
) -> None:
    item = _reasoning("rs", "opaque")
    payload = _payload(stream=stream)
    if override is not None:
        payload["reasoning_replay"] = override
    result, _ = await _post(monkeypatch, [_terminal([item])], payload, settings=_settings(reasoning_replay=setting))
    assert _replay(result, stream) == ([item] if enabled else [])


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", [None, 0, 1, "true", [], {}])
async def test_replay_flag_rejects_non_boolean_before_upstream(monkeypatch: pytest.MonkeyPatch, flag: Any) -> None:
    result, captured = await _post(monkeypatch, [], _payload(reasoning_replay=flag))
    assert result.status_code == 400
    assert result.json()["error"]["param"] == "reasoning_replay"
    assert captured == []


@pytest.mark.asyncio
@pytest.mark.parametrize("message", [
    {"role": "assistant", "reasoning_items": None},
    {"role": "assistant", "reasoning_items": {}},
    {"role": "assistant", "reasoning_items": ["opaque"]},
    {"role": "assistant", "reasoning_items": [{"type": "message"}]},
    {"role": "assistant", "reasoning_items": [{"type": "reasoning", "id": 1}]},
    {"role": "assistant", "reasoning_items": [{"type": "reasoning", "encrypted_content": {"bad": 1}}]},
    {"role": "assistant", "reasoning_items": [{"type": "reasoning", "summary": "text"}]},
    {"role": "assistant", "reasoning_items": [{"type": "reasoning", "summary": [{"type": "summary_text"}]}]},
    {"role": "user", "reasoning_items": []},
    {"role": "tool", "reasoning_items": []},
])
async def test_malformed_replay_is_400_even_when_output_disabled(
    monkeypatch: pytest.MonkeyPatch, message: dict[str, Any],
) -> None:
    result, captured = await _post(monkeypatch, [], _payload(messages=[message], reasoning_replay=False))
    assert result.status_code == 400
    assert "reasoning_items" in result.json()["error"]["message"]
    assert captured == []


def test_plain_reasoning_text_is_not_fabricated_as_an_upstream_reasoning_item() -> None:
    messages_module = importlib.import_module("gptmock.schemas.messages")
    converted = messages_module.convert_chat_messages_to_responses_input([
        {"role": "assistant", "content": "answer", "reasoning_content": "display only", "reasoning": "display"},
    ])
    assert converted == [_message("answer")]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("compat", ["standard", "legacy", "o3", "think-tags"])
async def test_display_reasoning_remains_independent_of_replay(
    monkeypatch: pytest.MonkeyPatch, stream: bool, compat: str,
) -> None:
    events = [
        {"type": "response.reasoning_summary_text.delta", "delta": "visible summary"},
        {"type": "response.output_text.delta", "delta": "answer"},
        {"type": "response.output_item.done", "item": _reasoning("rs", "secret-opaque")},
        _terminal(),
    ]
    result, _ = await _post(monkeypatch, events, _payload(stream=stream), settings=_settings(reasoning_compat=compat))
    assert _replay(result, stream) == []
    assert "secret-opaque" not in result.text
    if stream:
        parts = [choice["delta"] for chunk in _chunks(result) for choice in chunk["choices"]]
        assert "visible summary" in json.dumps(parts)
    else:
        message = result.json()["choices"][0]["message"]
        field = {"standard": "reasoning_content", "legacy": "reasoning_summary", "o3": "reasoning", "think-tags": "content"}[compat]
        assert "visible summary" in json.dumps(message[field])


@pytest.mark.asyncio
async def test_explicit_prompt_cache_key_and_body_session_are_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    result, captured = await _post(monkeypatch, [_terminal()], _payload(
        prompt_cache_key=" cache-key/opaque ", client_metadata={"session_id": " session-1 "},
    ))
    assert result.status_code == 200
    assert captured[0]["prompt_cache_key"] == " cache-key/opaque "
    result, captured = await _post(monkeypatch, [_terminal()], _payload(client_metadata={"session_id": " session-1 "}))
    assert result.status_code == 200
    assert captured[0]["prompt_cache_key"] == "session-1"


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_invalid_cache_key_is_rejected_before_upstream(monkeypatch: pytest.MonkeyPatch, stream: bool) -> None:
    result, captured = await _post(monkeypatch, [], _payload(stream=stream, prompt_cache_key=123))
    assert result.status_code == 400
    assert result.json()["error"]["param"] == "prompt_cache_key"
    assert captured == []


@pytest.mark.asyncio
async def test_terminal_snapshot_does_not_duplicate_streamed_text_or_tool_call(monkeypatch: pytest.MonkeyPatch) -> None:
    tool = {"type": "function_call", "call_id": "call_x", "name": "lookup", "arguments": "{}"}
    events = [
        {"type": "response.output_text.delta", "delta": "answer"},
        {"type": "response.output_item.done", "item": tool},
        _terminal([_message("answer"), tool]),
    ]
    result, _ = await _post(monkeypatch, events, _payload(stream=True, reasoning_replay=True))
    deltas = [choice["delta"] for chunk in _chunks(result) for choice in chunk["choices"]]
    assert "".join(delta.get("content", "") for delta in deltas) == "answer"
    assert len([call for delta in deltas for call in delta.get("tool_calls", [])]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", ["", '{"x":', '{"x":1}'])
@pytest.mark.parametrize("argument_event_id", ["item_x", "call_x"])
async def test_terminal_tool_arguments_fill_only_missing_suffix(
    monkeypatch: pytest.MonkeyPatch, prefix: str, argument_event_id: str,
) -> None:
    tool = {"type": "function_call", "id": "item_x", "call_id": "call_x", "name": "lookup", "arguments": '{"x":1}'}
    events = [{"type": "response.output_item.added", "item": {**tool, "arguments": ""}}]
    if prefix:
        events.append({"type": "response.function_call_arguments.delta", "item_id": argument_event_id, "delta": prefix})
    events.append(_terminal([tool]))
    result, _ = await _post(monkeypatch, events, _payload(stream=True))
    calls = [call for chunk in _chunks(result) for choice in chunk["choices"] for call in choice["delta"].get("tool_calls", [])]
    assert "".join(call["function"].get("arguments", "") for call in calls) == '{"x":1}'
    assert all(call["index"] == 0 for call in calls)
    assert sum("name" in call["function"] for call in calls) == 1


@pytest.mark.asyncio
async def test_buffered_argument_prefix_and_done_alias_do_not_duplicate_terminal_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    tool = {"type": "function_call", "id": "item_x", "call_id": "call_x", "name": "lookup", "arguments": '{"x":1}'}
    events = [
        {"type": "response.function_call_arguments.delta", "item_id": "item_x", "delta": '{"x":'},
        {"type": "response.output_item.added", "item": {**tool, "arguments": ""}},
        {"type": "response.function_call_arguments.done", "item_id": "call_x", "arguments": tool["arguments"]},
        {"type": "response.output_item.done", "item": tool},
        _terminal([tool]),
    ]
    result, _ = await _post(monkeypatch, events, _payload(stream=True))
    arguments = [
        call["function"].get("arguments", "") for chunk in _chunks(result)
        for choice in chunk["choices"] for call in choice["delta"].get("tool_calls", [])
    ]
    assert arguments == ["", '{"x":', "1}"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_incomplete_terminal_only_tool_call_is_not_executable(monkeypatch: pytest.MonkeyPatch, stream: bool) -> None:
    event = {"type": "response.incomplete", "response": {
        "id": "resp_incomplete", "status": "incomplete", "output": [
            _reasoning("rs", "opaque"),
            {"type": "function_call", "call_id": "call_x", "name": "lookup", "arguments": '{"unfinished"'},
        ],
    }}
    result, _ = await _post(monkeypatch, [event], _payload(stream=stream, reasoning_replay=True))
    if stream:
        assert "do not execute partial arguments" in result.text
        assert "tool_calls" not in result.text
        assert "reasoning_items" not in result.text
    else:
        assert result.status_code == 502
