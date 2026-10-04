from __future__ import annotations

import copy
import importlib
import json
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest

from gptmock.core.settings import Settings


def _sse_bytes(events: list[dict[str, Any]]) -> bytes:
    return b"".join(f"data: {json.dumps(event)}\n\n".encode() for event in events)


def _completed(output: list[dict[str, Any]], *, response_id: str = "resp_final") -> dict[str, Any]:
    return {
        "type": "response.completed",
        "response": {"id": response_id, "status": "completed", "output": output},
    }


def _reasoning(item_id: str) -> dict[str, Any]:
    return {
        "type": "reasoning",
        "id": item_id,
        "encrypted_content": f"  opaque/{item_id}+==\n",
        "summary": [{"type": "summary_text", "text": f"Summary {item_id}"}],
        "future_field": {"keep": ["unchanged", 2]},
    }


def _call(call_id: str, *, name: str = "view_image") -> dict[str, Any]:
    return {
        "type": "function_call",
        "id": f"fc_{call_id}",
        "call_id": call_id,
        "name": name,
        "arguments": json.dumps({"path": f"{call_id}/image.png", "detail": "original"}),
        "status": "completed",
        "future_call_field": {"opaque": "preserve"},
    }


def _message(text: str = "saw image") -> dict[str, Any]:
    return {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
        "future_message_field": ["preserve"],
    }


def _tool_result(call: dict[str, Any]) -> dict[str, Any]:
    return {"type": "function_call_output", "call_id": call["call_id"], "output": f"image:{call['arguments']}"}


@dataclass
class MockUpstream:
    rounds: list[list[dict[str, Any]]] = field(default_factory=list)
    payloads: list[dict[str, Any]] = field(default_factory=list)
    session_ids: list[str] = field(default_factory=list)
    executed_arguments: list[Any] = field(default_factory=list)

    async def send(
        self,
        payload: dict[str, Any],
        access_token: str,
        account_id: str,
        session_id: str,
        http_client: httpx.AsyncClient,
        *,
        verbose: bool = False,
    ) -> httpx.Response:
        del access_token, account_id, http_client, verbose
        self.payloads.append(copy.deepcopy(payload))
        self.session_ids.append(session_id)
        return httpx.Response(200, content=_sse_bytes(self.rounds[len(self.payloads) - 1]))

    def execute(self, arguments: Any) -> str:
        self.executed_arguments.append(arguments)
        return f"image:{arguments}"


@pytest.fixture
def upstream(monkeypatch: pytest.MonkeyPatch) -> MockUpstream:
    mock = MockUpstream()
    responses_module = importlib.import_module("gptmock.services.responses")

    async def auth() -> tuple[str, str]:
        return "fake-token", "fake-account"

    monkeypatch.setattr(responses_module, "get_effective_chatgpt_auth", auth)
    monkeypatch.setattr(responses_module, "send_upstream_request", mock.send)
    monkeypatch.setattr(responses_module, "execute_view_image", mock.execute)
    return mock


async def _process(
    payload: dict[str, Any], *, client_session_id: str | None = None, settings: Settings | None = None,
) -> tuple[Any, bool]:
    responses_module = importlib.import_module("gptmock.services.responses")
    async with httpx.AsyncClient() as client:
        return await responses_module.process_responses_api(
            payload, settings or Settings(), client, client_session_id=client_session_id,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("snapshot_source", ["terminal_only", "terminal_overrides_events", "item_done_fallback"])
async def test_internal_followup_replays_complete_ordered_output(
    upstream: MockUpstream,
    snapshot_source: str,
) -> None:
    calls = [_call("call_one"), _call("call_two")]
    output = [_reasoning("rs_1"), _message("Inspecting two images"), calls[0], _reasoning("rs_2"), calls[1]]
    events: list[dict[str, Any]] = []
    if snapshot_source != "terminal_only":
        for index in reversed(range(len(output))):
            item = copy.deepcopy(output[index])
            if snapshot_source == "terminal_overrides_events" and item["type"] == "function_call":
                item["name"] = "stale_caller_tool"
                item["arguments"] = "stale arguments"
            event = {"type": "response.output_item.done", "output_index": index, "item": item}
            events.extend([event, copy.deepcopy(event)])
    terminal = _completed(output, response_id="resp_tools")
    if snapshot_source == "item_done_fallback":
        del terminal["response"]["output"]
    terminal["response"]["usage"] = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}
    events.append(terminal)
    final = _completed([_message()])
    final["response"]["usage"] = {"input_tokens": 150, "output_tokens": 5, "total_tokens": 155}
    upstream.rounds = [events, [final]]
    payload = {
        "model": "gpt-5.4-mini",
        "input": [{"role": "user", "content": "Inspect the images", "caller_field": {"retain": True}}],
        "tools": [{"type": "view_image"}],
        "stream": False,
    }
    original = copy.deepcopy(payload)

    result, is_streaming = await _process(payload)

    assert is_streaming is False
    assert len(upstream.payloads) == 2
    assert upstream.payloads[0]["input"] == original["input"]
    assert upstream.payloads[1]["input"] == original["input"] + output + [_tool_result(call) for call in calls]
    assert upstream.executed_arguments == [call["arguments"] for call in calls]
    assert payload == original
    assert result["output"] == final["response"]["output"]
    # Native usage describes the last upstream response, not all internal requests.
    assert result["usage"] == final["response"]["usage"]


@pytest.mark.asyncio
async def test_repeated_followups_keep_each_round_once(upstream: MockUpstream) -> None:
    first_call, second_call = _call("first"), _call("second")
    first_output = [_reasoning("rs_first"), first_call]
    second_output = [_reasoning("rs_second"), _message("Need another image"), second_call]
    upstream.rounds = [[_completed(first_output)], [_completed(second_output)], [_completed([_message()])]]
    payload = {"model": "gpt-5.4-mini", "input": "Inspect", "tools": [{"type": "view_image"}]}
    original = copy.deepcopy(payload)

    await _process(payload)

    first_input = upstream.payloads[0]["input"]
    assert upstream.payloads[1]["input"] == first_input + first_output + [_tool_result(first_call)]
    assert upstream.payloads[2]["input"] == (
        first_input + first_output + [_tool_result(first_call)] + second_output + [_tool_result(second_call)]
    )
    assert upstream.executed_arguments == [first_call["arguments"], second_call["arguments"]]
    assert payload == original


@pytest.mark.asyncio
@pytest.mark.parametrize("caller_tool", [False, True])
async def test_terminal_snapshot_controls_internal_execution(upstream: MockUpstream, caller_tool: bool) -> None:
    output = [_reasoning("rs_1"), _call("view"), _call("external", name="lookup")] if caller_tool else []
    upstream.rounds = [[
        {"type": "response.output_item.done", "output_index": 0, "item": _call("stale_view")},
        _completed(output),
    ]]

    result, _ = await _process({"model": "gpt-5.4-mini", "input": "Inspect", "tools": [{"type": "view_image"}]})

    assert len(upstream.payloads) == 1
    assert not upstream.executed_arguments
    assert result["output"] == output


@pytest.mark.asyncio
@pytest.mark.parametrize("compat", ["standard", "think-tags"])
@pytest.mark.parametrize("output", [
    [],
    [_reasoning("rs_terminal")],
    [_call("terminal", name="lookup")],
    [_reasoning("rs_terminal"), _call("terminal", name="lookup")],
    [{"type": "message", "role": "assistant", "content": []}],
])
async def test_terminal_output_does_not_gain_synthetic_text_from_deltas(
    upstream: MockUpstream, compat: str, output: list[dict[str, Any]],
) -> None:
    upstream.rounds = [[
        {"type": "response.output_text.delta", "delta": "stale output"},
        {"type": "response.reasoning_summary_text.delta", "delta": "display summary"},
        _completed(output),
    ]]

    result, _ = await _process(
        {"model": "gpt-5.4-mini", "input": "Inspect"}, settings=Settings(reasoning_compat=compat),
    )

    assert result["output"] == output


@pytest.mark.asyncio
async def test_think_tags_render_terminal_text_instead_of_stale_deltas(upstream: MockUpstream) -> None:
    upstream.rounds = [[
        {"type": "response.output_text.delta", "delta": "stale output"},
        {"type": "response.reasoning_summary_text.delta", "delta": "display summary"},
        _completed([_message("terminal answer")]),
    ]]

    result, _ = await _process(
        {"model": "gpt-5.4-mini", "input": "Inspect"}, settings=Settings(reasoning_compat="think-tags"),
    )

    assert result["output"] == [_message("<think>display summary</think>terminal answer")]


@pytest.mark.asyncio
@pytest.mark.parametrize("compat", ["standard", "think-tags"])
async def test_missing_terminal_output_keeps_delta_text_fallback(upstream: MockUpstream, compat: str) -> None:
    upstream.rounds = [[
        {"type": "response.output_text.delta", "delta": "fallback answer"},
        {"type": "response.reasoning_summary_text.delta", "delta": "display summary"},
        {"type": "response.completed", "response": {"id": "resp_fallback"}},
    ]]

    result, _ = await _process(
        {"model": "gpt-5.4-mini", "input": "Inspect"}, settings=Settings(reasoning_compat=compat),
    )

    prefix = "<think>display summary</think>" if compat == "think-tags" else ""
    assert result["output"][0]["content"][0]["text"] == f"{prefix}fallback answer"


@pytest.mark.asyncio
async def test_unindexed_items_preserve_order_and_deduplicate_call_events(upstream: MockUpstream) -> None:
    output = [_reasoning("rs_1"), _call("call_1")]
    upstream.rounds = [[
        {"type": "response.output_item.done", "item": output[0]},
        {"type": "response.output_item.done", "item": output[1]},
        {"type": "response.output_item.done", "item": output[1]},
        {"type": "response.completed", "response": {"id": "resp_1"}},
    ], [_completed([_message()])]]

    await _process({"model": "gpt-5.4-mini", "input": [], "tools": [{"type": "view_image"}]})

    assert upstream.payloads[1]["input"] == output + [_tool_result(output[1])]
    assert upstream.executed_arguments == [output[1]["arguments"]]


@pytest.mark.asyncio
async def test_fallback_returns_complete_items_for_caller_owned_tools(upstream: MockUpstream) -> None:
    output = [_reasoning("rs_caller"), _call("caller", name="lookup")]
    upstream.rounds = [[
        {"type": "response.output_item.done", "output_index": 1, "item": output[1]},
        {"type": "response.output_item.done", "output_index": 0, "item": output[0]},
        {"type": "response.completed", "response": {"id": "resp_1"}},
    ]]

    result, _ = await _process({"model": "gpt-5.4-mini", "input": "Inspect"})

    assert result["output"] == output
    assert len(upstream.payloads) == 1
    assert not upstream.executed_arguments


@pytest.mark.asyncio
async def test_streaming_forwards_native_reasoning_and_tool_events_unchanged(upstream: MockUpstream) -> None:
    output = [_reasoning("rs_stream"), _call("stream")]
    events = [
        {"type": "response.output_item.done", "output_index": index, "item": item}
        for index, item in enumerate(output)
    ] + [_completed(output)]
    upstream.rounds = [events]

    result, is_streaming = await _process({
        "model": "gpt-5.4-mini", "input": "Inspect", "tools": [{"type": "view_image"}], "stream": True,
    })
    frames = "".join([frame async for frame in result])

    assert is_streaming is True
    assert frames.encode() == _sse_bytes(events)
    assert len(upstream.payloads) == 1
    assert not upstream.executed_arguments


@pytest.mark.asyncio
@pytest.mark.parametrize("include", [[], ["message.output_text.logprobs"], ["reasoning.encrypted_content", "other.value"]])
async def test_explicit_cache_key_and_session_preserve_include_values(upstream: MockUpstream, include: list[str]) -> None:
    upstream.rounds = [[_completed([_message()])]]
    payload = {
        "model": "gpt-5.4-mini",
        "input": "Inspect",
        "prompt_cache_key": "  explicit-cache-key  ",
        "client_metadata": {"session_id": "body-session"},
        "include": include,
    }
    original = copy.deepcopy(payload)

    await _process(payload, client_session_id="header-session")

    expected_include = list(include)
    if "reasoning.encrypted_content" not in expected_include:
        expected_include.append("reasoning.encrypted_content")
    assert upstream.payloads[0]["prompt_cache_key"] == "  explicit-cache-key  "
    assert upstream.session_ids == ["header-session"]
    assert upstream.payloads[0]["include"] == expected_include
    assert payload == original


@pytest.mark.asyncio
@pytest.mark.parametrize("cache_key", [None, "", " \t "])
async def test_empty_cache_key_falls_back_to_body_session(upstream: MockUpstream, cache_key: str | None) -> None:
    upstream.rounds = [[_completed([_message()])]]

    await _process({
        "model": "gpt-5.4-mini", "input": "Inspect", "prompt_cache_key": cache_key,
        "client_metadata": {"session_id": "body-session"},
    }, client_session_id="  ")

    assert upstream.payloads[0]["prompt_cache_key"] == "body-session"
    assert upstream.session_ids == ["body-session"]


@pytest.mark.asyncio
@pytest.mark.parametrize("cache_key", [123, True, ["bad"], {"bad": "key"}])
async def test_invalid_cache_key_is_a_client_error_before_upstream(upstream: MockUpstream, cache_key: Any) -> None:
    responses_module = importlib.import_module("gptmock.services.responses")

    with pytest.raises(responses_module.ChatCompletionError) as exc_info:
        await _process({"model": "gpt-5.4-mini", "input": "Inspect", "prompt_cache_key": cache_key})

    assert exc_info.value.status_code == 400
    assert "prompt_cache_key" in str(exc_info.value)
    assert not upstream.payloads
