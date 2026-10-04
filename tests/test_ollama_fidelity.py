from __future__ import annotations

import json
from collections.abc import AsyncIterator
from copy import deepcopy
from typing import Any

import pytest
from starlette.testclient import TestClient

from gptmock.app import create_app
from gptmock.core.settings import Settings
from gptmock.routers import ollama
from gptmock.schemas.transform import convert_ollama_messages


def _reasoning_items() -> list[dict[str, Any]]:
    return [
        {"type": "reasoning", "id": "rs_one", "summary": [{"type": "summary_text", "text": "plan"}],
         "encrypted_content": "opaque-one"},
        {"type": "reasoning", "id": "rs_two", "summary": [], "encrypted_content": "opaque-two"},
    ]


def _usage() -> dict[str, Any]:
    return {
        "prompt_tokens": 30,
        "completion_tokens": 12,
        "total_tokens": 42,
        "prompt_tokens_details": {"cached_tokens": 20, "cache_creation_tokens": 3},
        "completion_tokens_details": {"reasoning_tokens": 7},
    }


def _chat_response() -> dict[str, Any]:
    return {
        "model": "actual-model",
        "service_tier": "default",
        "choices": [{"message": {
            "role": "assistant", "content": "answer", "reasoning_content": "plan",
            "reasoning_items": _reasoning_items(),
        }, "finish_reason": "stop"}],
        "usage": _usage(),
    }


class _Frames:
    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            yield f"data: {json.dumps(chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    async def aclose(self) -> None:
        self.closed = True


def _chat_stream() -> _Frames:
    return _Frames([
        {"model": "actual-model", "choices": [{"delta": {"content": "answer"}}]},
        {"choices": [{"delta": {"reasoning_content": "plan"}}]},
        {"service_tier": "default", "choices": [{
            "delta": {"reasoning_items": _reasoning_items()}, "finish_reason": "stop",
        }]},
        {"choices": [], "usage": _usage()},
    ])


@pytest.mark.parametrize("generate", [False, True])
def test_non_stream_preserves_detailed_usage_and_replay_without_mutation(generate: bool) -> None:
    source = _chat_response()
    before = deepcopy(source)
    converter = (
        ollama._convert_openai_to_ollama_generate_response
        if generate else ollama._convert_openai_to_ollama_response
    )

    result = converter(source, "alias")

    assert result["usage"] == _usage()
    assert result["prompt_eval_count"] == 30
    assert result["eval_count"] == 12
    assert result["model"] == "actual-model"
    assert result["service_tier"] == "default"
    carrier = result if generate else result["message"]
    assert carrier["reasoning_items"] == _reasoning_items()
    assert carrier["thinking"] == "plan"
    assert not any("duration" in key for key in result)
    carrier["reasoning_items"][0]["summary"][0]["text"] = "changed"
    result["usage"]["prompt_tokens_details"]["cached_tokens"] = 0
    assert source == before


@pytest.mark.parametrize("usage", [None, {}, {"completion_tokens": 0}, {"prompt_tokens": True}])
def test_non_stream_does_not_invent_unknown_counters(usage: Any) -> None:
    source = _chat_response()
    source["usage"] = usage
    result = ollama._convert_openai_to_ollama_response(source, "alias")

    assert "prompt_eval_count" not in result
    if isinstance(usage, dict) and "completion_tokens" in usage:
        assert result["eval_count"] == 0
    else:
        assert "eval_count" not in result
    assert ("usage" in result) == isinstance(usage, dict)
    assert not any("duration" in key for key in result)


@pytest.mark.asyncio
@pytest.mark.parametrize("generate", [False, True])
async def test_stream_preserves_usage_only_chunk_and_terminal_replay(generate: bool) -> None:
    source = _chat_stream()
    before = deepcopy(source.chunks)
    converter = (
        ollama._convert_openai_to_ollama_generate_stream
        if generate else ollama._convert_openai_to_ollama_stream
    )
    frames = [json.loads(frame) async for frame in converter(source, "alias")]

    assert len(frames) == 3
    terminal = frames[-1]
    assert terminal["done"] is True
    assert terminal["done_reason"] == "stop"
    assert terminal["usage"] == _usage()
    assert terminal["prompt_eval_count"] == 30
    assert terminal["eval_count"] == 12
    assert terminal["model"] == "actual-model"
    carrier = terminal if generate else terminal["message"]
    assert carrier["reasoning_items"] == _reasoning_items()
    assert not any("duration" in key for key in terminal)
    assert all("usage" not in frame for frame in frames[:-1])
    assert source.chunks == before
    assert source.closed is True


@pytest.mark.asyncio
async def test_stream_without_usage_omits_counters() -> None:
    source = _Frames([{"choices": [{"delta": {}, "finish_reason": "stop"}]}])
    frames = [json.loads(frame) async for frame in ollama._convert_openai_to_ollama_stream(source, "model")]

    assert frames[-1]["done"] is True
    assert "usage" not in frames[-1]
    assert "prompt_eval_count" not in frames[-1]
    assert "eval_count" not in frames[-1]
    assert "reasoning_items" not in frames[-1]["message"]


def test_request_preserves_replay_options_and_detaches_mutable_values() -> None:
    source = {
        "messages": [{"role": "assistant", "reasoning_items": _reasoning_items()},
                     {"role": "user", "content": "next"}],
        "reasoning_replay": True,
        "prompt_cache_key": "cache-key",
        "client_metadata": {"session_id": "body-session"},
    }
    before = deepcopy(source)
    result = ollama._build_openai_payload(source, "model")

    assert result["stream_options"] == {"include_usage": True}
    assert result["reasoning_replay"] is True
    assert result["prompt_cache_key"] == "cache-key"
    assert result["client_metadata"] == {"session_id": "body-session"}
    assert result["messages"][0]["reasoning_items"] == _reasoning_items()
    result["messages"][0]["reasoning_items"][0]["summary"][0]["text"] = "changed"
    result["client_metadata"]["session_id"] = "changed"
    assert source == before
    assert "stream_options" not in ollama._build_openai_payload({**source, "stream": False}, "model")


def test_input_replay_is_not_gated_by_output_opt_in() -> None:
    items = _reasoning_items()
    source = [{"role": "assistant", "reasoning_items": items}]
    result = convert_ollama_messages(source, None)

    assert result[0]["reasoning_items"] == items
    result[0]["reasoning_items"][0]["encrypted_content"] = "changed"
    assert source[0]["reasoning_items"] == _reasoning_items()


def test_generate_does_not_invent_replay_history() -> None:
    source = {"prompt": "new", "system": "instructions", "reasoning_items": _reasoning_items()}
    result = ollama._generate_chat_payload(source)

    assert result["messages"] == [
        {"role": "system", "content": "instructions"}, {"role": "user", "content": "new"},
    ]
    assert all("reasoning_items" not in message for message in result["messages"])


@pytest.mark.parametrize("endpoint", ["chat", "generate"])
@pytest.mark.parametrize("stream", [False, True])
def test_routes_forward_fidelity_options_and_session_header(
    monkeypatch: pytest.MonkeyPatch, endpoint: str, stream: bool,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_process(payload: dict[str, Any], **kwargs: Any) -> tuple[Any, bool]:
        captured.update(payload=deepcopy(payload), client_session_id=kwargs["client_session_id"])
        return (_chat_stream(), True) if stream else (_chat_response(), False)

    monkeypatch.setattr(ollama, "process_chat_completion", fake_process)
    settings = Settings(_env_file=None, api_key=None)
    payload: dict[str, Any] = {
        "model": "model", "stream": stream, "reasoning_replay": True,
        "prompt_cache_key": "cache-key", "client_metadata": {"session_id": "body-session"},
    }
    if endpoint == "chat":
        payload["messages"] = [{"role": "assistant", "reasoning_items": _reasoning_items()},
                               {"role": "user", "content": "next"}]
    else:
        payload["prompt"] = "new"
    with TestClient(create_app(settings)) as client:
        response = client.post(f"/api/{endpoint}", json=payload, headers={"session_id": "header-session"})

    assert response.status_code == 200
    assert captured["client_session_id"] == "header-session"
    assert captured["payload"]["reasoning_replay"] is True
    assert captured["payload"]["prompt_cache_key"] == "cache-key"
    assert captured["payload"]["client_metadata"] == {"session_id": "body-session"}
    if stream:
        assert captured["payload"]["stream_options"] == {"include_usage": True}
        result = json.loads(response.text.splitlines()[-1])
    else:
        result = response.json()
    assert result["usage"] == _usage()
    carrier = result if endpoint == "generate" else result["message"]
    assert carrier["reasoning_items"] == _reasoning_items()
