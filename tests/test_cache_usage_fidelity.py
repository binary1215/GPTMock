from __future__ import annotations

import importlib
import json
import sys
from copy import deepcopy
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from gptmock.app import create_app
from gptmock.core.settings import Settings
from gptmock.core.utils import extract_usage
from gptmock.infra.session import resolve_client_session_id, resolve_prompt_cache_key

USAGE = {
    "input_tokens": 1200,
    "output_tokens": 80,
    "total_tokens": 1280,
    "input_tokens_details": {"cached_tokens": 1024, "future_detail": {"tokens": 4}},
    "output_tokens_details": {"reasoning_tokens": 64},
}
CHAT_USAGE = {
    "prompt_tokens": 1200,
    "completion_tokens": 80,
    "total_tokens": 1280,
    "prompt_tokens_details": USAGE["input_tokens_details"],
    "completion_tokens_details": USAGE["output_tokens_details"],
}


def test_usage_details_preserve_unknown_fields_without_mutation() -> None:
    event = {"response": {"usage": deepcopy(USAGE)}}
    result = extract_usage(event)
    assert result == CHAT_USAGE
    result["prompt_tokens_details"]["future_detail"]["tokens"] = 999
    assert event["response"]["usage"] == USAGE


@pytest.mark.parametrize("details", [None, {}, {"cached_tokens": 0}])
def test_usage_does_not_invent_missing_cache_details(details: Any) -> None:
    usage = {"input_tokens": 12, "output_tokens": 2}
    if details is not None:
        usage["input_tokens_details"] = details
    result = extract_usage({"response": {"usage": usage}})
    assert result["total_tokens"] == 14
    assert ("prompt_tokens_details" in result) == (details is not None)
    if details is not None:
        assert result["prompt_tokens_details"] == details
    assert "completion_tokens_details" not in result


def test_session_and_cache_key_are_independent() -> None:
    payload = {"client_metadata": {"session_id": "body-session"}, "prompt_cache_key": " explicit key "}
    assert resolve_client_session_id(payload, " header-session ") == "header-session"
    assert resolve_client_session_id(payload, "  ") == "body-session"
    assert resolve_prompt_cache_key(payload, "header-session") == " explicit key "
    assert resolve_client_session_id({"metadata": {"user_id": "not-a-session"}}) is None


@pytest.mark.parametrize("key", [None, "", "  "])
def test_missing_cache_key_uses_session(key: Any) -> None:
    assert resolve_prompt_cache_key({"prompt_cache_key": key}, "session") == "session"


@pytest.mark.parametrize("key", [False, 123, {}, []])
def test_invalid_cache_key_is_rejected(key: Any) -> None:
    with pytest.raises(ValueError, match="prompt_cache_key"):
        resolve_prompt_cache_key({"prompt_cache_key": key}, "session")


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("endpoint", ["/v1/chat/completions", "/v1/completions", "/v1/responses"])
def test_http_usage_cache_key_and_session_roundtrip(
    monkeypatch: pytest.MonkeyPatch, endpoint: str, stream: bool,
) -> None:
    module = importlib.import_module(
        "gptmock.services.responses" if endpoint.endswith("responses") else "gptmock.services.chat",
    )
    captured: list[tuple[dict[str, Any], str]] = []

    async def fake_auth() -> tuple[str, str]:
        return "test-token", "test-account"

    async def fake_send(payload: dict[str, Any], *args: Any, **kwargs: Any) -> httpx.Response:
        captured.append((deepcopy(payload), kwargs.get("session_id", args[2] if len(args) > 2 else None)))
        events = [
            {"type": "response.output_text.delta", "delta": "ok"},
            {"type": "response.completed", "response": {
                "id": "resp_test", "status": "completed", "model": "gpt-5.6-sol", "usage": deepcopy(USAGE),
                "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}],
            }},
        ]
        return httpx.Response(200, content="".join(f"data: {json.dumps(event)}\n\n" for event in events).encode())

    monkeypatch.setattr(module, "get_effective_chatgpt_auth", fake_auth)
    monkeypatch.setattr(module, "send_upstream_request", fake_send)
    monkeypatch.delenv("GPTMOCK_API_KEY", raising=False)
    payload = {
        "model": "gpt-5.6-sol", "stream": stream, "stream_options": {"include_usage": True},
        "prompt_cache_key": "client-key", "client_metadata": {"session_id": "body-session"},
    }
    if endpoint.endswith("responses"):
        payload["input"] = "hello"
    elif endpoint.endswith("chat/completions"):
        payload["messages"] = [{"role": "user", "content": "hello"}]
    else:
        payload["prompt"] = "hello"
    with TestClient(create_app()) as client:
        response = client.post(endpoint, json=payload, headers={"session_id": "header-session"})
    assert response.status_code == 200, response.text
    assert captured[0][0]["prompt_cache_key"] == "client-key"
    assert captured[0][1] == "header-session"
    if stream:
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ") and line != "data: [DONE]"]
        usage = events[-1]["response"]["usage"] if endpoint.endswith("responses") else next(e["usage"] for e in events if e.get("usage"))
    else:
        usage = response.json()["usage"]
    assert usage == (USAGE if endpoint.endswith("responses") else CHAT_USAGE)


def test_replay_setting_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GPTMOCK_REASONING_REPLAY", raising=False)
    assert Settings(_env_file=None).reasoning_replay is False
    monkeypatch.setenv("GPTMOCK_REASONING_REPLAY", "true")
    assert Settings(_env_file=None).reasoning_replay is True


@pytest.mark.parametrize(("flags", "expected"), [([], None), (["--reasoning-replay"], True), (["--no-reasoning-replay"], False)])
def test_replay_cli_does_not_override_environment_when_omitted(
    monkeypatch: pytest.MonkeyPatch, flags: list[str], expected: bool | None,
) -> None:
    from gptmock import cli

    captured: dict[str, Any] = {}

    def fake_serve(**kwargs: Any) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setenv("GPTMOCK_REASONING_REPLAY", "true")
    monkeypatch.setattr(cli, "cmd_serve", fake_serve)
    monkeypatch.setattr(sys, "argv", ["gptmock", "serve", *flags])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    assert captured["reasoning_replay"] is expected
