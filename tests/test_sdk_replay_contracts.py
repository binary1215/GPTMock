from __future__ import annotations

import importlib
import json
from copy import deepcopy
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import OpenAI


@pytest.mark.parametrize("protocol", ["chat", "responses"])
def test_openai_sdk_two_turn_tool_replay(monkeypatch: pytest.MonkeyPatch, protocol: str) -> None:
    app_module = importlib.import_module("gptmock.app")
    settings_module = importlib.import_module("gptmock.core.settings")
    service = importlib.import_module(f"gptmock.services.{protocol}")
    reasoning = {"type": "reasoning", "id": "rs_sdk", "encrypted_content": " opaque\n==/+ ", "summary": [], "future": {"keep": True}}
    call = {"type": "function_call", "id": "fc_sdk", "call_id": "call_sdk", "name": "lookup", "arguments": '{"key":"value"}', "status": "completed"}
    captured: list[dict[str, Any]] = []

    async def fake_auth() -> tuple[str, str]:
        return "test-token", "test-account"

    async def fake_send(payload: dict[str, Any], *args: Any, **kwargs: Any) -> httpx.Response:
        captured.append(deepcopy(payload))
        output = [reasoning, call] if len(captured) == 1 else [{
            "type": "message", "id": "msg_sdk", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": "done", "annotations": []}],
        }]
        response = {"id": f"resp_sdk_{len(captured)}", "object": "response", "created_at": 1, "status": "completed", "model": "gpt-5.6-sol", "output": output}
        event = {"type": "response.completed", "response": response}
        return httpx.Response(200, content=f"data: {json.dumps(event)}\n\n".encode())

    monkeypatch.setattr(service, "get_effective_chatgpt_auth", fake_auth)
    monkeypatch.setattr(service, "send_upstream_request", fake_send)
    app = app_module.create_app(settings_module.Settings(reasoning_replay=True, api_key=None, verbose=False))
    with TestClient(app) as http_client:
        sdk = OpenAI(api_key="test-proxy-key", base_url="http://testserver/v1", http_client=http_client)
        history: list[dict[str, Any]] = [{"role": "user", "content": "Look up the key."}]
        if protocol == "responses":
            tools = [{"type": "function", "name": "lookup", "parameters": {"type": "object", "properties": {"key": {"type": "string"}}}}]
            first = sdk.responses.create(model="gpt-5.6-sol", input=history, tools=tools, store=False)
            history.extend(item.model_dump(exclude_none=True) for item in first.output)
            history.append({"type": "function_call_output", "call_id": "call_sdk", "output": "found"})
            second = sdk.responses.create(model="gpt-5.6-sol", input=history, tools=tools, store=False)
            assert second.output_text == "done"
        else:
            tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {"key": {"type": "string"}}}}}]
            first = sdk.chat.completions.create(model="gpt-5.6-sol", messages=history, tools=tools)
            history.append(first.choices[0].message.model_dump(exclude_none=True))
            history.append({"role": "tool", "tool_call_id": "call_sdk", "content": "found"})
            second = sdk.chat.completions.create(model="gpt-5.6-sol", messages=history, tools=tools)
            assert second.choices[0].message.content == "done"
    replayed = captured[1]["input"]
    assert next(item for item in replayed if item.get("type") == "reasoning") == reasoning
    assert [item["type"] for item in replayed if item.get("type") in ("reasoning", "function_call", "function_call_output")] == ["reasoning", "function_call", "function_call_output"]
    assert next(item for item in replayed if item.get("type") == "function_call")["call_id"] == "call_sdk"
