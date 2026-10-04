from __future__ import annotations

import os
import socket

import httpx
import pytest

pytestmark = pytest.mark.skipif(os.getenv("GPTMOCK_RUN_LIVE_TESTS") == "1", reason="Tests the offline-only guard")


def test_real_sync_http_transport_is_blocked() -> None:
    with pytest.raises(AssertionError, match="Real HTTP transport disabled"):
        with httpx.Client(trust_env=False) as client:
            client.get("http://192.0.2.1")


@pytest.mark.asyncio
async def test_real_async_http_transport_is_blocked() -> None:
    with pytest.raises(AssertionError, match="Real HTTP transport disabled"):
        async with httpx.AsyncClient(trust_env=False) as client:
            await client.get("http://192.0.2.1")


@pytest.mark.parametrize("method", ["connect", "connect_ex"])
def test_external_socket_is_blocked(method: str) -> None:
    with socket.socket() as sock, pytest.raises(AssertionError, match="External socket connection disabled"):
        getattr(sock, method)(("192.0.2.1", 80))


@pytest.mark.asyncio
async def test_mock_http_transport_remains_available() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"ok": True}))
    async with httpx.AsyncClient(transport=transport) as client:
        assert (await client.get("http://192.0.2.1")).json() == {"ok": True}
