from __future__ import annotations

import os
import shutil
import socket
from collections.abc import Generator
from pathlib import Path

import httpx
import pytest
from starlette.testclient import TestClient

from gptmock.app import create_app
from gptmock.core.badges import update_gist_badges
from gptmock.infra.auth import read_auth_file
from gptmock.services.model_registry import get_model_list

TEST_PROMPT = "Say 'hello' and nothing else."
TIMEOUT = 120

_INTEGRATION_TEST_FILES = {
    "test_rest_api.py",
    "test_openai_client.py",
    "test_responses_api.py",
    "test_ollama_client.py",
    "test_langchain_client.py",
}


def _get_all_models() -> list[str]:
    return get_model_list(expose_reasoning=False)


ALL_MODELS: list[str] = _get_all_models()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-gist",
        action="store_true",
        default=False,
        help="Update gist badges after a full tests run",
    )


def _is_full_tests_run(config: pytest.Config) -> bool:
    args = [str(arg).rstrip("/") for arg in config.args]
    return args in ([], ["tests"])


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if not session.config.getoption("update_gist"):
        return
    if not _is_full_tests_run(session.config):
        terminal = session.config.pluginmanager.get_plugin("terminalreporter")
        if terminal is not None:
            terminal.write_line("Skipping gist update for non-full test target")
        return

    terminal = session.config.pluginmanager.get_plugin("terminalreporter")
    stats = getattr(terminal, "stats", {}) if terminal is not None else {}
    passed = len(stats.get("passed", []))
    failed = len(stats.get("failed", [])) + len(stats.get("error", []))
    skipped = len(stats.get("skipped", []))
    collected = session.testscollected or passed + failed + skipped
    ran = collected - skipped
    tests_pct = round(passed / ran * 100) if ran > 0 else 0
    update_gist_badges(
        tests_label="tests",
        tests_pct=tests_pct,
        tests_collected=collected,
        tests_skipped=skipped,
    )


@pytest.fixture(scope="session", autouse=True)
def isolated_gptmock_home(tmp_path_factory: pytest.TempPathFactory) -> Generator[Path]:
    tmp_home = tmp_path_factory.mktemp("gptmock")
    previous = {
        name: os.environ.get(name)
        for name in ("GPTMOCK_HOME", "CHATGPT_LOCAL_HOME", "CODEX_HOME")
    }
    os.environ["GPTMOCK_HOME"] = str(tmp_home)
    os.environ.pop("CHATGPT_LOCAL_HOME", None)
    os.environ.pop("CODEX_HOME", None)

    if os.getenv("GPTMOCK_RUN_LIVE_TESTS") == "1":
        source = os.getenv("GPTMOCK_TEST_AUTH_FILE")
        if source:
            source_path = Path(source)
            if source_path.is_file():
                shutil.copy2(source_path, tmp_home / "auth.json")

    yield tmp_home

    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


@pytest.fixture(autouse=True)
def block_network_in_offline_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail closed if a mock is lost; live calls require explicit opt-in."""
    if os.getenv("GPTMOCK_RUN_LIVE_TESTS") == "1":
        return

    def blocked_http(*args, **kwargs):
        raise AssertionError("Real HTTP transport disabled in offline tests; use MockTransport or ASGITransport")

    async def blocked_async_http(*args, **kwargs):
        blocked_http()

    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def check_address(address):
        if isinstance(address, tuple) and address[0] not in ("127.0.0.1", "::1", "localhost"):
            raise AssertionError("External socket connection disabled in offline tests")

    def guarded_connect(sock, address):
        check_address(address)
        return original_connect(sock, address)

    def guarded_connect_ex(sock, address):
        check_address(address)
        return original_connect_ex(sock, address)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", blocked_http)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", blocked_async_http)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)


@pytest.fixture(autouse=True)
def skip_without_auth(request: pytest.FixtureRequest) -> None:
    filename = os.path.basename(str(request.fspath))
    if filename in _INTEGRATION_TEST_FILES:
        live_enabled = os.getenv("GPTMOCK_RUN_LIVE_TESTS") == "1"
        if not live_enabled or read_auth_file() is None:
            pytest.skip(
                "Live tests require GPTMOCK_RUN_LIVE_TESTS=1 and GPTMOCK_TEST_AUTH_FILE",
            )


@pytest.fixture(scope="session")
def client() -> Generator[TestClient]:
    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture(scope="session")
def all_models() -> list[str]:
    return ALL_MODELS
