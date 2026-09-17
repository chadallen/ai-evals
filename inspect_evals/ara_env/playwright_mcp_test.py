"""Playwright MCP configuration checks without model calls."""

import asyncio
import json
import os
from types import SimpleNamespace
from unittest.mock import patch

from inspect_ai.tool import ToolDef, mcp_connection

from inspect_evals.ara_env import playwright_mcp, playwright_mcp_launcher


def test_server_filters_tools_and_keeps_secrets_out_of_arguments(monkeypatch):
    monkeypatch.setenv(playwright_mcp.API_KEY_NAME, "test-secret-key")
    captured = {}
    server = object()
    source = object()

    def server_factory(**kwargs):
        captured.update(kwargs)
        return server

    with (
        patch.object(playwright_mcp, "mcp_server_stdio", side_effect=server_factory),
        patch.object(playwright_mcp, "mcp_tools", return_value=source) as select,
    ):
        assert playwright_mcp.playwright_browser_tools(tool_timeout=45) is source

    assert captured["command"] == os.sys.executable
    assert captured["args"] == [str(playwright_mcp.LAUNCHER)]
    assert "test-secret-key" not in " ".join(captured["args"])
    assert captured["env"][playwright_mcp.API_KEY_NAME] == "test-secret-key"
    assert captured["env"]["ARA_BROWSER_TOOL_TIMEOUT_MS"] == "45000"
    select.assert_called_once_with(
        server, tools=list(playwright_mcp.PLAYWRIGHT_BROWSER_TOOL_NAMES)
    )


def test_provider_endpoint_changes_without_changing_solver_tools(monkeypatch):
    monkeypatch.delenv(playwright_mcp.API_KEY_NAME, raising=False)
    with (
        patch.object(playwright_mcp, "mcp_server_stdio", return_value=object()) as server,
        patch.object(playwright_mcp, "mcp_tools", return_value=object()) as select,
    ):
        playwright_mcp.playwright_browser_tools(cdp_endpoint="ws://provider.test/session")

    assert server.call_args.kwargs["env"][playwright_mcp.CDP_ENDPOINT_NAME] == (
        "ws://provider.test/session"
    )
    assert playwright_mcp.API_KEY_NAME not in server.call_args.kwargs["env"]
    assert select.call_args.kwargs["tools"] == list(
        playwright_mcp.PLAYWRIGHT_BROWSER_TOOL_NAMES
    )


def test_noninteractive_selection_omits_mutating_actions():
    with (
        patch.object(playwright_mcp, "mcp_server_stdio", return_value=object()),
        patch.object(playwright_mcp, "mcp_tools", return_value=object()) as select,
    ):
        playwright_mcp.playwright_browser_tools(
            cdp_endpoint="ws://provider.test/session", interactive=False
        )
    assert select.call_args.kwargs["tools"] == list(
        playwright_mcp.PLAYWRIGHT_READ_ONLY_TOOL_NAMES
    )


def test_launcher_strips_credentials_from_playwright_environment():
    source = {
        playwright_mcp_launcher.API_KEY_NAME: "secret",
        playwright_mcp_launcher.CDP_ENDPOINT_NAME: "ws://configured",
        "ARA_PLAYWRIGHT_MCP_EXECUTABLE": "/bin/server",
        "ARA_BROWSER_TOOL_TIMEOUT_MS": "42000",
        "PATH": "/bin",
    }
    env = playwright_mcp_launcher._mcp_environment(source, "ws://generated")
    assert playwright_mcp_launcher.API_KEY_NAME not in env
    assert playwright_mcp_launcher.CDP_ENDPOINT_NAME not in env
    assert env["PLAYWRIGHT_MCP_CDP_ENDPOINT"] == "ws://generated"
    assert env["PLAYWRIGHT_MCP_ACTION_TIMEOUT"] == "42000"
    assert env["PLAYWRIGHT_MCP_NAVIGATION_TIMEOUT"] == "42000"


def test_launcher_redacts_credentials_and_cdp_urls():
    redacted = playwright_mcp_launcher._redact(
        "key=test-secret ws=wss://provider.test/private-token", ["test-secret"]
    )
    assert "test-secret" not in redacted
    assert "private-token" not in redacted


def test_node_dependency_matches_declared_server_version():
    package = json.loads((playwright_mcp.REPO_ROOT / "package.json").read_text())
    assert package["dependencies"]["@playwright/mcp"] == (
        playwright_mcp.PLAYWRIGHT_MCP_VERSION
    )


def test_launcher_stops_owned_browser_session(monkeypatch):
    stopped = []
    browsers = SimpleNamespace(
        create=lambda **kwargs: SimpleNamespace(id="session-id", cdp_url="ws://generated"),
        stop=lambda session_id: stopped.append(session_id),
    )

    class Client:
        def __enter__(self):
            return SimpleNamespace(browsers=browsers)

        def __exit__(self, *args):
            return None

    monkeypatch.setenv(playwright_mcp_launcher.API_KEY_NAME, "test-secret-key")
    monkeypatch.setenv("ARA_PLAYWRIGHT_MCP_EXECUTABLE", "/bin/server")
    monkeypatch.delenv(playwright_mcp_launcher.CDP_ENDPOINT_NAME, raising=False)
    monkeypatch.setattr(playwright_mcp_launcher, "BrowserUse", lambda **kwargs: Client())
    monkeypatch.setattr(playwright_mcp_launcher, "_run_server", lambda *args: 0)

    assert playwright_mcp_launcher.main() == 0
    assert stopped == ["session-id"]


def test_actual_mcp_server_exposes_only_reviewed_tools(monkeypatch):
    monkeypatch.setenv(
        playwright_mcp.CDP_ENDPOINT_NAME, "http://127.0.0.1:9"
    )
    source = playwright_mcp.playwright_browser_tools(tool_timeout=1)

    async def check():
        async with mcp_connection(source):
            definitions = [ToolDef(tool) for tool in await source.tools()]
            names = {definition.name for definition in definitions}
        assert names == set(playwright_mcp.PLAYWRIGHT_BROWSER_TOOL_NAMES)
        assert playwright_mcp.API_KEY_NAME not in "".join(
            repr((definition.name, definition.description, definition.parameters.model_dump()))
            for definition in definitions
        )

    asyncio.run(check())
