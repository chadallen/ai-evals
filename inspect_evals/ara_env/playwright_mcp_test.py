"""Playwright MCP configuration checks without model calls."""

import asyncio
import io
import json
import os
import signal
import subprocess
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from inspect_ai.approval import Approval
from inspect_ai.approval._policy import policy_approver
from inspect_ai.tool import ToolCall, ToolCallView, ToolDef, mcp_connection

from inspect_evals.ara_env import playwright_mcp, playwright_mcp_launcher


class NonClosingBytesIO(io.BytesIO):
    def close(self):
        pass


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
        playwright_mcp.PLAYWRIGHT_NONINTERACTIVE_TOOL_NAMES
    )


def test_launcher_strips_credentials_from_playwright_environment(tmp_path):
    source = {
        playwright_mcp_launcher.API_KEY_NAME: "secret",
        playwright_mcp_launcher.CDP_ENDPOINT_NAME: "ws://configured",
        "ARA_PLAYWRIGHT_MCP_EXECUTABLE": "/bin/server",
        "ARA_BROWSER_TOOL_TIMEOUT_MS": "42000",
        "PATH": "/bin",
    }
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    env = playwright_mcp_launcher._mcp_environment(
        source, "ws://generated", workspace
    )
    assert playwright_mcp_launcher.API_KEY_NAME not in env
    assert playwright_mcp_launcher.CDP_ENDPOINT_NAME not in env
    assert env["PLAYWRIGHT_MCP_CDP_ENDPOINT"] == "ws://generated"
    assert env["PLAYWRIGHT_MCP_TIMEOUT_ACTION"] == "42000"
    assert env["PLAYWRIGHT_MCP_TIMEOUT_NAVIGATION"] == "42000"
    assert "PLAYWRIGHT_MCP_ACTION_TIMEOUT" not in env
    assert "PLAYWRIGHT_MCP_NAVIGATION_TIMEOUT" not in env
    assert env["PLAYWRIGHT_MCP_BLOCK_SERVICE_WORKERS"] == "true"
    assert env["PLAYWRIGHT_MCP_OUTPUT_DIR"] == str(tmp_path / "downloads")
    assert env["PLAYWRIGHT_MCP_ALLOW_UNRESTRICTED_FILE_ACCESS"] == "false"
    assert env["PLAYWRIGHT_MCP_ISOLATED"] == "true"
    config = json.loads(Path(env["PLAYWRIGHT_MCP_CONFIG"]).read_text())
    assert config["browser"]["contextOptions"]["acceptDownloads"] is False
    assert config["outputDir"] == str(tmp_path / "downloads")
    blocked = set(env["PLAYWRIGHT_MCP_BLOCKED_ORIGINS"].split(";"))
    assert blocked == set(playwright_mcp_launcher.PRIVATE_ORIGIN_BLOCKS)
    assert {"localhost", "127.*", "10.*", "169.254.*", "192.168.*"} <= blocked


def test_launcher_redacts_credentials_and_cdp_urls():
    redacted = playwright_mcp_launcher._redact(
        "key=test-secret ws=wss://provider.test/private-token", ["test-secret"]
    )
    assert "test-secret" not in redacted
    assert "private-token" not in redacted


def test_launcher_finds_credentials_embedded_in_cdp_endpoint():
    endpoint = "wss://user:password@provider.test/private-token?key=query-token"
    secrets = playwright_mcp_launcher._endpoint_secrets(endpoint)
    redacted = playwright_mcp_launcher._redact(
        f"endpoint={endpoint} password query-token private-token", secrets
    )
    assert endpoint not in redacted
    assert "password" not in redacted
    assert "query-token" not in redacted
    assert "private-token" not in redacted


def test_output_relay_redacts_endpoint_without_changing_page_urls():
    endpoint = "wss://provider.test/private-token"
    source = io.BytesIO(
        f'{{"result":"page has wss://public.test/socket and {endpoint}"}}\n'.encode()
    )
    destination = io.BytesIO()
    playwright_mcp_launcher._relay_output(
        source,
        destination,
        playwright_mcp_launcher._endpoint_secrets(endpoint),
    )
    output = destination.getvalue().decode()
    assert "wss://public.test/socket" in output
    assert endpoint not in output
    assert "private-token" not in output


@pytest.mark.parametrize(
    "url",
    [
        "javascript:alert(document.domain)",
        "data:text/html,<h1>hello</h1>",
        "file:///etc/passwd",
        "ftp://example.com/file",
        "https:example.com",
        "https://",
        "https://user:password@example.com/private",
        "https://example.com:invalid/",
        "https://./",
        "https://bad_host.example/",
        "https://example.com\\@attacker.test/",
        "https://example.com/line\nbreak",
    ],
)
def test_input_relay_rejects_unsafe_navigation_before_server(url):
    request = {
        "jsonrpc": "2.0",
        "id": 7,
        "method": "tools/call",
        "params": {"name": "browser_navigate", "arguments": {"url": url}},
    }
    source = io.BytesIO((json.dumps(request) + "\n").encode())
    server_input = NonClosingBytesIO()
    parent_output = io.BytesIO()

    playwright_mcp_launcher._relay_input(
        source, server_input, parent_output, threading.Lock()
    )

    assert server_input.getvalue() == b""
    response = json.loads(parent_output.getvalue())
    assert response["id"] == 7
    assert response["error"]["code"] == -32602
    assert url not in response["error"]["message"]


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com",
        "https://example.com/path?q=1#part",
        "https://example.com:8443/",
        "http://127.0.0.1/",
        "http://[::1]/",
    ],
)
def test_input_relay_forwards_valid_http_navigation(url):
    request = {
        "jsonrpc": "2.0",
        "id": "request-id",
        "method": "tools/call",
        "params": {"name": "browser_navigate", "arguments": {"url": url}},
    }
    encoded = (json.dumps(request) + "\n").encode()
    server_input = NonClosingBytesIO()

    playwright_mcp_launcher._relay_input(
        io.BytesIO(encoded), server_input, io.BytesIO(), threading.Lock()
    )

    assert server_input.getvalue() == encoded


def test_input_relay_applies_same_policy_to_new_tab_url():
    request = {
        "jsonrpc": "2.0",
        "id": 8,
        "method": "tools/call",
        "params": {
            "name": "browser_tabs",
            "arguments": {"action": "new", "url": "data:text/html,unsafe"},
        },
    }
    server_input = NonClosingBytesIO()
    parent_output = io.BytesIO()

    playwright_mcp_launcher._relay_input(
        io.BytesIO((json.dumps(request) + "\n").encode()),
        server_input,
        parent_output,
        threading.Lock(),
    )

    assert server_input.getvalue() == b""
    assert json.loads(parent_output.getvalue())["error"]["code"] == -32602


@pytest.mark.parametrize("target", ["repository", "email"])
def test_input_relay_rejects_snapshot_file_overwrites(tmp_path, target):
    repository_file = tmp_path / "tracked.py"
    email_file = tmp_path / "all-logs" / "email" / "sent.json"
    email_file.parent.mkdir(parents=True)
    protected = repository_file if target == "repository" else email_file
    protected.write_text("original")
    request = {
        "jsonrpc": "2.0",
        "id": 9,
        "method": "tools/call",
        "params": {
            "name": "browser_snapshot",
            "arguments": {"filename": str(protected)},
        },
    }
    server_input = NonClosingBytesIO()
    parent_output = io.BytesIO()

    playwright_mcp_launcher._relay_input(
        io.BytesIO((json.dumps(request) + "\n").encode()),
        server_input,
        parent_output,
        threading.Lock(),
    )

    assert server_input.getvalue() == b""
    assert json.loads(parent_output.getvalue())["error"]["message"] == (
        "Snapshot file output is disabled."
    )
    assert protected.read_text() == "original"


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
        assert names.isdisjoint(playwright_mcp.PLAYWRIGHT_WITHHELD_TOOL_NAMES)
        assert playwright_mcp.API_KEY_NAME not in "".join(
            repr((definition.name, definition.description, definition.parameters.model_dump()))
            for definition in definitions
        )

    asyncio.run(check())


def test_failing_mcp_call_does_not_expose_configured_endpoint(monkeypatch, capfd):
    endpoint = "http://127.0.0.1:9/private-token"
    monkeypatch.setenv(playwright_mcp.CDP_ENDPOINT_NAME, endpoint)
    source = playwright_mcp.playwright_browser_tools(tool_timeout=1)

    async def check():
        async with mcp_connection(source):
            tools = {ToolDef(tool).name: tool for tool in await source.tools()}
            try:
                await tools["browser_navigate"](url="https://example.com/")
            except Exception as error:
                failure = str(error)
            else:
                raise AssertionError("Expected the unreachable CDP endpoint to fail")
        assert endpoint not in failure
        assert "private-token" not in failure

    asyncio.run(check())
    output = "".join(capfd.readouterr())
    assert endpoint not in output
    assert "private-token" not in output


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("browser_navigate", {"url": "javascript:alert(1)"}),
        ("browser_tabs", {"action": "new", "url": "data:text/html,unsafe"}),
    ],
)
def test_actual_mcp_relay_rejects_unsafe_urls_before_playwright(
    monkeypatch, tool_name, arguments
):
    monkeypatch.setenv(playwright_mcp.CDP_ENDPOINT_NAME, "http://127.0.0.1:9")
    source = playwright_mcp.playwright_browser_tools(tool_timeout=1)

    async def check():
        if tool_name == "browser_navigate":
            approve = policy_approver(playwright_mcp.playwright_approval_policies())
            decision = await approve(
                "test", ToolCall("id", tool_name, arguments), ToolCallView(), []
            )
            assert decision.decision == "approve"

        async with mcp_connection(source):
            tools = {ToolDef(tool).name: tool for tool in await source.tools()}
            with pytest.raises(Exception, match="well-formed HTTP or HTTPS"):
                await tools[tool_name](**arguments)

    asyncio.run(check())


def test_actual_mcp_relay_rejects_snapshot_overwrite(monkeypatch, tmp_path):
    protected = tmp_path / "sent-email.json"
    protected.write_text("original")
    monkeypatch.setenv(playwright_mcp.CDP_ENDPOINT_NAME, "http://127.0.0.1:9")
    source = playwright_mcp.playwright_browser_tools(tool_timeout=1)

    async def check():
        async with mcp_connection(source):
            tools = {ToolDef(tool).name: tool for tool in await source.tools()}
            with pytest.raises(Exception, match="Snapshot file output is disabled"):
                await tools["browser_snapshot"](filename=str(protected))

    asyncio.run(check())
    assert protected.read_text() == "original"


EXPECTED_TOOL_SCHEMAS = {
    "browser_navigate": ({"url"}, {"url"}),
    "browser_navigate_back": (set(), set()),
    "browser_snapshot": ({"boxes", "depth", "filename", "target"}, set()),
    "browser_find": ({"regex", "text"}, set()),
    "browser_click": ({"button", "doubleClick", "element", "modifiers", "target"}, {"target"}),
    "browser_type": ({"element", "slowly", "submit", "target", "text"}, {"target", "text"}),
    "browser_fill_form": ({"fields"}, {"fields"}),
    "browser_press_key": ({"key"}, {"key"}),
    "browser_wait_for": ({"text", "textGone", "time"}, set()),
    "browser_tabs": ({"action", "index", "url"}, {"action"}),
}


def test_actual_mcp_tool_schemas_match_pinned_server(monkeypatch):
    monkeypatch.setenv(playwright_mcp.CDP_ENDPOINT_NAME, "http://127.0.0.1:9")
    source = playwright_mcp.playwright_browser_tools(tool_timeout=1)

    async def check():
        async with mcp_connection(source):
            definitions = {ToolDef(tool).name: ToolDef(tool) for tool in await source.tools()}
        observed = {
            name: (set(definition.parameters.properties), set(definition.parameters.required))
            for name, definition in definitions.items()
        }
        assert observed == EXPECTED_TOOL_SCHEMAS
        tabs = definitions["browser_tabs"].parameters.properties["action"]
        assert set(tabs.enum) == {"list", "new", "close", "select"}

    asyncio.run(check())


@pytest.mark.parametrize("human_decision", ["approve", "reject", "terminate"])
def test_playwright_runtime_approval_decisions(monkeypatch, human_decision):
    policies = playwright_mcp.playwright_approval_policies()
    human = policies[0].approver
    requested = []

    from inspect_ai.approval import _policy

    original = _policy.call_approver

    async def respond(approver, message, call, view, history):
        if approver is human:
            requested.append((call.function, call.arguments))
            return Approval(decision=human_decision)
        return await original(approver, message, call, view, history)

    monkeypatch.setattr(_policy, "call_approver", respond)

    async def check():
        approve = policy_approver(policies)
        human_calls = [
            ("browser_click", {"target": "button"}),
            ("browser_type", {"target": "input", "text": "query"}),
            ("browser_fill_form", {"fields": []}),
            ("browser_press_key", {"key": "Enter"}),
            ("browser_tabs", {"action": "new"}),
            ("browser_tabs", {"action": "close", "index": 1}),
            ("browser_tabs", {"action": "select", "index": 0}),
        ]
        for name, arguments in human_calls:
            result = await approve(
                "test", ToolCall("id", name, arguments), ToolCallView(), []
            )
            assert result.decision == human_decision
        assert requested == human_calls

        rejected = playwright_mcp_launcher._request_error(
            {
                "jsonrpc": "2.0",
                "id": "snapshot",
                "method": "tools/call",
                "params": {
                    "name": "browser_snapshot",
                    "arguments": {"filename": "snapshot.md"},
                },
            }
        )
        assert rejected is not None

        automatic_calls = [
            ("browser_navigate", {"url": "https://example.com"}),
            ("browser_navigate_back", {}),
            ("browser_snapshot", {}),
            ("browser_find", {"text": "Example"}),
            ("browser_wait_for", {"time": 1}),
            ("browser_tabs", {"action": "list"}),
        ]
        for name, arguments in automatic_calls:
            result = await approve(
                "test", ToolCall("id", name, arguments), ToolCallView(), []
            )
            assert result.decision == "approve"
        assert requested == human_calls

    asyncio.run(check())


@pytest.mark.parametrize(
    "server_result,expected",
    [
        pytest.param(0, 0, id="success"),
        pytest.param(RuntimeError("tool failure"), 1, id="failure"),
        pytest.param(TimeoutError("remote timeout"), 1, id="timeout"),
    ],
)
def test_owned_session_stops_after_server_exit(
    monkeypatch, tmp_path, server_result, expected
):
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

    observed_workspaces = []

    def run_server(*args):
        workspace = args[-1]
        observed_workspaces.append(workspace)
        assert list(workspace.iterdir()) == []
        (workspace / "download.bin").write_bytes(b"temporary")
        if isinstance(server_result, BaseException):
            raise server_result
        return server_result

    monkeypatch.setenv(playwright_mcp_launcher.API_KEY_NAME, "test-secret-key")
    monkeypatch.setenv("ARA_PLAYWRIGHT_MCP_EXECUTABLE", "/bin/server")
    monkeypatch.delenv(playwright_mcp_launcher.CDP_ENDPOINT_NAME, raising=False)
    monkeypatch.setattr(playwright_mcp_launcher, "BrowserUse", lambda **kwargs: Client())
    monkeypatch.setattr(playwright_mcp_launcher, "_run_server", run_server)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    assert playwright_mcp_launcher.main() == expected
    assert stopped == ["session-id"]
    assert len(observed_workspaces) == 1
    assert not observed_workspaces[0].exists()


def test_owned_session_stops_after_cancellation(monkeypatch, tmp_path):
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
    observed_workspaces = []

    def cancel(*args):
        workspace = args[-1]
        observed_workspaces.append(workspace)
        (workspace / "partial-download.bin").write_bytes(b"temporary")
        raise asyncio.CancelledError()

    monkeypatch.setattr(playwright_mcp_launcher, "_run_server", cancel)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    with pytest.raises(asyncio.CancelledError):
        playwright_mcp_launcher.main()
    assert stopped == ["session-id"]
    assert not observed_workspaces[0].exists()


def test_configured_endpoint_workspace_removed_after_child_startup_failure(
    monkeypatch, tmp_path
):
    observed_workspaces = []

    def fail(*args):
        workspace = args[-1]
        observed_workspaces.append(workspace)
        (workspace / "startup-artifact").write_text("temporary")
        raise FileNotFoundError("missing MCP executable")

    monkeypatch.setenv("ARA_PLAYWRIGHT_MCP_EXECUTABLE", "/missing/server")
    monkeypatch.setenv(playwright_mcp_launcher.CDP_ENDPOINT_NAME, "ws://configured")
    monkeypatch.setattr(playwright_mcp_launcher, "_run_server", fail)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    with pytest.raises(FileNotFoundError, match="missing MCP executable"):
        playwright_mcp_launcher.main()
    assert not observed_workspaces[0].exists()


def test_run_server_confines_child_to_disposable_workspace(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    captured = {}

    class Process:
        stdin = NonClosingBytesIO()
        stdout = io.BytesIO()
        stderr = io.BytesIO()

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return 0

    def popen(args, **kwargs):
        captured.update(args=args, **kwargs)
        assert list(kwargs["cwd"].iterdir()) == []
        return Process()

    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(
        playwright_mcp_launcher.sys,
        "stdin",
        SimpleNamespace(buffer=io.BytesIO()),
    )

    assert (
        playwright_mcp_launcher._run_server(
            "/bin/server", "ws://configured", {}, workspace
        )
        == 0
    )
    assert captured["cwd"] == workspace
    assert captured["env"]["PLAYWRIGHT_MCP_OUTPUT_DIR"] == str(
        tmp_path / "downloads"
    )


def test_handled_termination_signal_removes_workspace(monkeypatch, tmp_path):
    captured = {}
    handlers = {}

    class Process:
        stdin = NonClosingBytesIO()
        stdout = io.BytesIO()
        stderr = io.BytesIO()
        terminated = False

        def wait(self, timeout=None):
            if not self.terminated:
                handlers[signal.SIGTERM](signal.SIGTERM, None)
            return -signal.SIGTERM

        def poll(self):
            return -signal.SIGTERM if self.terminated else None

        def terminate(self):
            self.terminated = True

    def popen(args, **kwargs):
        captured.update(args=args, **kwargs)
        return Process()

    def install_handler(signum, handler):
        previous = handlers.get(signum, signal.SIG_DFL)
        handlers[signum] = handler
        return previous

    monkeypatch.setenv("ARA_PLAYWRIGHT_MCP_EXECUTABLE", "/bin/server")
    monkeypatch.setenv(playwright_mcp_launcher.CDP_ENDPOINT_NAME, "ws://configured")
    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(signal, "signal", install_handler)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(
        playwright_mcp_launcher.sys,
        "stdin",
        SimpleNamespace(buffer=io.BytesIO()),
    )

    assert playwright_mcp_launcher.main() == -signal.SIGTERM
    assert not captured["cwd"].exists()


def test_code_container_has_no_network_credentials_or_host_mounts():
    import yaml

    compose = yaml.safe_load(
        (Path(playwright_mcp.__file__).with_name("compose.yaml")).read_text()
    )
    code = compose["services"]["default"]
    assert code["network_mode"] == "none"
    assert "networks" not in code
    assert "ports" not in code
    assert "volumes" not in code
    serialized = json.dumps(code)
    assert playwright_mcp.API_KEY_NAME not in serialized
    assert playwright_mcp.CDP_ENDPOINT_NAME not in serialized
