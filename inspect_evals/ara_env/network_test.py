"""Proxy policy and tool-routing checks; no model calls."""

import asyncio
import inspect
import socket
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml
from inspect_ai.tool import ToolDef
from inspect_ai.util._sandbox.context import (
    sandbox_environments_context_var,
    sandbox_with_environments_context_var,
)

from inspect_evals.ara_env import browser_tools, egress_proxy
from inspect_evals.ara_env import local_preview_tools as preview_module

COMPOSE = Path(__file__).with_name("compose.yaml")


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.1.2.3",
        "172.16.0.1",
        "192.168.1.1",
        "169.254.169.254",
        "0.0.0.0",
        "100.64.0.1",
        "224.0.0.1",
        "192.0.2.1",
        "::1",
        "fc00::1",
        "fe80::1",
        "::ffff:127.0.0.1",
        "2002:7f00:1::",
        "ff02::1",
    ],
)
def test_private_dns_answers_never_connect(address):
    async def check():
        loop = asyncio.get_running_loop()
        answers = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 80))]
        with (
            patch.object(loop, "getaddrinfo", AsyncMock(return_value=answers)),
            patch.object(loop, "sock_connect", AsyncMock()) as connect,
        ):
            with pytest.raises(ValueError, match="Non-public"):
                await egress_proxy.public_connection("rebind.test", 80)
            connect.assert_not_called()

    asyncio.run(check())


def test_checked_address_is_pinned_and_mixed_answers_fail():
    async def check():
        loop = asyncio.get_running_loop()
        public = (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80))
        private = (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))
        with (
            patch.object(
                loop, "getaddrinfo", AsyncMock(side_effect=[[public], [private]])
            ) as resolve,
            patch.object(loop, "sock_connect", AsyncMock()) as connect,
            patch.object(
                egress_proxy.asyncio, "open_connection", AsyncMock(return_value=(None, None))
            ),
        ):
            await egress_proxy.public_connection("rebind.test", 80)
            resolve.assert_awaited_once()
            sock, address = connect.await_args.args
            assert address == ("93.184.216.34", 80)
            sock.close()
            with pytest.raises(ValueError):
                await egress_proxy.public_connection("rebind.test", 80)
        with patch.object(loop, "getaddrinfo", AsyncMock(return_value=[public, private])):
            with pytest.raises(ValueError):
                await egress_proxy.public_connection("mixed.test", 80)

    asyncio.run(check())


def test_browser_discovery_cannot_select_code_or_egress():
    async def check():
        browser = object()
        seen = []

        async def original(url: str) -> str:
            """Navigate.

            Args:
                url: Address.
            """
            seen.append(sandbox_environments_context_var.get())
            assert sandbox_with_environments_context_var.get() == {}
            return url

        original_tool = ToolDef(original, name="web_browser_go").as_tool()
        env = sandbox_environments_context_var.set(
            {"default": object(), "browser": browser, "egress": object()}
        )
        cache = sandbox_with_environments_context_var.set({"poisoned": object()})
        try:
            with patch.object(browser_tools, "web_browser", return_value=[original_tool]):
                tool = browser_tools.browser_tools()[0]
            assert set(ToolDef(tool).parameters.properties) == {"url"}
            assert list(inspect.signature(ToolDef(tool).tool).parameters) == ["url"]
            assert await tool(url="https://example.com") == "https://example.com"
            assert seen == [{"browser": browser}]
            assert "default" in sandbox_environments_context_var.get()
            assert "poisoned" in sandbox_with_environments_context_var.get()
        finally:
            sandbox_environments_context_var.reset(env)
            sandbox_with_environments_context_var.reset(cache)

    asyncio.run(check())


def test_local_preview_tools_have_separate_names_and_route_only_to_preview():
    async def check():
        preview = object()
        seen = []

        async def original(url: str) -> str:
            """Navigate.

            Args:
                url: Address.
            """
            seen.append(sandbox_environments_context_var.get())
            return url

        original_tool = ToolDef(original, name="web_browser_go").as_tool()
        env = sandbox_environments_context_var.set(
            {"default": object(), "preview": preview, "browser": object()}
        )
        cache = sandbox_with_environments_context_var.set({"poisoned": object()})
        try:
            with (
                patch.object(preview_module, "web_browser", return_value=[original_tool]),
                patch.object(preview_module, "local_preview_screenshot", return_value=original_tool),
            ):
                tools = preview_module.local_preview_tools()
            definition = ToolDef(tools[0])
            assert definition.name == "local_preview_go"
            assert "no public Internet route" in definition.description
            assert await tools[0](url="http://default:8000/") == "http://default:8000/"
            assert seen == [{"preview": preview}]
            assert "default" in sandbox_environments_context_var.get()
            assert "poisoned" in sandbox_with_environments_context_var.get()
        finally:
            sandbox_environments_context_var.reset(env)
            sandbox_with_environments_context_var.reset(cache)

    asyncio.run(check())


def test_local_preview_screenshot_returns_image_and_redacts_failure(monkeypatch):
    async def check():
        success = AsyncMock(
            return_value=type("Result", (), {"success": True, "stdout": "cG5n\n", "stderr": ""})()
        )
        environment = type("Environment", (), {"exec": success})()
        monkeypatch.setattr(preview_module, "sandbox", lambda name: environment)
        tool = preview_module.local_preview_screenshot()
        image = await tool("http://default:8000/")
        assert image.image == "data:image/png;base64,cG5n"
        assert success.await_args.args[0] == [
            "/opt/inspect/pipx/venvs/inspect-tool-support/bin/python",
            "/opt/ara/preview_screenshot.py",
            "http://default:8000/",
        ]

        failure = AsyncMock(
            return_value=type(
                "Result",
                (),
                {"success": False, "stdout": "", "stderr": "browser failed"},
            )()
        )
        monkeypatch.setattr(
            preview_module,
            "sandbox",
            lambda name: type("Environment", (), {"exec": failure})(),
        )
        with pytest.raises(Exception, match="browser failed"):
            await preview_module.local_preview_screenshot()("http://default:8000/")

    asyncio.run(check())


def test_preview_compose_is_internal_and_published_only_on_loopback():
    config = yaml.safe_load(COMPOSE.read_text())
    default = config["services"]["default"]
    preview = config["services"]["preview"]
    network = config["networks"]["preview_internal"]

    publisher = config["services"]["preview_publish"]
    assert "ports" not in default
    assert publisher["ports"] == ["127.0.0.1:8000:8000"]
    assert default["networks"] == ["preview_internal"]
    assert preview["networks"] == ["preview_internal"]
    assert publisher["networks"] == ["preview_internal", "host_preview"]
    assert publisher["read_only"] is True
    assert network["internal"] is True
    assert network["driver_opts"]["com.docker.network.bridge.gateway_mode_ipv4"] == "isolated"
    assert "internet" not in default["networks"]
    assert "internet" not in preview["networks"]


def test_tavily_runs_on_host_with_mocked_http(monkeypatch):
    import httpx
    from inspect_ai.tool import web_search

    monkeypatch.setenv("TAVILY_API_KEY", "test-only-not-a-real-key")
    original_client = httpx.AsyncClient
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "query": "fixture",
                "answer": "Fixture answer",
                "images": [],
                "results": [],
                "response_time": 0.1,
            },
        )

    async def check():
        with patch(
            "httpx.AsyncClient",
            side_effect=lambda **kw: original_client(transport=httpx.MockTransport(respond), **kw),
        ):
            result = await web_search(providers="tavily")("fixture")
        assert "Fixture answer" in str(result)
        assert len(requests) == 1
        assert str(requests[0].url) == "https://api.tavily.com/search"

    asyncio.run(check())


@pytest.mark.parametrize("factory_name", ["ara_env", "spearphish"])
def test_task_code_tools_fix_destination(factory_name):
    from importlib import import_module
    from types import SimpleNamespace

    from inspect_ai.solver import TaskState

    factory = getattr(import_module(f"inspect_evals.ara_env.{factory_name}"), factory_name)

    async def check():
        with patch(
            f"inspect_evals.ara_env.{factory_name}.playwright_browser_tools",
            return_value=[],
        ):
            task = factory()
        state = TaskState(model="mockllm/model", sample_id=1, epoch=1, input="check", messages=[])
        async def no_generate(current, **kwargs):
            return current

        await task.solver[2](state, no_generate)
        tools = {ToolDef(t).name: t for t in state.tools}
        environment = SimpleNamespace(
            exec=AsyncMock(return_value=SimpleNamespace(stderr="", stdout="ok"))
        )
        with patch(
            "inspect_ai.tool._tools._execute.sandbox_env", return_value=environment
        ) as route:
            await tools["bash"]("id")
            await tools["python"]("print(1)")
            assert [call.args for call in route.call_args_list] == [("default",), ("default",)]
        with pytest.raises(TypeError):
            await tools["bash"]("id", sandbox="browser")
        with pytest.raises(TypeError):
            await tools["python"]("print(1)", sandbox="egress")

    asyncio.run(check())


@pytest.mark.parametrize(
    "raw_request",
    [
        b"CONNECT public.test:25 HTTP/1.1\r\n\r\n",
        b"CONNECT public.test:443/path HTTP/1.1\r\n\r\n",
        b"GET ftp://public.test/file HTTP/1.1\r\n\r\n",
        b"GET http://public.test:8080/ HTTP/1.1\r\n\r\n",
        b"GET http://user:pass@public.test/ HTTP/1.1\r\n\r\n",
        b"GET /relative HTTP/1.1\r\nHost: public.test\r\n\r\n",
        b"not an HTTP request\r\n\r\n",
    ],
)
def test_invalid_proxy_targets_do_not_connect(raw_request):
    async def check():
        reader = asyncio.StreamReader()
        reader.feed_data(raw_request)
        reader.feed_eof()
        from unittest.mock import Mock

        writer = Mock(wait_closed=AsyncMock())
        with patch.object(egress_proxy, "public_connection", AsyncMock()) as connect:
            await egress_proxy.handle(reader, writer)
        connect.assert_not_called()
        assert b"403 Forbidden" in writer.write.call_args.args[0]

    asyncio.run(check())
