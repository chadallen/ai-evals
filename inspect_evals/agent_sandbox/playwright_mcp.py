"""Playwright MCP tools backed by a replaceable remote CDP browser."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from pathlib import Path

from inspect_ai.approval import ApprovalPolicy, auto_approver, human_approver
from inspect_ai.solver import Generate, Solver, TaskState, generate, solver, use_tools
from inspect_ai.tool import Tool, ToolDef, ToolSource, mcp_connection, mcp_server_stdio, mcp_tools

API_KEY_NAME = "BROWSER_USE_API_KEY"
CDP_ENDPOINT_NAME = "ARA_BROWSER_CDP_ENDPOINT"
PLAYWRIGHT_MCP_VERSION = "0.0.81"

# These names are the policy boundary. Keep additions explicit rather than
# accepting new tools when Playwright MCP adds capabilities.
PLAYWRIGHT_BROWSER_TOOL_NAMES = (
    "browser_navigate",
    "browser_navigate_back",
    "browser_snapshot",
    "browser_find",
    "browser_click",
    "browser_type",
    "browser_fill_form",
    "browser_press_key",
    "browser_wait_for",
    "browser_tabs",
)
PLAYWRIGHT_NONINTERACTIVE_TOOL_NAMES = (
    "browser_navigate",
    "browser_navigate_back",
    "browser_snapshot",
    "browser_find",
    "browser_wait_for",
    "browser_tabs",
)

# These tools are present in the pinned server but withheld from the solver.
# The list documents capabilities whose accidental exposure would expand the
# containment boundary or allow the model to bypass the reviewed action set.
PLAYWRIGHT_WITHHELD_TOOL_NAMES = (
    "browser_evaluate",
    "browser_run_code",
    "browser_run_code_unsafe",
    "browser_file_upload",
    "browser_drag",
    "browser_route",
    "browser_route_remove",
    "browser_localstorage",
    "browser_sessionstorage",
    "browser_take_screenshot",
)

# Tabs need argument-aware rules: listing is passive, while opening, closing,
# and selecting changes browser state. The launcher rejects snapshot filenames
# before Playwright because approval alone cannot make arbitrary paths safe.
PLAYWRIGHT_HUMAN_APPROVAL_PATTERNS = (
    "browser_click",
    "browser_type",
    "browser_fill_form",
    "browser_press_key",
    "browser_tabs*action='new'*",
    "browser_tabs*action='close'*",
    "browser_tabs*action='select'*",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = Path(__file__).with_name("playwright_mcp_launcher.py")


def playwright_browser_tools(
    *,
    cdp_endpoint: str | None = None,
    tool_timeout: int = 120,
    interactive: bool = True,
) -> ToolSource:
    """Return the reviewed Playwright MCP browser tools.

    A configured endpoint selects any compatible CDP provider. Without one,
    the launcher creates and owns a Browser Use Cloud session.
    """
    endpoint = cdp_endpoint or os.environ.get(CDP_ENDPOINT_NAME)
    child_env = {
        "ARA_BROWSER_TOOL_TIMEOUT_MS": str(tool_timeout * 1_000),
        "ARA_PLAYWRIGHT_MCP_EXECUTABLE": str(
            REPO_ROOT / "node_modules" / ".bin" / "playwright-mcp"
        ),
    }
    if endpoint:
        child_env[CDP_ENDPOINT_NAME] = endpoint
    else:
        api_key = os.environ.get(API_KEY_NAME)
        if api_key:
            child_env[API_KEY_NAME] = api_key

    server = mcp_server_stdio(
        name="playwright-browser",
        command=sys.executable,
        args=[str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=child_env,
    )
    names = (
        PLAYWRIGHT_BROWSER_TOOL_NAMES
        if interactive
        else PLAYWRIGHT_NONINTERACTIVE_TOOL_NAMES
    )
    return mcp_tools(server, tools=list(names))


def playwright_approval_policies() -> list[ApprovalPolicy]:
    """Gate configured browser interactions and approve remaining calls."""
    return [
        ApprovalPolicy(
            approver=human_approver(),
            tools=list(PLAYWRIGHT_HUMAN_APPROVAL_PATTERNS),
        ),
        ApprovalPolicy(approver=auto_approver(), tools="*"),
    ]


@solver
def generate_with_tools(
    *tools: Tool | ToolDef | ToolSource | Sequence[Tool | ToolDef | ToolSource],
) -> Solver:
    """Resolve tools and run the model loop within one MCP connection."""
    flattened: list[Tool | ToolDef | ToolSource] = []
    for item in tools:
        if isinstance(item, Sequence):
            flattened.extend(item)
        else:
            flattened.append(item)

    install = use_tools(*flattened)
    run = generate()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        async with mcp_connection(flattened):
            state = await install(state, generate)
            return await run(state, generate)

    return solve
