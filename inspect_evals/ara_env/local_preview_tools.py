"""Browser tools confined to the generated site's preview network."""

from functools import wraps

from inspect_ai._util.content import ContentImage
from inspect_ai.tool import Tool, ToolDef, ToolError, web_browser
from inspect_ai.util import sandbox
from inspect_ai.util._sandbox.context import (
    sandbox_environments_context_var,
    sandbox_with_environments_context_var,
)

PREVIEW_SANDBOX = "preview"
PREVIEW_TOOL_PREFIX = "local_preview_"


def _route(original: Tool, *, name: str) -> Tool:
    @wraps(original)
    async def execute(*args, **kwargs):
        environments = sandbox_environments_context_var.set(
            {PREVIEW_SANDBOX: sandbox(PREVIEW_SANDBOX)}
        )
        discovery = sandbox_with_environments_context_var.set({})
        try:
            return await original(*args, **kwargs)
        finally:
            sandbox_with_environments_context_var.reset(discovery)
            sandbox_environments_context_var.reset(environments)

    definition = ToolDef(original)
    definition.name = name
    definition.description = (
        f"Local preview only. {definition.description} This browser has no public Internet route."
    )
    definition.tool = execute
    return definition.as_tool()


def local_preview_tools(interactive: bool = True) -> list[Tool]:
    """Return a stateful browser whose network contains only the generated site."""
    tools = web_browser(interactive=interactive, instance="local-preview")
    routed = []
    for original in tools:
        definition = ToolDef(original)
        suffix = definition.name.removeprefix("web_browser_")
        routed.append(_route(original, name=f"{PREVIEW_TOOL_PREFIX}{suffix}"))
    return [*routed, local_preview_screenshot()]


def local_preview_screenshot() -> Tool:
    """Create a screenshot tool that runs a fresh Chromium page in the preview sandbox."""

    async def execute(url: str) -> ContentImage:
        """Capture a full-page PNG from the isolated local preview browser.

        Args:
            url: Absolute HTTP or HTTPS URL reachable inside the preview network.

        Returns:
            The rendered page as a PNG image.
        """
        result = await sandbox(PREVIEW_SANDBOX).exec(
            [
                "/opt/inspect/pipx/venvs/inspect-tool-support/bin/python",
                "/opt/ara/preview_screenshot.py",
                url,
            ],
            timeout=30,
        )
        if not result.success:
            details = result.stderr.strip()[-500:]
            raise ToolError(f"Unable to capture local preview: {details or 'unknown error'}")
        encoded = result.stdout.strip()
        if not encoded:
            raise ToolError("Unable to capture local preview: screenshot was empty")
        return ContentImage(image=f"data:image/png;base64,{encoded}", detail="original")

    return ToolDef(
        execute,
        name="local_preview_screenshot",
        description=(
            "Capture a full-page screenshot using Chromium inside the isolated local preview "
            "network. The browser cannot reach the public Internet or host services."
        ),
    ).as_tool()
