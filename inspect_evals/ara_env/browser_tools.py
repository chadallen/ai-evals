"""Keep Inspect browser discovery inside its dedicated sandbox."""

from inspect_ai.tool import Tool, ToolDef, web_browser
from inspect_ai.util import sandbox
from inspect_ai.util._sandbox.context import (
    sandbox_environments_context_var,
    sandbox_with_environments_context_var,
)


def browser_tools(interactive: bool = True) -> list[Tool]:
    def route(original: Tool) -> Tool:
        async def execute(*args, **kwargs):
            # Context-local routing also prevents a code-created executable from winning discovery.
            environments = sandbox_environments_context_var.set({"browser": sandbox("browser")})
            discovery = sandbox_with_environments_context_var.set({})
            try:
                return await original(*args, **kwargs)
            finally:
                sandbox_with_environments_context_var.reset(discovery)
                sandbox_environments_context_var.reset(environments)

        definition = ToolDef(original)
        definition.tool = execute
        return definition.as_tool()

    return [route(tool) for tool in web_browser(interactive=interactive)]
