"""Direct Browser Use Cloud and Playwright MCP check with no solver model."""

from __future__ import annotations

import asyncio
import os
import sys

from dotenv import load_dotenv
from inspect_ai.tool import ToolDef, mcp_connection

from inspect_evals.ara_env.playwright_mcp import (
    API_KEY_NAME,
    PLAYWRIGHT_BROWSER_TOOL_NAMES,
    REPO_ROOT,
    playwright_browser_tools,
)


async def smoke() -> None:
    source = playwright_browser_tools(tool_timeout=60)
    async with mcp_connection(source):
        tools = {ToolDef(tool).name: tool for tool in await source.tools()}
        assert set(tools) == set(PLAYWRIGHT_BROWSER_TOOL_NAMES)
        first = str(
            await tools["browser_navigate"](
                url="https://www.whatarecookies.com/cookietest.asp"
            )
        )
        first_cookie_check = str(await tools["browser_find"](text="Cookies are enabled"))
        await tools["browser_navigate"](url="https://example.com/")
        second = str(await tools["browser_snapshot"]())
        await tools["browser_navigate_back"]()
        third = str(await tools["browser_snapshot"]())
        second_cookie_check = str(await tools["browser_find"](text="Cookies are enabled"))

        assert "Example Domain" in second, second[:1_000]
        assert "Cookie checker" in first, first[:1_000]
        assert "Cookie checker" in third, third[:1_000]
        assert "Cookies are enabled" in first_cookie_check, first_cookie_check[:1_000]
        assert "Cookies are enabled" in second_cookie_check, second_cookie_check[:1_000]


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")
    if not os.environ.get(API_KEY_NAME):
        print(f"Missing {API_KEY_NAME}", file=sys.stderr)
        return 2
    asyncio.run(smoke())
    print("Playwright MCP Browser Use smoke check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
