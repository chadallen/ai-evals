"""Build-time patch for the pinned Inspect browser launcher."""

from pathlib import Path

path = next(
    Path("/opt/inspect").rglob(
        "inspect_tool_support/_remote_tools/_web_browser/playwright_browser.py"
    )
)
source = path.read_text()
needle = "                headless=headless,"
assert source.count(needle) == 1, "Pinned browser launcher changed; review proxy configuration"
source = source.replace(
    needle,
    needle
    + '\n                proxy=None if getenv("ARA_LOCAL_PREVIEW") == "1" '
    + 'else {"server": "http://egress:3128", "bypass": "<-loopback>"},',
)
path.write_text(source)
