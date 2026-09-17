"""Direct Browser Use Cloud and Playwright MCP checks with no solver model."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv
from inspect_ai.tool import ToolDef, mcp_connection

from inspect_evals.ara_env.playwright_mcp import (
    API_KEY_NAME,
    PLAYWRIGHT_BROWSER_TOOL_NAMES,
    REPO_ROOT,
    playwright_browser_tools,
)

BLOCK_MARKERS = (
    "unusual traffic",
    "verify you are human",
    "captcha",
    "access denied",
    "automated queries",
    "security check",
    "google.com/sorry/",
)
LINKEDIN_DISMISS_LABELS = ("dismiss", "close", "not now", "×")


def access_result(name: str, requested_url: str, response: str) -> dict[str, object]:
    """Summarize a snapshot without recording its full page contents."""
    normalized = " ".join(response.split())
    lowered = normalized.lower()
    markers = [marker for marker in BLOCK_MARKERS if marker in lowered]
    if name == "linkedin_public" and any(
        marker in lowered
        for marker in ("sign up | linkedin", "join linkedin", "linkedin.com/authwall")
    ):
        markers.append("linkedin sign-up interstitial remains after dismissal")
    return {
        "name": name,
        "requested_url": requested_url,
        "response_characters": len(response),
        "block_markers": markers,
        "usable": len(response) >= 200 and not markers,
        "text_preview": normalized[:300],
    }


async def blocked_probe(navigate, url: str) -> dict[str, object]:
    """Record whether Playwright MCP rejected a non-public target."""
    try:
        response = str(await navigate(url=url))
    except Exception as error:
        return {"url": url, "blocked": True, "result": str(error)[:200]}
    lowered = response.lower()
    blocked = any(word in lowered for word in ("blocked", "denied", "error", "failed"))
    return {"url": url, "blocked": blocked, "result": response[:200]}


def linkedin_dismiss_targets(snapshot: str) -> list[tuple[str, str]]:
    """Extract refs for visible LinkedIn marketing-prompt controls."""
    targets = []
    for line in snapshot.splitlines():
        lowered = line.lower()
        if not any(label in lowered for label in LINKEDIN_DISMISS_LABELS):
            continue
        match = re.search(r"\[ref=([^\]]+)\]", line)
        if match:
            targets.append((match.group(1), line.strip()[:100]))
    return targets


async def smoke(query: str, linkedin_url: str) -> dict[str, object]:
    started = time.monotonic()
    source = playwright_browser_tools(tool_timeout=30)
    pages = []
    private_targets = []
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
        example = str(await tools["browser_snapshot"]())
        await tools["browser_navigate_back"]()
        returned = str(await tools["browser_snapshot"]())
        second_cookie_check = str(await tools["browser_find"](text="Cookies are enabled"))

        assert "Example Domain" in example, example[:1_000]
        assert "Cookie checker" in first, first[:1_000]
        assert "Cookie checker" in returned, returned[:1_000]
        assert "Cookies are enabled" in first_cookie_check, first_cookie_check[:1_000]
        assert "Cookies are enabled" in second_cookie_check, second_cookie_check[:1_000]

        urls = (
            ("google_search", f"https://www.google.com/search?q={quote_plus(query)}"),
            ("bing_search", f"https://www.bing.com/search?q={quote_plus(query)}"),
            ("linkedin_public", linkedin_url),
        )
        for name, url in urls:
            try:
                response = str(await tools["browser_navigate"](url=url))
                dismissal = None
                if name == "linkedin_public":
                    attempted = []
                    dismissed = False
                    for target, label in linkedin_dismiss_targets(response):
                        attempted.append(label)
                        try:
                            await tools["browser_click"](
                                target=target,
                                element=label,
                            )
                            dismissed = True
                            break
                        except Exception:
                            continue
                    if not dismissed:
                        attempted.append("Escape")
                        try:
                            await tools["browser_press_key"](key="Escape")
                        except Exception:
                            pass
                    response = str(await tools["browser_snapshot"]())
                    dismissal = {"attempted": attempted, "dismissed": dismissed}
                result = access_result(name, url, response)
                if dismissal is not None:
                    result["marketing_prompt"] = dismissal
                pages.append(result)
            except Exception as error:
                pages.append(
                    {
                        "name": name,
                        "requested_url": url,
                        "usable": False,
                        "error": str(error)[:500],
                    }
                )

        for url in (
            "file:///etc/passwd",
            "http://127.0.0.1:9/",
            "http://169.254.169.254/latest/meta-data/",
            "http://host.docker.internal/",
        ):
            private_targets.append(await blocked_probe(tools["browser_navigate"], url))

    assert all(target["blocked"] for target in private_targets), private_targets
    return {
        "run_at": datetime.now(UTC).isoformat(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "same_session_state_preserved": True,
        "pages": pages,
        "private_targets": private_targets,
        "session_closed": True,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="Chad Allen Google product manager GitHub")
    parser.add_argument("--linkedin-url", default="https://www.linkedin.com/in/allenchad")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scratch/playwright-mcp-results.json"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    load_dotenv(REPO_ROOT / ".env")
    if not os.environ.get(API_KEY_NAME):
        print(f"Missing {API_KEY_NAME}", file=sys.stderr)
        return 2
    report = asyncio.run(smoke(args.query, args.linkedin_url))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
