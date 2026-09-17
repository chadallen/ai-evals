"""Direct Browser Use Cloud and Playwright MCP checks with no solver model."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

from browser_use_sdk.v4 import BrowserUse
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
PHASE_TIMEOUT_SECONDS = 35
TERMINATION_TIMEOUT_SECONDS = 15


def access_result(name: str, requested_url: str, response: str) -> dict[str, object]:
    """Summarize an inline snapshot without recording its full page contents."""
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


def redacted_error(error: BaseException, api_key: str) -> str:
    message = str(error).replace(api_key, "<redacted-api-key>")
    return re.sub(r"wss?://\S+", "<redacted-cdp-url>", message)[:500]


async def run_phase(
    name: str,
    operation: Callable[[], Awaitable[dict[str, Any]]],
    api_key: str,
    timeout: float = PHASE_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Bound one check so its failure cannot skip later checks."""
    started = time.monotonic()
    try:
        result = await asyncio.wait_for(operation(), timeout=timeout)
        return {
            "name": name,
            "ok": True,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            **result,
        }
    except Exception as error:
        return {
            "name": name,
            "ok": False,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "error": redacted_error(error, api_key),
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


async def live_site_check(tools: dict[str, Any], name: str, url: str) -> dict[str, Any]:
    """Navigate, then inspect only inline accessibility snapshots."""
    await tools["browser_navigate"](url=url)
    snapshot = str(await tools["browser_snapshot"]())
    dismissal = None
    if name == "linkedin_public":
        attempted = []
        dismissed = False
        for target, label in linkedin_dismiss_targets(snapshot):
            attempted.append(label)
            try:
                await tools["browser_click"](target=target, element=label)
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
        snapshot = str(await tools["browser_snapshot"]())
        dismissal = {"attempted": attempted, "dismissed": dismissed}

    result = access_result(name, url, snapshot)
    if dismissal is not None:
        result["marketing_prompt"] = dismissal
    return result


async def persistence_check(tools: dict[str, Any]) -> dict[str, Any]:
    """Prove cookies and history survive across calls in one MCP session."""
    await tools["browser_navigate"](
        url="https://www.whatarecookies.com/cookietest.asp"
    )
    cookie_snapshot = str(await tools["browser_snapshot"]())
    await tools["browser_navigate"](url="https://example.com/")
    example_snapshot = str(await tools["browser_snapshot"]())
    await tools["browser_navigate_back"]()
    returned_snapshot = str(await tools["browser_snapshot"]())
    checks = {
        "cookie_page_loaded": "Cookie checker" in cookie_snapshot,
        "cookies_enabled": "Cookies are enabled" in cookie_snapshot,
        "example_loaded": "Example Domain" in example_snapshot,
        "history_returned": "Cookie checker" in returned_snapshot,
        "cookie_state_preserved": "Cookies are enabled" in returned_snapshot,
    }
    # History is the session-lifetime assertion. Cookie behavior is recorded
    # separately because the public tester can reject or disable third-party cookies.
    passed = all(
        checks[name]
        for name in ("cookie_page_loaded", "example_loaded", "history_returned")
    )
    return {"passed": passed, "checks": checks}


async def provider_termination_check(
    tools: dict[str, Any], stop: Callable[[], Any]
) -> dict[str, Any]:
    """Stop the provider, then prove the next MCP call fails within its bound."""
    stop()
    started = time.monotonic()
    try:
        response = str(
            await asyncio.wait_for(
                tools["browser_navigate"](
                    url="https://example.com/?ara_provider_termination_probe=1"
                ),
                timeout=TERMINATION_TIMEOUT_SECONDS,
            )
        )
    except Exception as error:
        return {
            "failed_cleanly": True,
            "failure": str(error)[:300],
            "failure_seconds": round(time.monotonic() - started, 3),
        }
    return {
        "failed_cleanly": False,
        "result": response[:300],
        "failure_seconds": round(time.monotonic() - started, 3),
    }


def new_report() -> dict[str, Any]:
    return {
        "run_at": datetime.now(UTC).isoformat(),
        "phases": [],
        "pages": [],
        "private_targets": [],
        "same_session_state_preserved": False,
        "provider_termination": {"failed_cleanly": False, "not_run": True},
        "session_closed": False,
    }


async def smoke(query: str, linkedin_url: str, api_key: str) -> dict[str, Any]:
    started = time.monotonic()
    report = new_report()
    manager = client = session = None
    stopped = False
    try:
        manager = BrowserUse(api_key=api_key)
        client = manager.__enter__()
        session = client.browsers.create(
            proxy_country_code="us",
            metadata={"purpose": "ara-playwright-mcp-smoke"},
            timeout=10,
            solve_captchas=True,
            enable_recording=False,
        )
        if not session.cdp_url:
            raise RuntimeError("Browser provider returned no CDP endpoint")
        source = playwright_browser_tools(
            cdp_endpoint=session.cdp_url, tool_timeout=PHASE_TIMEOUT_SECONDS
        )
        async with mcp_connection(source):
            tools = {ToolDef(tool).name: tool for tool in await source.tools()}
            report["tool_surface_matches"] = set(tools) == set(
                PLAYWRIGHT_BROWSER_TOOL_NAMES
            )
            persistence = await run_phase(
                "session_persistence", lambda: persistence_check(tools), api_key
            )
            report["phases"].append(persistence)
            report["same_session_state_preserved"] = bool(
                persistence.get("ok") and persistence.get("passed")
            )
            urls = (
                ("google_search", f"https://www.google.com/search?q={quote_plus(query)}"),
                ("bing_search", f"https://www.bing.com/search?q={quote_plus(query)}"),
                ("linkedin_public", linkedin_url),
            )
            for name, url in urls:
                result = await run_phase(
                    name,
                    lambda name=name, url=url: live_site_check(tools, name, url),
                    api_key,
                )
                report["phases"].append(result)
                report["pages"].append(
                    {"name": name, "requested_url": url, **result}
                )
            for index, url in enumerate(
                (
                    "file:///etc/passwd",
                    "http://127.0.0.1:9/",
                    "http://169.254.169.254/latest/meta-data/",
                    "http://host.docker.internal/",
                )
            ):
                result = await run_phase(
                    f"private_target_{index}",
                    lambda url=url: blocked_probe(tools["browser_navigate"], url),
                    api_key,
                )
                report["phases"].append(result)
                report["private_targets"].append(result)

            def stop_provider() -> None:
                nonlocal stopped
                client.browsers.stop(session.id)
                stopped = True

            termination = await run_phase(
                "provider_termination",
                lambda: provider_termination_check(tools, stop_provider),
                api_key,
                timeout=TERMINATION_TIMEOUT_SECONDS + 5,
            )
            report["phases"].append(termination)
            report["provider_termination"] = termination
    except Exception as error:
        report["failure"] = redacted_error(error, api_key)
    finally:
        if client is not None and session is not None and not stopped:
            try:
                client.browsers.stop(session.id)
                stopped = True
            except Exception as error:
                report["cleanup_error"] = redacted_error(error, api_key)
        if manager is not None:
            try:
                manager.__exit__(*sys.exc_info())
            except Exception as error:
                report["client_close_error"] = redacted_error(error, api_key)
        report["session_closed"] = stopped
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        report["validation_passed"] = bool(
            report.get("tool_surface_matches")
            and report["same_session_state_preserved"]
            and len(report["pages"]) == 3
            and all(page.get("ok") for page in report["pages"])
            and len(report["private_targets"]) == 4
            and all(
                target.get("ok") and target.get("blocked")
                for target in report["private_targets"]
            )
            and report["provider_termination"].get("ok")
            and report["provider_termination"].get("failed_cleanly")
            and stopped
        )
    return report


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
    api_key = os.environ.get(API_KEY_NAME)
    if api_key:
        report = asyncio.run(smoke(args.query, args.linkedin_url, api_key))
        status = 0 if report.get("validation_passed") else 1
    else:
        report = new_report()
        report["failure"] = f"Missing {API_KEY_NAME}"
        status = 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output.resolve())
    return status


if __name__ == "__main__":
    raise SystemExit(main())
