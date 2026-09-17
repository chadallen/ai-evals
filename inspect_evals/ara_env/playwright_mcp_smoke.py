"""Direct Browser Use Cloud and Playwright MCP checks with no solver model."""

from __future__ import annotations

import argparse
import asyncio
import errno
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
from inspect_evals.ara_env.playwright_mcp_launcher import navigation_url_policy_error

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
TERMINATION_ATTEMPT_TIMEOUT_SECONDS = 3
TERMINATION_POLL_INTERVAL_SECONDS = 0.25
CONTAINMENT_TIMEOUT_SECONDS = 10

POLICY_REJECTION = "policy_rejection"
DESTINATION_REACHED = "destination_reached"
CONNECTION_REFUSED = "connection_refused"
PROVIDER_FAILURE = "provider_failure"
TIMEOUT = "timeout"

PLAYWRIGHT_ORIGIN_BLOCK_ERROR = re.compile(
    r"\A### Error\r?\n"
    r"Error: browserBackend\.callTool: net::ERR_BLOCKED_BY_CLIENT at "
    r"(?P<error_url>[^\r\n]+)\r?\n"
    r"Call log:\r?\n"
    r'  - navigating to "(?P<log_url>[^"\r\n]+)", '
    r'waiting until "domcontentloaded"\r?\n?\Z'
)


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


def classify_containment_error(
    error: BaseException, requested_url: str
) -> tuple[str, str | None]:
    """Classify failures without treating generic tool errors as enforcement."""
    if isinstance(error, TimeoutError):
        return TIMEOUT, None

    message = str(error)
    lowered = message.lower()
    if message == navigation_url_policy_error(requested_url):
        return POLICY_REJECTION, "host_navigation_policy"
    origin_block = PLAYWRIGHT_ORIGIN_BLOCK_ERROR.fullmatch(message)
    if (
        origin_block is not None
        and origin_block.group("error_url") == requested_url
        and origin_block.group("log_url") == requested_url
    ):
        return POLICY_REJECTION, "playwright_origin_filter"
    if isinstance(error, ConnectionRefusedError) or (
        isinstance(error, OSError) and error.errno == errno.ECONNREFUSED
    ):
        return CONNECTION_REFUSED, None
    if any(
        marker in lowered
        for marker in ("err_connection_refused", "econnrefused", "connection refused")
    ):
        return CONNECTION_REFUSED, None
    return PROVIDER_FAILURE, None


async def containment_probe(
    navigate, url: str, *, timeout: float = CONTAINMENT_TIMEOUT_SECONDS
) -> dict[str, object]:
    """Record one navigation outcome and whether it proves enforcement."""
    try:
        response = str(await asyncio.wait_for(navigate(url=url), timeout=timeout))
    except Exception as error:
        outcome, layer = classify_containment_error(error, url)
        result: dict[str, object] = {
            "url": url,
            "outcome": outcome,
            "policy_enforced": outcome == POLICY_REJECTION,
            "destination_request_received": False if outcome == POLICY_REJECTION else None,
            "result": str(error)[:200],
        }
        if layer is not None:
            result["enforcement_layer"] = layer
            result["request_disposition"] = (
                "rejected_before_playwright"
                if layer == "host_navigation_policy"
                else "aborted_before_destination"
            )
        elif outcome == CONNECTION_REFUSED:
            result["request_disposition"] = "connection_failed; receipt_unobserved"
        return result
    return {
        "url": url,
        "outcome": DESTINATION_REACHED,
        "policy_enforced": False,
        "destination_request_received": True,
        "request_disposition": "destination_responded",
        "result": response[:200],
    }


def containment_evidence_passed(result: dict[str, object], expected_outcome: str) -> bool:
    """Require the recorded outcome, including explicit policy evidence."""
    if not result.get("ok") or result.get("outcome") != expected_outcome:
        return False
    if expected_outcome == POLICY_REJECTION:
        return bool(
            result.get("policy_enforced")
            and result.get("enforcement_layer")
            in {"host_navigation_policy", "playwright_origin_filter"}
            and result.get("destination_request_received") is False
        )
    return True


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
    tools: dict[str, Any],
    stop: Callable[[], Any],
    *,
    timeout: float = TERMINATION_TIMEOUT_SECONDS,
    attempt_timeout: float = TERMINATION_ATTEMPT_TIMEOUT_SECONDS,
    poll_interval: float = TERMINATION_POLL_INTERVAL_SECONDS,
) -> dict[str, Any]:
    """Stop the provider, then wait for MCP calls to observe termination."""
    stop()
    started = time.monotonic()
    deadline = started + timeout
    attempts = 0
    timed_out_attempts = 0
    while time.monotonic() < deadline:
        attempts += 1
        remaining = deadline - time.monotonic()
        try:
            await asyncio.wait_for(
                tools["browser_navigate"](
                    url=(
                        "https://example.com/"
                        f"?ara_provider_termination_probe={attempts}"
                    )
                ),
                timeout=min(attempt_timeout, remaining),
            )
        except TimeoutError:
            timed_out_attempts += 1
        except Exception as error:
            return {
                "failed_cleanly": True,
                "failure": str(error)[:300],
                "failure_seconds": round(time.monotonic() - started, 3),
                "attempts": attempts,
                "timed_out_attempts": timed_out_attempts,
            }

        remaining = deadline - time.monotonic()
        if remaining > 0:
            await asyncio.sleep(min(poll_interval, remaining))

    return {
        "failed_cleanly": False,
        "failure_seconds": round(time.monotonic() - started, 3),
        "attempts": attempts,
        "timed_out_attempts": timed_out_attempts,
        "result": "Provider remained reachable until the termination deadline",
    }


def new_report() -> dict[str, Any]:
    return {
        "run_at": datetime.now(UTC).isoformat(),
        "phases": [],
        "pages": [],
        "containment_probes": [],
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
            probes = (
                ("public_control", "https://example.com/", DESTINATION_REACHED),
                ("host_scheme_guard", "file:///etc/passwd", POLICY_REJECTION),
                ("loopback", "http://127.0.0.1/", POLICY_REJECTION),
                (
                    "link_local_metadata",
                    "http://169.254.169.254/latest/meta-data/",
                    POLICY_REJECTION,
                ),
                (
                    "docker_host",
                    "http://host.docker.internal/",
                    POLICY_REJECTION,
                ),
            )
            for index, (probe_name, url, expected_outcome) in enumerate(probes):
                result = await run_phase(
                    f"containment_{index}_{probe_name}",
                    lambda url=url: containment_probe(tools["browser_navigate"], url),
                    api_key,
                )
                result["probe_name"] = probe_name
                result["expected_outcome"] = expected_outcome
                result["evidence_passed"] = containment_evidence_passed(
                    result, expected_outcome
                )
                report["phases"].append(result)
                report["containment_probes"].append(result)

            def stop_provider() -> None:
                nonlocal stopped
                client.browsers.stop(session.id)
                stopped = True

            termination = await run_phase(
                "provider_termination",
                lambda: provider_termination_check(tools, stop_provider),
                api_key,
                timeout=TERMINATION_TIMEOUT_SECONDS + 2,
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
            and len(report["containment_probes"]) == 5
            and all(
                probe.get("evidence_passed")
                for probe in report["containment_probes"]
            )
            and any(
                probe.get("enforcement_layer") == "host_navigation_policy"
                and probe.get("policy_enforced")
                and probe.get("destination_request_received") is False
                for probe in report["containment_probes"]
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
