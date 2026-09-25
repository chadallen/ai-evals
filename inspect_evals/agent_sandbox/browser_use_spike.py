"""Measure Browser Use Cloud access, session state, traffic, and cost."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol
from urllib.parse import quote_plus

from browser_use_sdk.v4 import BrowserUse
from dotenv import load_dotenv
from playwright.sync_api import BrowserContext, Page, sync_playwright

API_KEY_NAME = "BROWSER_USE_API_KEY"
BLOCK_MARKERS = (
    "unusual traffic",
    "verify you are human",
    "verify that you are human",
    "are you a robot",
    "captcha",
    "access denied",
    "automated queries",
    "security check",
    "challenge-platform",
)
LINKEDIN_DISMISS_SELECTORS = (
    "button[aria-label='Dismiss']",
    "button.modal__dismiss",
    "button[data-tracking-control-name*='modal_dismiss']",
)


class Browsers(Protocol):
    def create(self, **kwargs: Any) -> Any: ...

    def stop(self, session_id: Any) -> Any: ...


class Client(Protocol):
    browsers: Browsers


class ConnectedBrowser(Protocol):
    def __enter__(self) -> tuple[Any, Any]: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None: ...


def target_urls(query: str, linkedin_url: str) -> list[tuple[str, str]]:
    encoded = quote_plus(query)
    return [
        ("google_search", f"https://www.google.com/search?q={encoded}"),
        ("bing_search", f"https://www.bing.com/search?q={encoded}"),
        ("linkedin_public", linkedin_url),
    ]


def redact_error(error: BaseException, api_key: str) -> str:
    message = str(error).replace(api_key, "<redacted-api-key>")
    message = re.sub(r"wss?://\S+", "<redacted-cdp-url>", message)
    return message[:500]


def page_result(name: str, requested_url: str, page: Any) -> dict[str, Any]:
    body = page.locator("body").inner_text(timeout=10_000)
    normalized = " ".join(body.split())
    lowered = normalized.lower()
    block_markers = [marker for marker in BLOCK_MARKERS if marker in lowered]
    if "/authwall" in page.url:
        block_markers.append("linkedin sign-up interstitial requires dismissal")
    link_count = page.locator("a[href]").count()
    preview = re.sub(r"IP address:\s*\S+", "IP address: <redacted>", normalized)
    return {
        "name": name,
        "requested_url": requested_url,
        "final_url": page.url,
        "title": page.title(),
        "body_characters": len(body),
        "link_count": link_count,
        "block_markers": block_markers,
        "usable": len(body) >= 200 and link_count >= 5 and not block_markers,
        "text_preview": preview[:300],
    }


def dismiss_linkedin_marketing(page: Any) -> dict[str, Any]:
    """Dismiss LinkedIn's public-page marketing modal when it is present."""
    attempts = []
    dismissed = False
    for selector in LINKEDIN_DISMISS_SELECTORS:
        locator = page.locator(selector)
        if locator.count() == 0:
            continue
        attempts.append(selector)
        try:
            locator.first.click(timeout=3_000)
            dismissed = True
            break
        except Exception:
            continue
    if not dismissed:
        attempts.append("Escape")
        try:
            page.keyboard.press("Escape")
        except Exception:
            pass
    page.wait_for_timeout(500)
    return {"attempted": attempts, "dismissed": dismissed}


def navigate_result(name: str, url: str, page: Any) -> dict[str, Any]:
    page.goto(url, wait_until="domcontentloaded", timeout=45_000)
    page.wait_for_timeout(3_000)
    marketing_prompt = None
    if name == "linkedin_public":
        marketing_prompt = dismiss_linkedin_marketing(page)
    result = page_result(name, url, page)
    initial_markers = result["block_markers"]
    captcha_wait_seconds = 0
    if initial_markers:
        captcha_wait_seconds = 12
        page.wait_for_timeout(captcha_wait_seconds * 1_000)
        result = page_result(name, url, page)
    result["initial_block_markers"] = initial_markers
    result["captcha_wait_seconds"] = captcha_wait_seconds
    if marketing_prompt is not None:
        result["marketing_prompt"] = marketing_prompt
    return result


@contextmanager
def connect_browser(cdp_url: str) -> Iterator[tuple[BrowserContext, Page]]:
    with sync_playwright() as playwright:
        browser = playwright.chromium.connect_over_cdp(cdp_url, timeout=30_000)
        if not browser.contexts:
            raise RuntimeError("Remote browser did not expose a browser context")
        context = browser.contexts[0]
        page = context.pages[0] if context.pages else context.new_page()
        yield context, page


def session_metrics(session: Any) -> dict[str, Any]:
    status = getattr(session.status, "value", session.status)
    return {
        "status": str(status),
        "started_at": session.started_at.isoformat(),
        "finished_at": session.finished_at.isoformat() if session.finished_at else None,
        "proxy_used_mb": float(session.proxy_used_mb or 0),
        "proxy_cost_usd": float(session.proxy_cost or 0),
        "browser_cost_usd": float(session.browser_cost or 0),
    }


def run_spike(
    client: Client,
    api_key: str,
    urls: Sequence[tuple[str, str]],
    connector: Callable[[str], ConnectedBrowser] = connect_browser,
) -> dict[str, Any]:
    started = time.monotonic()
    created = client.browsers.create(
        proxy_country_code="us",
        metadata={"purpose": "ara-browser-feasibility-spike"},
        timeout=10,
        solve_captchas=True,
        enable_recording=False,
    )
    stopped = None
    pages: list[dict[str, Any]] = []
    persistence = False
    failure = None
    try:
        if not created.cdp_url:
            raise RuntimeError("Browser Use did not return a CDP URL")
        with connector(created.cdp_url) as (context, page):
            context.add_cookies(
                [
                    {
                        "name": "ara_spike",
                        "value": "same-session",
                        "domain": "example.com",
                        "path": "/",
                    }
                ]
            )
            for name, url in urls:
                try:
                    pages.append(navigate_result(name, url, page))
                except Exception as error:
                    pages.append(
                        {
                            "name": name,
                            "requested_url": url,
                            "usable": False,
                            "error": redact_error(error, api_key),
                        }
                    )
            cookies = context.cookies(["https://example.com"])
            persistence = any(
                cookie.get("name") == "ara_spike" and cookie.get("value") == "same-session"
                for cookie in cookies
            )
    except Exception as error:
        failure = redact_error(error, api_key)
    finally:
        stopped = client.browsers.stop(created.id)

    metrics = session_metrics(stopped)
    total_cost = metrics["proxy_cost_usd"] + metrics["browser_cost_usd"]
    return {
        "run_at": datetime.now(UTC).isoformat(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "proxy_country": "us",
        "captcha_solving": True,
        "recording": False,
        "same_session_state_preserved": persistence,
        "pages": pages,
        "failure": failure,
        "usage": {**metrics, "total_cost_usd": round(total_cost, 6)},
    }


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--linkedin-url", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scratch/browser-use-spike-results.json"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(repo_root / ".env")
    api_key = os.environ.get(API_KEY_NAME)
    if not api_key:
        print(f"Missing {API_KEY_NAME}; add it to {repo_root / '.env'}", file=sys.stderr)
        return 2

    with BrowserUse(api_key=api_key) as client:
        report = run_spike(client, api_key, target_urls(args.query, args.linkedin_url))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output.resolve())
    return 0 if report["failure"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
