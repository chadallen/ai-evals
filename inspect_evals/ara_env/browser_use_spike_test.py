"""Tests for the direct Browser Use Cloud feasibility check."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
from types import SimpleNamespace

from inspect_evals.ara_env.browser_use_spike import (
    navigate_result,
    page_result,
    redact_error,
    run_spike,
)


class FakeLocator:
    def __init__(self, text: str = "", count: int = 0):
        self.text = text
        self.link_count = count

    def inner_text(self, timeout: int) -> str:
        assert timeout == 10_000
        return self.text

    def count(self) -> int:
        return self.link_count


class FakePage:
    def __init__(self):
        self.url = "about:blank"
        self.visits = []
        self.waits = []

    def goto(self, url: str, **kwargs):
        self.url = url
        self.visits.append((url, kwargs))

    def wait_for_timeout(self, milliseconds: int):
        self.waits.append(milliseconds)

    def locator(self, selector: str) -> FakeLocator:
        if selector == "body":
            return FakeLocator("Useful result " * 30)
        assert selector == "a[href]"
        return FakeLocator(count=12)

    def title(self) -> str:
        return "Results"


class FakeContext:
    def __init__(self):
        self.cookies_added = []

    def add_cookies(self, cookies):
        self.cookies_added.extend(cookies)

    def cookies(self, urls):
        assert urls == ["https://example.com"]
        return self.cookies_added


class FakeBrowsers:
    def __init__(self):
        now = datetime.now(UTC)
        self.created = SimpleNamespace(id="session-id", cdp_url="wss://secret/session")
        self.stopped = SimpleNamespace(
            status="stopped",
            started_at=now,
            finished_at=now,
            proxy_used_mb="1.25",
            proxy_cost="0.00625",
            browser_cost="0.00034",
        )
        self.create_kwargs = None
        self.stopped_id = None

    def create(self, **kwargs):
        self.create_kwargs = kwargs
        return self.created

    def stop(self, session_id):
        self.stopped_id = session_id
        return self.stopped


def test_page_result_marks_bot_interstitial_unusable():
    page = FakePage()
    page.locator = lambda selector: (
        FakeLocator("Our systems detected unusual traffic", 0)
        if selector == "body"
        else FakeLocator(count=2)
    )

    result = page_result("google", "https://google.test", page)

    assert not result["usable"]
    assert result["block_markers"] == ["unusual traffic"]


def test_page_result_marks_linkedin_authwall_unusable():
    page = FakePage()
    page.url = "https://www.linkedin.com/authwall?sessionRedirect=profile"

    result = page_result("linkedin", "https://www.linkedin.com/in/person", page)

    assert not result["usable"]
    assert result["block_markers"] == ["linkedin authentication wall"]


def test_navigation_gives_captcha_solver_more_time():
    page = FakePage()
    bodies = iter(
        [
            "IP address: 2001:db8::1 unusual traffic",
            "Useful result " * 30,
        ]
    )
    page.locator = lambda selector: (
        FakeLocator(next(bodies)) if selector == "body" else FakeLocator(count=12)
    )

    result = navigate_result("google", "https://google.test", page)

    assert page.waits == [3_000, 12_000]
    assert result["initial_block_markers"] == ["unusual traffic"]
    assert result["captcha_wait_seconds"] == 12
    assert result["usable"]


def test_run_spike_stops_session_and_reports_cost():
    browsers = FakeBrowsers()
    client = SimpleNamespace(browsers=browsers)
    page = FakePage()
    context = FakeContext()

    @contextmanager
    def connector(cdp_url):
        assert cdp_url == "wss://secret/session"
        yield context, page

    report = run_spike(
        client,
        "api-secret",
        [("one", "https://one.test"), ("two", "https://two.test")],
        connector,
    )

    assert browsers.stopped_id == "session-id"
    assert browsers.create_kwargs["solve_captchas"] is True
    assert report["same_session_state_preserved"] is True
    assert [result["usable"] for result in report["pages"]] == [True, True]
    assert report["usage"] == {
        "status": "stopped",
        "started_at": browsers.stopped.started_at.isoformat(),
        "finished_at": browsers.stopped.finished_at.isoformat(),
        "proxy_used_mb": 1.25,
        "proxy_cost_usd": 0.00625,
        "browser_cost_usd": 0.00034,
        "total_cost_usd": 0.00659,
    }


def test_redact_error_removes_credentials_and_cdp_url():
    error = RuntimeError("failed wss://host/session?token=api-secret for api-secret")

    message = redact_error(error, "api-secret")

    assert "api-secret" not in message
    assert "wss://" not in message
