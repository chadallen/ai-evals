"""Report-classification tests for the no-model Playwright MCP check."""

import asyncio
import json

import pytest

from inspect_evals.ara_env.playwright_mcp_smoke import (
    CONNECTION_REFUSED,
    DESTINATION_REACHED,
    POLICY_REJECTION,
    PROVIDER_FAILURE,
    TIMEOUT,
    access_result,
    containment_evidence_passed,
    containment_probe,
    linkedin_dismiss_targets,
    live_site_check,
    main,
    persistence_check,
    provider_termination_check,
    run_phase,
)


def test_access_result_rejects_bot_interstitial():
    result = access_result(
        "google_search",
        "https://google.test/search",
        "Our systems detected unusual traffic " * 20,
    )

    assert result["usable"] is False
    assert result["block_markers"] == ["unusual traffic"]


def test_access_result_rejects_google_sorry_redirect():
    result = access_result(
        "google_search",
        "https://google.test/search",
        "Page URL: https://www.google.com/sorry/index?continue=search " * 10,
    )

    assert result["usable"] is False
    assert result["block_markers"] == ["google.com/sorry/"]


def test_access_result_marks_linkedin_sign_up_after_dismissal():
    result = access_result(
        "linkedin_public",
        "https://linkedin.test/in/person",
        "Sign Up | LinkedIn Join LinkedIn " * 20,
    )

    assert result["usable"] is False
    assert result["block_markers"] == [
        "linkedin sign-up interstitial remains after dismissal"
    ]


def test_containment_probe_records_explicit_host_policy_rejection():
    async def navigate(**kwargs):
        raise RuntimeError(
            "Navigation URL must be a well-formed HTTP or HTTPS URL "
            "without embedded credentials."
        )

    result = asyncio.run(containment_probe(navigate, "file:///etc/passwd"))

    assert result["outcome"] == POLICY_REJECTION
    assert result["policy_enforced"] is True
    assert result["enforcement_layer"] == "host_navigation_policy"
    assert result["destination_request_received"] is False
    assert result["request_disposition"] == "rejected_before_playwright"


def test_containment_probe_records_playwright_origin_rejection():
    url = "http://127.0.0.1/"

    async def navigate(**kwargs):
        raise RuntimeError(
            "### Error\n"
            f"Error: browserBackend.callTool: net::ERR_BLOCKED_BY_CLIENT at {url}\n"
            "Call log:\n"
            f'  - navigating to "{url}", waiting until "domcontentloaded"\n'
        )

    result = asyncio.run(containment_probe(navigate, url))

    assert result["outcome"] == POLICY_REJECTION
    assert result["policy_enforced"] is True
    assert result["enforcement_layer"] == "playwright_origin_filter"
    assert result["destination_request_received"] is False


@pytest.mark.parametrize(
    ("requested_url", "error_url", "log_url"),
    [
        ("http://127.0.0.1/", "http://169.254.169.254/", "http://169.254.169.254/"),
        ("http://127.0.0.1/", "http://127.0.0.1", "http://127.0.0.1"),
        (
            "http://127.0.0.1/?probe=one",
            "http://127.0.0.1/?probe=two",
            "http://127.0.0.1/?probe=two",
        ),
        (
            "http://127.0.0.1/?first=1&second=2",
            "http://127.0.0.1/?second=2&first=1",
            "http://127.0.0.1/?second=2&first=1",
        ),
        (
            "http://127.0.0.1/?value=%2F",
            "http://127.0.0.1/?value=/",
            "http://127.0.0.1/?value=/",
        ),
        (
            "http://127.0.0.1/",
            "http://127.0.0.1/",
            "http://169.254.169.254/",
        ),
    ],
)
def test_containment_probe_rejects_origin_block_for_a_different_exact_url(
    requested_url, error_url, log_url
):
    async def navigate(**kwargs):
        raise RuntimeError(
            "### Error\n"
            "Error: browserBackend.callTool: net::ERR_BLOCKED_BY_CLIENT "
            f"at {error_url}\n"
            "Call log:\n"
            f'  - navigating to "{log_url}", waiting until "domcontentloaded"\n'
        )

    result = asyncio.run(containment_probe(navigate, requested_url))

    assert result["outcome"] == PROVIDER_FAILURE
    assert result["policy_enforced"] is False
    assert result["destination_request_received"] is None


@pytest.mark.parametrize(
    "message",
    [
        "Provider reported ERR_BLOCKED_BY_CLIENT while processing the request.",
        (
            "Provider failure: ### Error\n"
            "Error: browserBackend.callTool: net::ERR_BLOCKED_BY_CLIENT "
            "at http://127.0.0.1/\n"
            "Call log:\n"
            '  - navigating to "http://127.0.0.1/", waiting until '
            '"domcontentloaded"\n'
        ),
        "page.goto: net::ERR_BLOCKED_BY_CLIENT at http://127.0.0.1/",
    ],
)
def test_containment_probe_rejects_provider_prose_that_mentions_block_code(message):
    async def navigate(**kwargs):
        raise RuntimeError(message)

    result = asyncio.run(containment_probe(navigate, "http://127.0.0.1/"))

    assert result["outcome"] == PROVIDER_FAILURE
    assert result["policy_enforced"] is False


def test_containment_probe_requires_exact_host_guard_error():
    async def navigate(**kwargs):
        raise RuntimeError(
            "Provider repeated: Navigation URL must be a well-formed HTTP or HTTPS "
            "URL without embedded credentials."
        )

    result = asyncio.run(containment_probe(navigate, "file:///etc/passwd"))

    assert result["outcome"] == PROVIDER_FAILURE
    assert result["policy_enforced"] is False


@pytest.mark.parametrize("url", ["https://example.com/", "http://127.0.0.1/"])
def test_containment_probe_rejects_exact_host_guard_text_for_allowed_url(url):
    async def navigate(**kwargs):
        raise RuntimeError(
            "Navigation URL must be a well-formed HTTP or HTTPS URL "
            "without embedded credentials."
        )

    result = asyncio.run(containment_probe(navigate, url))

    assert result["outcome"] == PROVIDER_FAILURE
    assert result["policy_enforced"] is False
    assert result["destination_request_received"] is None


def test_containment_probe_records_destination_reached_even_with_error_text():
    async def navigate(**kwargs):
        return "Application error: request failed"

    result = asyncio.run(containment_probe(navigate, "https://fixture.test"))

    assert result["outcome"] == DESTINATION_REACHED
    assert result["policy_enforced"] is False
    assert result["destination_request_received"] is True


def test_containment_probe_records_connection_refused_without_policy_credit():
    async def navigate(**kwargs):
        raise ConnectionRefusedError("connection refused")

    result = asyncio.run(containment_probe(navigate, "http://127.0.0.1"))

    assert result["outcome"] == CONNECTION_REFUSED
    assert result["policy_enforced"] is False
    assert result["destination_request_received"] is None
    assert result["request_disposition"] == "connection_failed; receipt_unobserved"


def test_containment_probe_records_provider_failure_without_policy_credit():
    async def navigate(**kwargs):
        raise RuntimeError("Request blocked by origin policy: provider failed")

    result = asyncio.run(containment_probe(navigate, "http://127.0.0.1"))

    assert result["outcome"] == PROVIDER_FAILURE
    assert result["policy_enforced"] is False
    assert result["destination_request_received"] is None


def test_containment_probe_records_timeout_without_policy_credit():
    async def navigate(**kwargs):
        await asyncio.sleep(0.05)

    result = asyncio.run(
        containment_probe(navigate, "http://127.0.0.1", timeout=0.001)
    )

    assert result["outcome"] == TIMEOUT
    assert result["policy_enforced"] is False
    assert result["destination_request_received"] is None


def test_report_does_not_accept_non_policy_failure_as_enforcement():
    provider_failure = {
        "ok": True,
        "outcome": PROVIDER_FAILURE,
        "policy_enforced": False,
        "destination_request_received": None,
    }
    closed_port = {
        "ok": True,
        "outcome": CONNECTION_REFUSED,
        "policy_enforced": False,
        "destination_request_received": None,
    }

    assert containment_evidence_passed(provider_failure, POLICY_REJECTION) is False
    assert containment_evidence_passed(closed_port, POLICY_REJECTION) is False


def test_report_requires_named_layer_and_non_arrival_for_policy_evidence():
    incomplete_policy_result = {
        "ok": True,
        "outcome": POLICY_REJECTION,
        "policy_enforced": True,
        "destination_request_received": None,
    }
    explicit_policy_result = {
        **incomplete_policy_result,
        "enforcement_layer": "host_navigation_policy",
        "destination_request_received": False,
    }

    assert (
        containment_evidence_passed(incomplete_policy_result, POLICY_REJECTION)
        is False
    )
    assert (
        containment_evidence_passed(explicit_policy_result, POLICY_REJECTION) is True
    )


def test_linkedin_dismiss_targets_uses_visible_snapshot_refs_only():
    snapshot = """
    - button "Dismiss" [ref=e12]
    - button "Connect" [ref=e13]
    - button "Not now" [ref=e14]
    """

    assert linkedin_dismiss_targets(snapshot) == [
        ("e12", '- button "Dismiss" [ref=e12]'),
        ("e14", '- button "Not now" [ref=e14]'),
    ]


def test_live_site_uses_snapshot_instead_of_navigation_response():
    calls = []

    async def navigate(**kwargs):
        calls.append(("navigate", kwargs))
        return "unusual traffic from navigation metadata"

    async def snapshot(**kwargs):
        calls.append(("snapshot", kwargs))
        return "Search results with useful public links " * 20

    result = asyncio.run(
        live_site_check(
            {"browser_navigate": navigate, "browser_snapshot": snapshot},
            "google_search",
            "https://google.test/search",
        )
    )

    assert result["usable"] is True
    assert calls == [
        ("navigate", {"url": "https://google.test/search"}),
        ("snapshot", {}),
    ]


def test_linkedin_dismisses_from_initial_snapshot_and_classifies_fresh_snapshot():
    calls = []
    snapshots = iter(
        [
            '- dialog "Join LinkedIn"\n- button "Dismiss" [ref=e12]',
            "Public profile Experience Skills Education " * 20,
        ]
    )

    async def record(name, result=None, **kwargs):
        calls.append((name, kwargs))
        return result

    tools = {
        "browser_navigate": lambda **kwargs: record("navigate", **kwargs),
        "browser_snapshot": lambda **kwargs: record(
            "snapshot", next(snapshots), **kwargs
        ),
        "browser_click": lambda **kwargs: record("click", **kwargs),
        "browser_press_key": lambda **kwargs: record("key", **kwargs),
    }
    result = asyncio.run(
        live_site_check(
            tools, "linkedin_public", "https://linkedin.test/in/person"
        )
    )

    assert result["usable"] is True
    assert result["marketing_prompt"]["dismissed"] is True
    assert [call[0] for call in calls] == ["navigate", "snapshot", "click", "snapshot"]
    assert calls[2][1]["target"] == "e12"


def test_persistence_snapshots_after_each_navigation():
    calls = []
    snapshots = iter(
        [
            "Cookie checker Cookies are enabled",
            "Example Domain",
            "Cookie checker Cookies are enabled",
        ]
    )

    async def navigate(**kwargs):
        calls.append("navigate")

    async def snapshot(**kwargs):
        calls.append("snapshot")
        return next(snapshots)

    async def back(**kwargs):
        calls.append("back")

    result = asyncio.run(
        persistence_check(
            {
                "browser_navigate": navigate,
                "browser_snapshot": snapshot,
                "browser_navigate_back": back,
            }
        )
    )

    assert result["passed"] is True
    assert calls == [
        "navigate",
        "snapshot",
        "navigate",
        "snapshot",
        "back",
        "snapshot",
    ]


def test_persistence_does_not_treat_public_cookie_policy_as_session_loss():
    snapshots = iter(
        [
            "Cookie checker third-party cookies disabled",
            "Example Domain",
            "Cookie checker",
        ]
    )

    async def navigate(**kwargs):
        return None

    async def snapshot(**kwargs):
        return next(snapshots)

    async def back(**kwargs):
        return None

    result = asyncio.run(
        persistence_check(
            {
                "browser_navigate": navigate,
                "browser_snapshot": snapshot,
                "browser_navigate_back": back,
            }
        )
    )

    assert result["passed"] is True
    assert result["checks"]["cookies_enabled"] is False


def test_phase_timeout_does_not_prevent_later_phase():
    completed = []

    async def too_slow():
        await asyncio.sleep(0.05)
        return {"value": "late"}

    async def next_phase():
        completed.append(True)
        return {"value": "done"}

    async def check():
        first = await run_phase("first", too_slow, "secret", timeout=0.001)
        second = await run_phase("second", next_phase, "secret", timeout=1)
        return first, second

    first, second = asyncio.run(check())
    assert first["ok"] is False
    assert second["ok"] is True
    assert completed == [True]


def test_provider_termination_precedes_bounded_tool_failure():
    calls = []

    def stop():
        calls.append("stop")

    async def navigate(**kwargs):
        calls.append(("navigate", kwargs))
        raise RuntimeError("browser session closed")

    result = asyncio.run(
        provider_termination_check(
            {"browser_navigate": navigate}, stop, poll_interval=0
        )
    )

    assert result["failed_cleanly"] is True
    assert calls == [
        "stop",
        (
            "navigate",
            {"url": "https://example.com/?ara_provider_termination_probe=1"},
        ),
    ]
    assert result["attempts"] == 1
    assert result["timed_out_attempts"] == 0


def test_provider_termination_polls_during_delayed_cloud_shutdown():
    calls = []

    def stop():
        calls.append("stop")

    async def navigate(**kwargs):
        calls.append(kwargs["url"])
        if len(calls) < 4:
            return "Example Domain"
        raise RuntimeError("browser session closed")

    result = asyncio.run(
        provider_termination_check(
            {"browser_navigate": navigate},
            stop,
            timeout=1,
            attempt_timeout=0.1,
            poll_interval=0,
        )
    )

    assert result["failed_cleanly"] is True
    assert result["attempts"] == 3
    assert result["timed_out_attempts"] == 0
    assert calls == [
        "stop",
        "https://example.com/?ara_provider_termination_probe=1",
        "https://example.com/?ara_provider_termination_probe=2",
        "https://example.com/?ara_provider_termination_probe=3",
    ]


def test_main_writes_report_when_api_key_is_missing(tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    monkeypatch.delenv("BROWSER_USE_API_KEY", raising=False)
    monkeypatch.setattr(
        "inspect_evals.ara_env.playwright_mcp_smoke.load_dotenv", lambda *args: None
    )

    status = main(["--output", str(output)])

    assert status == 2
    report = json.loads(output.read_text())
    assert report["failure"] == "Missing BROWSER_USE_API_KEY"
    assert report["session_closed"] is False
