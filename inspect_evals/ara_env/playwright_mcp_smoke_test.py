"""Report-classification tests for the no-model Playwright MCP check."""

import asyncio
import json

from inspect_evals.ara_env.playwright_mcp_smoke import (
    access_result,
    blocked_probe,
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


def test_blocked_probe_accepts_explicit_tool_error():
    async def navigate(**kwargs):
        raise RuntimeError("Request blocked by origin policy")

    result = asyncio.run(blocked_probe(navigate, "http://127.0.0.1"))

    assert result["blocked"] is True
    assert "blocked" in result["result"]


def test_blocked_probe_rejects_successful_private_response():
    async def navigate(**kwargs):
        return "Private service content"

    result = asyncio.run(blocked_probe(navigate, "http://127.0.0.1"))

    assert result["blocked"] is False


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
