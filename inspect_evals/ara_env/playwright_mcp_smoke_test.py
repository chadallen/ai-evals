"""Report-classification tests for the no-model Playwright MCP check."""

import asyncio

from inspect_evals.ara_env.playwright_mcp_smoke import (
    access_result,
    blocked_probe,
    linkedin_dismiss_targets,
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
