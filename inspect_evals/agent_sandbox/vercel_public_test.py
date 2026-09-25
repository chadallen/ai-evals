"""Tests for sample-scoped Vercel protection removal."""

import asyncio
import json
from unittest.mock import patch

import httpx
import pytest
from inspect_ai.tool import ToolDef, ToolError

from inspect_evals.agent_sandbox.vercel_public import (
    VERCEL_TOKEN_ENV,
    VERCEL_USER_ID_ENV,
    VercelPublicConfig,
    make_vercel_deployment_public,
)

STARTED_AT = 1_000_000


def _config(token: str = "host-secret") -> VercelPublicConfig:
    return VercelPublicConfig(token=token, user_id="user_fixed123")


def _deployment(**overrides):
    value = {
        "id": "dpl_preview123",
        "url": "site-preview.vercel.app",
        "projectId": "prj_created456",
        "readyState": "READY",
        "target": None,
        "createdAt": STARTED_AT + 1,
        "creator": {"uid": "user_fixed123"},
    }
    value.update(overrides)
    return value


def _run_tool(responder, *, deployment="dpl_preview123", team="team_selected789"):
    original_client = httpx.AsyncClient

    async def run():
        with patch(
            "inspect_evals.agent_sandbox.vercel_public.httpx.AsyncClient",
            side_effect=lambda **kwargs: original_client(
                transport=httpx.MockTransport(responder), **kwargs
            ),
        ):
            public_tool = make_vercel_deployment_public(
                _config(), started_at_ms=STARTED_AT
            )
            return await public_tool(deployment, team)

    return asyncio.run(run())


def test_config_requires_token_and_user_id(monkeypatch):
    monkeypatch.delenv(VERCEL_TOKEN_ENV, raising=False)
    monkeypatch.delenv(VERCEL_USER_ID_ENV, raising=False)
    with pytest.raises(RuntimeError, match=VERCEL_TOKEN_ENV):
        VercelPublicConfig.from_env()

    monkeypatch.setenv(VERCEL_TOKEN_ENV, "token")
    with pytest.raises(RuntimeError, match=VERCEL_USER_ID_ENV):
        VercelPublicConfig.from_env()


def test_config_rejects_invalid_user_id(monkeypatch):
    monkeypatch.setenv(VERCEL_TOKEN_ENV, "token")
    monkeypatch.setenv(VERCEL_USER_ID_ENV, "bad/user")
    with pytest.raises(RuntimeError, match="Vercel user ID"):
        VercelPublicConfig.from_env()


def test_model_schema_contains_only_deployment_and_team():
    parameters = ToolDef(
        make_vercel_deployment_public(_config(), started_at_ms=STARTED_AT)
    ).parameters
    assert set(parameters.properties) == {"deployment_id_or_url", "team_id"}
    assert set(parameters.required) == {"deployment_id_or_url", "team_id"}
    serialized = json.dumps(parameters.model_dump())
    assert "host-secret" not in serialized
    assert "user_fixed123" not in serialized


def test_ready_sample_deployment_makes_only_its_project_public():
    requests = []

    def respond(request):
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, json=_deployment())
        return httpx.Response(
            200,
            json={"id": "prj_created456", "ssoProtection": None},
        )

    result = _run_tool(respond)

    assert result == "Vercel deployment is public: https://site-preview.vercel.app"
    assert [request.method for request in requests] == ["GET", "PATCH"]
    assert [request.url.path for request in requests] == [
        "/v13/deployments/dpl_preview123",
        "/v9/projects/prj_created456",
    ]
    for request in requests:
        assert dict(request.url.params) == {"teamId": "team_selected789"}
        assert request.headers["Authorization"] == "Bearer host-secret"
    assert json.loads(requests[-1].content) == {"ssoProtection": None}


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"readyState": "BUILDING"}, "READY preview"),
        ({"target": "production"}, "READY preview"),
        ({"creator": {"uid": "user_other"}}, "configured Vercel user"),
        ({"createdAt": STARTED_AT - 5 * 60 * 1000 - 1}, "during this Inspect sample"),
        ({"projectId": "bad/project"}, "valid project identity"),
    ],
)
def test_unsafe_deployments_are_rejected_before_project_mutation(overrides, message):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json=_deployment(**overrides))

    with pytest.raises(ToolError, match=message):
        _run_tool(respond)
    assert len(requests) == 1
    assert requests[0].method == "GET"


def test_team_id_is_validated_before_network_request():
    def unexpected(request):
        raise AssertionError(f"unexpected request: {request}")

    with pytest.raises(ToolError, match="starting with team_"):
        _run_tool(unexpected, team="personal")


def test_project_update_must_confirm_protection_is_disabled():
    def respond(request):
        if request.method == "GET":
            return httpx.Response(200, json=_deployment())
        return httpx.Response(
            200,
            json={
                "id": "prj_created456",
                "ssoProtection": {"deploymentType": "all_except_custom_domains"},
            },
        )

    with pytest.raises(ToolError, match="remains enabled"):
        _run_tool(respond)


def test_upstream_error_body_and_token_are_not_exposed():
    secret = "host-secret"

    def respond(request):
        if request.method == "GET":
            return httpx.Response(200, json=_deployment())
        return httpx.Response(500, text=f"upstream echoed {secret}")

    with pytest.raises(ToolError) as caught:
        _run_tool(respond)
    assert secret not in str(caught.value)
    assert "upstream echoed" not in str(caught.value)
