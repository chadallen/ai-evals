"""Tests for the project-scoped Vercel alias tool."""

import asyncio
import json
from unittest.mock import patch

import httpx
import pytest
from inspect_ai.tool import ToolDef, ToolError

from inspect_evals.ara_env.vercel_alias import (
    VERCEL_BASE_DOMAIN_ENV,
    VERCEL_PROJECT_ID_ENV,
    VERCEL_TEAM_ID_ENV,
    VERCEL_TOKEN_ENV,
    VercelAliasConfig,
    assign_vercel_subdomain,
    normalize_base_domain,
    normalize_deployment_reference,
    normalize_dns_label,
)


def _config(token: str = "host-secret") -> VercelAliasConfig:
    return VercelAliasConfig(
        token=token,
        team_id="team_fixed123",
        project_id="prj_fixed456",
        base_domain="training.example.org",
    )


def _run_tool(responder, *, subdomain="Benefits-Review", deployment="dpl_preview123"):
    original_client = httpx.AsyncClient

    async def run():
        with patch(
            "inspect_evals.ara_env.vercel_alias.httpx.AsyncClient",
            side_effect=lambda **kwargs: original_client(
                transport=httpx.MockTransport(responder), **kwargs
            ),
        ):
            return await assign_vercel_subdomain(_config())(subdomain, deployment)

    return asyncio.run(run())


@pytest.mark.parametrize(
    "value,expected",
    [
        ("Benefits-Review", "benefits-review"),
        ("a", "a"),
        ("a" * 63, "a" * 63),
    ],
)
def test_dns_label_normalization(value, expected):
    assert normalize_dns_label(value) == expected


@pytest.mark.parametrize(
    "value",
    ["", " two", "two ", "two.parts", "-start", "end-", "*", "café"],
)
def test_dns_label_rejects_non_label_inputs(value):
    with pytest.raises(ValueError):
        normalize_dns_label(value)


@pytest.mark.parametrize(
    "value",
    [
        "example.com.",
        "https://example.com",
        "*.example.com",
        "localhost",
        "127.0.0.1",
        "example.test",
        "bad_domain.com",
    ],
)
def test_base_domain_rejects_non_public_or_non_bare_values(value):
    with pytest.raises(ValueError):
        normalize_base_domain(value)


def test_base_domain_is_normalized():
    assert normalize_base_domain("Training.Example.ORG") == "training.example.org"


@pytest.mark.parametrize(
    "value,expected",
    [
        ("dpl_preview123", "dpl_preview123"),
        ("site-123.vercel.app", "site-123.vercel.app"),
        ("https://site-123.vercel.app/", "site-123.vercel.app"),
    ],
)
def test_deployment_reference_normalization(value, expected):
    assert normalize_deployment_reference(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "http://site.vercel.app",
        "https://site.vercel.app/path",
        "https://user@site.vercel.app",
        "site",
        "dpl_bad/value",
    ],
)
def test_deployment_reference_rejects_unsafe_values(value):
    with pytest.raises(ValueError):
        normalize_deployment_reference(value)


def test_config_requires_every_host_value(monkeypatch):
    values = {
        VERCEL_TOKEN_ENV: "secret",
        VERCEL_TEAM_ID_ENV: "team_fixed123",
        VERCEL_PROJECT_ID_ENV: "prj_fixed456",
        VERCEL_BASE_DOMAIN_ENV: "training.example.org",
    }
    for missing in values:
        for name, value in values.items():
            monkeypatch.setenv(name, value)
        monkeypatch.delenv(missing)
        with pytest.raises(RuntimeError, match=missing):
            VercelAliasConfig.from_env()


@pytest.mark.parametrize(
    "name,value,message",
    [
        (VERCEL_TEAM_ID_ENV, "user_123", "starting with team_"),
        (VERCEL_PROJECT_ID_ENV, "project-name", "starting with prj_"),
        (VERCEL_BASE_DOMAIN_ENV, "*.example.org", "without a wildcard"),
    ],
)
def test_config_rejects_invalid_scope(monkeypatch, name, value, message):
    monkeypatch.setenv(VERCEL_TOKEN_ENV, "secret")
    monkeypatch.setenv(VERCEL_TEAM_ID_ENV, "team_fixed123")
    monkeypatch.setenv(VERCEL_PROJECT_ID_ENV, "prj_fixed456")
    monkeypatch.setenv(VERCEL_BASE_DOMAIN_ENV, "training.example.org")
    monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match=message):
        VercelAliasConfig.from_env()


def test_model_visible_schema_contains_only_two_inputs():
    parameters = ToolDef(assign_vercel_subdomain(_config())).parameters
    assert set(parameters.properties) == {"subdomain", "deployment_id_or_url"}
    assert set(parameters.required) == {"subdomain", "deployment_id_or_url"}
    serialized = json.dumps(parameters.model_dump())
    for hidden in ("host-secret", "team_fixed123", "prj_fixed456", "training.example.org"):
        assert hidden not in serialized


def test_alias_request_uses_fixed_scope_and_official_endpoints():
    requests = []

    def respond(request):
        requests.append(request)
        if request.url.path.startswith("/v13/deployments/"):
            return httpx.Response(
                200,
                json={
                    "id": "dpl_preview123",
                    "projectId": "prj_fixed456",
                    "readyState": "READY",
                    "target": None,
                },
            )
        if request.method == "GET":
            return httpx.Response(404, json={"error": {"code": "not_found"}})
        return httpx.Response(
            200,
            json={"alias": "benefits-review.training.example.org", "uid": "alias_1"},
        )

    result = _run_tool(respond)

    assert result == "Vercel subdomain assigned: https://benefits-review.training.example.org"
    assert [request.method for request in requests] == ["GET", "GET", "POST"]
    assert [request.url.path for request in requests] == [
        "/v13/deployments/dpl_preview123",
        "/v4/aliases/benefits-review.training.example.org",
        "/v2/deployments/dpl_preview123/aliases",
    ]
    for request in requests:
        assert dict(request.url.params) == {"teamId": "team_fixed123"}
        assert request.headers["Authorization"] == "Bearer host-secret"
    assert json.loads(requests[-1].content) == {"alias": "benefits-review.training.example.org"}


def test_project_ownership_is_verified_before_alias_requests():
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "projectId": "prj_someone_else",
                "readyState": "READY",
                "target": None,
            },
        )

    with pytest.raises(ToolError, match="configured Vercel project"):
        _run_tool(respond)
    assert len(requests) == 1
    assert requests[0].method == "GET"


@pytest.mark.parametrize(
    "status,message",
    [
        (401, "authentication failed"),
        (403, "denied access"),
        (404, "deployment was not found"),
    ],
)
def test_deployment_lookup_maps_bounded_errors(status, message):
    def respond(request):
        return httpx.Response(status, text="untrusted upstream detail")

    with pytest.raises(ToolError, match=message) as caught:
        _run_tool(respond)
    assert "untrusted" not in str(caught.value)


@pytest.mark.parametrize(
    "ready_state,target",
    [("BUILDING", None), ("READY", "production")],
)
def test_alias_requires_ready_preview_before_mutation(ready_state, target):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "projectId": "prj_fixed456",
                "readyState": ready_state,
                "target": target,
            },
        )

    with pytest.raises(ToolError, match="READY preview"):
        _run_tool(respond)
    assert len(requests) == 1


def test_existing_alias_is_rejected_before_mutation():
    requests = []

    def respond(request):
        requests.append(request)
        if request.url.path.startswith("/v13/deployments/"):
            return httpx.Response(
                200,
                json={
                    "projectId": "prj_fixed456",
                    "readyState": "READY",
                    "target": None,
                },
            )
        return httpx.Response(200, json={"alias": "benefits-review.training.example.org"})

    with pytest.raises(ToolError, match="already assigned"):
        _run_tool(respond)
    assert [request.method for request in requests] == ["GET", "GET"]


def test_assignment_conflict_is_bounded():
    def respond(request):
        if request.url.path.startswith("/v13/deployments/"):
            return httpx.Response(
                200,
                json={
                    "projectId": "prj_fixed456",
                    "readyState": "READY",
                    "target": None,
                },
            )
        if request.method == "GET":
            return httpx.Response(404)
        return httpx.Response(409, text="internal details")

    with pytest.raises(ToolError, match="conflicts with an existing resource"):
        _run_tool(respond)


def test_upstream_error_does_not_expose_credentials_or_body():
    secret = "never-return-this-secret"
    original_client = httpx.AsyncClient

    def respond(request):
        return httpx.Response(500, text=f"upstream echoed {secret}")

    async def run():
        with patch(
            "inspect_evals.ara_env.vercel_alias.httpx.AsyncClient",
            side_effect=lambda **kwargs: original_client(
                transport=httpx.MockTransport(respond), **kwargs
            ),
        ):
            await assign_vercel_subdomain(_config(secret))("benefits", "dpl_preview123")

    with pytest.raises(ToolError) as caught:
        asyncio.run(run())
    assert str(caught.value) == "Vercel deployment request failed with status 500."
    assert secret not in str(caught.value)
    assert "echoed" not in str(caught.value)
