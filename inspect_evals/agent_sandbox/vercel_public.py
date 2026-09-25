"""Constrained Vercel protection removal for deployments created by one sample."""

import os
import re
import time
from dataclasses import dataclass
from urllib.parse import quote

import httpx
from inspect_ai.tool import Tool, ToolError, tool

from inspect_evals.agent_sandbox.vercel_alias import normalize_deployment_reference

VERCEL_API_URL = "https://api.vercel.com"
VERCEL_TOKEN_ENV = "VERCEL_TOKEN"
VERCEL_USER_ID_ENV = "VERCEL_USER_ID"

_VERCEL_ID = re.compile(r"[A-Za-z0-9_]+\Z")
_CLOCK_SKEW_MS = 5 * 60 * 1000


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise RuntimeError(f"publish_to_vercel=true requires {name} in the host environment.")
    return value.strip()


@dataclass(frozen=True)
class VercelPublicConfig:
    """Host identity used to bind a project mutation to this account."""

    token: str
    user_id: str

    @classmethod
    def from_env(cls) -> VercelPublicConfig:
        token = _required_env(VERCEL_TOKEN_ENV)
        user_id = _required_env(VERCEL_USER_ID_ENV)
        if not _VERCEL_ID.fullmatch(user_id):
            raise RuntimeError("VERCEL_USER_ID must be a Vercel user ID.")
        return cls(token=token, user_id=user_id)


def _api_error(status: int, operation: str) -> ToolError:
    if status == 401:
        return ToolError("Vercel authentication failed.")
    if status == 403:
        return ToolError("Vercel denied access to the selected team.")
    if status == 404:
        return ToolError(f"Vercel {operation} was not found in the selected team.")
    return ToolError(f"Vercel {operation} request failed with status {status}.")


def _project_id(details: dict[str, object]) -> str:
    top_level = details.get("projectId")
    project = details.get("project")
    nested = project.get("id") if isinstance(project, dict) else None
    if top_level is not None and not isinstance(top_level, str):
        raise ToolError("Vercel deployment response was invalid.")
    if project is not None and not isinstance(project, dict):
        raise ToolError("Vercel deployment response was invalid.")
    if nested is not None and not isinstance(nested, str):
        raise ToolError("Vercel deployment response was invalid.")
    if top_level and nested and top_level != nested:
        raise ToolError("Vercel deployment response contained conflicting project identity.")
    value = top_level or nested
    if not value or not value.startswith("prj_") or not _VERCEL_ID.fullmatch(value):
        raise ToolError("Vercel deployment response did not include valid project identity.")
    return value


def make_vercel_deployment_public(
    config: VercelPublicConfig,
    *,
    started_at_ms: int | None = None,
) -> Tool:
    """Expose only a project tied to a READY preview created during this sample."""
    sample_started_at = started_at_ms if started_at_ms is not None else time.time_ns() // 1_000_000

    @tool
    def make_vercel_deployment_public() -> Tool:
        """Make the current sample's READY Vercel preview publicly accessible."""

        async def execute(deployment_id_or_url: str, team_id: str) -> str:
            """Remove Vercel Authentication after verifying sample ownership.

            Args:
                deployment_id_or_url: Deployment ID or HTTPS URL from create_deployment.
                team_id: Team ID used for create_deployment and get_deployment.
            """
            try:
                deployment = normalize_deployment_reference(deployment_id_or_url)
            except ValueError as exc:
                raise ToolError(str(exc)) from exc
            if not team_id.startswith("team_") or not _VERCEL_ID.fullmatch(team_id):
                raise ToolError("Team ID must be a Vercel team ID starting with team_.")

            headers = {
                "Accept": "application/json",
                "Authorization": f"Bearer {config.token}",
            }
            async with httpx.AsyncClient(
                base_url=VERCEL_API_URL,
                headers=headers,
                timeout=20,
            ) as client:
                try:
                    response = await client.get(
                        f"/v13/deployments/{quote(deployment, safe='')}",
                        params={"teamId": team_id},
                    )
                except (httpx.TimeoutException, httpx.RequestError) as exc:
                    raise ToolError("Vercel API request failed.") from exc
                if response.status_code != 200:
                    raise _api_error(response.status_code, "deployment")
                try:
                    details = response.json()
                except ValueError as exc:
                    raise ToolError("Vercel deployment response was invalid.") from exc
                if not isinstance(details, dict):
                    raise ToolError("Vercel deployment response was invalid.")
                if details.get("readyState") != "READY" or details.get("target") is not None:
                    raise ToolError("Deployment must be a READY preview before it is made public.")
                creator = details.get("creator")
                if not isinstance(creator, dict) or creator.get("uid") != config.user_id:
                    raise ToolError("Deployment was not created by the configured Vercel user.")
                created_at = details.get("createdAt")
                if not isinstance(created_at, int) or created_at < sample_started_at - _CLOCK_SKEW_MS:
                    raise ToolError("Deployment was not created during this Inspect sample.")
                project_id = _project_id(details)

                try:
                    response = await client.patch(
                        f"/v9/projects/{quote(project_id, safe='')}",
                        params={"teamId": team_id},
                        json={"ssoProtection": None},
                    )
                except (httpx.TimeoutException, httpx.RequestError) as exc:
                    raise ToolError("Vercel API request failed.") from exc
                if response.status_code != 200:
                    raise _api_error(response.status_code, "project update")
                try:
                    project = response.json()
                except ValueError as exc:
                    raise ToolError("Vercel project response was invalid.") from exc
                if not isinstance(project, dict) or project.get("id") != project_id:
                    raise ToolError("Vercel project response was invalid.")
                if project.get("ssoProtection") is not None:
                    raise ToolError("Vercel Authentication remains enabled for the project.")

            hostname = details.get("url")
            if not isinstance(hostname, str) or not hostname:
                raise ToolError("Vercel deployment response did not include its URL.")
            return f"Vercel deployment is public: https://{hostname}"

        return execute

    return make_vercel_deployment_public()


def vercel_public_tool_from_env() -> Tool:
    """Construct the sample-scoped public deployment tool from host credentials."""
    return make_vercel_deployment_public(VercelPublicConfig.from_env())
