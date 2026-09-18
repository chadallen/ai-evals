"""Project-scoped Vercel alias assignment for the spearphish task."""

import os
import re
from dataclasses import dataclass
from ipaddress import ip_address
from urllib.parse import quote, urlsplit

import httpx
from inspect_ai.tool import Tool, ToolError, tool

VERCEL_API_URL = "https://api.vercel.com"
VERCEL_TOKEN_ENV = "VERCEL_TOKEN"
VERCEL_TEAM_ID_ENV = "VERCEL_TEAM_ID"
VERCEL_PROJECT_ID_ENV = "VERCEL_PROJECT_ID"
VERCEL_BASE_DOMAIN_ENV = "VERCEL_BASE_DOMAIN"

_DNS_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")
_VERCEL_ID = re.compile(r"[A-Za-z0-9_]+\Z")
_RESERVED_SUFFIXES = {"example", "invalid", "local", "localhost", "test"}


def normalize_dns_label(value: str) -> str:
    """Normalize one ASCII DNS label for use beneath the configured domain."""
    if value != value.strip():
        raise ValueError("Subdomain must not contain surrounding whitespace.")
    try:
        label = value.lower().encode("ascii").decode("ascii")
    except UnicodeError as exc:
        raise ValueError("Subdomain must contain ASCII characters only.") from exc
    if not _DNS_LABEL.fullmatch(label):
        raise ValueError("Subdomain must be one DNS label using letters, digits, or hyphens.")
    return label


def normalize_base_domain(value: str) -> str:
    """Normalize a public ASCII base domain supplied by the host."""
    if value != value.strip():
        raise ValueError("VERCEL_BASE_DOMAIN must not contain surrounding whitespace.")
    try:
        domain = value.lower().encode("ascii").decode("ascii")
    except UnicodeError as exc:
        raise ValueError("VERCEL_BASE_DOMAIN must contain ASCII characters only.") from exc
    if not domain or len(domain) > 253 or domain.endswith("."):
        raise ValueError("VERCEL_BASE_DOMAIN must be a bare domain without a trailing dot.")
    if any(character in domain for character in ":/@?#\\*"):
        raise ValueError(
            "VERCEL_BASE_DOMAIN must be a bare domain without a wildcard or URL parts."
        )
    labels = domain.split(".")
    if len(labels) < 2 or any(not _DNS_LABEL.fullmatch(label) for label in labels):
        raise ValueError("VERCEL_BASE_DOMAIN must be a valid bare domain.")
    try:
        ip_address(domain)
    except ValueError:
        pass
    else:
        raise ValueError("VERCEL_BASE_DOMAIN must not be an IP address.")
    if labels[-1] in _RESERVED_SUFFIXES:
        raise ValueError("VERCEL_BASE_DOMAIN must use a public DNS suffix.")
    return domain


def normalize_deployment_reference(value: str) -> str:
    """Return a Vercel deployment ID or hostname safe for one URL path segment."""
    if value != value.strip() or not value or len(value) > 253:
        raise ValueError("Deployment reference is invalid.")
    reference = value
    if "://" in reference:
        parsed = urlsplit(reference)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.port
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "Deployment URL must be one HTTPS origin without credentials or a path."
            )
        reference = parsed.hostname
    try:
        reference.encode("ascii")
    except UnicodeError as exc:
        raise ValueError("Deployment reference must contain ASCII characters only.") from exc
    if reference.startswith("dpl_"):
        if not _VERCEL_ID.fullmatch(reference):
            raise ValueError("Deployment ID is invalid.")
        return reference
    labels = reference.lower().split(".")
    if len(labels) < 2 or any(not _DNS_LABEL.fullmatch(label) for label in labels):
        raise ValueError("Deployment reference must be a deployment ID or hostname.")
    return reference.lower()


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise RuntimeError(f"publish_to_vercel=true requires {name} in the host environment.")
    return value.strip()


@dataclass(frozen=True)
class VercelAliasConfig:
    """Host-owned scope for Vercel alias operations."""

    token: str
    team_id: str
    project_id: str
    base_domain: str

    @classmethod
    def from_env(cls) -> VercelAliasConfig:
        token = _required_env(VERCEL_TOKEN_ENV)
        team_id = _required_env(VERCEL_TEAM_ID_ENV)
        project_id = _required_env(VERCEL_PROJECT_ID_ENV)
        base_domain_value = _required_env(VERCEL_BASE_DOMAIN_ENV)
        if not team_id.startswith("team_") or not _VERCEL_ID.fullmatch(team_id):
            raise RuntimeError("VERCEL_TEAM_ID must be a Vercel team ID starting with team_.")
        if not project_id.startswith("prj_") or not _VERCEL_ID.fullmatch(project_id):
            raise RuntimeError("VERCEL_PROJECT_ID must be a Vercel project ID starting with prj_.")
        try:
            base_domain = normalize_base_domain(base_domain_value)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        return cls(token, team_id, project_id, base_domain)


def _api_error(status: int, operation: str) -> ToolError:
    if status == 401:
        return ToolError("Vercel authentication failed.")
    if status == 403:
        return ToolError("Vercel denied access to the configured scope.")
    if status == 404 and operation == "deployment":
        return ToolError("Vercel deployment was not found in the configured team.")
    if status == 409:
        return ToolError("The requested Vercel alias conflicts with an existing resource.")
    return ToolError(f"Vercel {operation} request failed with status {status}.")


async def _request(
    client: httpx.AsyncClient,
    config: VercelAliasConfig,
    method: str,
    path: str,
    *,
    body: dict[str, str] | None = None,
) -> httpx.Response:
    try:
        return await client.request(
            method,
            path,
            params={"teamId": config.team_id},
            headers={"Authorization": f"Bearer {config.token}"},
            json=body,
        )
    except (httpx.TimeoutException, httpx.RequestError) as exc:
        raise ToolError("Vercel API request failed.") from exc


def assign_vercel_subdomain(config: VercelAliasConfig) -> Tool:
    """Create the constrained Vercel alias tool for one host-owned project."""

    @tool
    def assign_vercel_subdomain() -> Tool:
        """Assign a configured-domain subdomain to a READY preview deployment."""

        async def execute(subdomain: str, deployment_id_or_url: str) -> str:
            """Assign a subdomain after verifying deployment ownership and readiness.

            Args:
                subdomain: One DNS label to place under the configured base domain.
                deployment_id_or_url: Vercel deployment ID or HTTPS deployment URL.
            """
            try:
                label = normalize_dns_label(subdomain)
                deployment = normalize_deployment_reference(deployment_id_or_url)
            except ValueError as exc:
                raise ToolError(str(exc)) from exc
            alias = f"{label}.{config.base_domain}"
            if len(alias) > 253:
                raise ToolError("The configured domain and subdomain exceed the DNS length limit.")
            headers = {"Accept": "application/json"}
            async with httpx.AsyncClient(
                base_url=VERCEL_API_URL,
                headers=headers,
                timeout=20,
            ) as client:
                encoded_deployment = quote(deployment, safe="")
                response = await _request(
                    client,
                    config,
                    "GET",
                    f"/v13/deployments/{encoded_deployment}",
                )
                if response.status_code != 200:
                    raise _api_error(response.status_code, "deployment")
                try:
                    details = response.json()
                except ValueError as exc:
                    raise ToolError("Vercel deployment response was invalid.") from exc
                if not isinstance(details, dict):
                    raise ToolError("Vercel deployment response was invalid.")
                if details.get("projectId") != config.project_id:
                    raise ToolError("Deployment does not belong to the configured Vercel project.")
                if details.get("readyState") != "READY" or details.get("target") is not None:
                    raise ToolError("Deployment must be a READY preview before alias assignment.")

                encoded_alias = quote(alias, safe="")
                response = await _request(
                    client,
                    config,
                    "GET",
                    f"/v4/aliases/{encoded_alias}",
                )
                if response.status_code == 200:
                    raise ToolError("The requested Vercel alias is already assigned.")
                if response.status_code != 404:
                    raise _api_error(response.status_code, "alias lookup")

                response = await _request(
                    client,
                    config,
                    "POST",
                    f"/v2/deployments/{encoded_deployment}/aliases",
                    body={"alias": alias},
                )
                if response.status_code != 200:
                    raise _api_error(response.status_code, "alias assignment")
                try:
                    result = response.json()
                except ValueError as exc:
                    raise ToolError("Vercel alias response was invalid.") from exc
                if not isinstance(result, dict):
                    raise ToolError("Vercel alias response was invalid.")
                if result.get("alias") != alias or result.get("oldDeploymentId"):
                    raise ToolError("Vercel returned an unexpected alias assignment result.")
            return f"Vercel subdomain assigned: https://{alias}"

        return execute

    return assign_vercel_subdomain()


def vercel_alias_tool_from_env() -> Tool:
    """Validate host configuration and construct the constrained alias tool."""
    return assign_vercel_subdomain(VercelAliasConfig.from_env())
