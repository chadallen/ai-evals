"""Own a cloud browser session and relay stdio to Playwright MCP."""

from __future__ import annotations

import contextlib
import io
import ipaddress
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, unquote, urlsplit

from browser_use_sdk.v4 import BrowserUse

API_KEY_NAME = "BROWSER_USE_API_KEY"
CDP_ENDPOINT_NAME = "ARA_BROWSER_CDP_ENDPOINT"
REPO_ROOT = Path(__file__).resolve().parents[2]
SHUTDOWN_GRACE_SECONDS = 2.0
SHUTDOWN_KILL_SECONDS = 2.0


class _ShutdownRequested(BaseException):
    """Unwind provider and workspace state after a launcher signal."""

    def __init__(self, signum: int) -> None:
        self.signum = signum
        super().__init__(signum)

# Playwright documents origin filters as defense in depth, not a security
# boundary. These patterns still prevent direct requests to common local,
# private, link-local, and task-only names from the remote browser.
PRIVATE_ORIGIN_BLOCKS = (
    "localhost",
    "127.*",
    "[::1]",
    "0.0.0.0",
    "10.*",
    "100.64.*",
    "169.254.*",
    *(f"172.{second}.*" for second in range(16, 32)),
    "192.168.*",
    "host.docker.internal",
    "*.local",
)


def _redact_values(message: str, secrets: list[str]) -> str:
    for secret in secrets:
        if secret:
            message = message.replace(secret, "<redacted>")
    return message


def _redact(message: str, secrets: list[str]) -> str:
    message = _redact_values(message, secrets)
    return re.sub(r"wss?://\S+", "<redacted-cdp-endpoint>", message)


def _endpoint_secrets(endpoint: str) -> list[str]:
    """Return endpoint values that can reveal provider credentials."""
    parsed = urlsplit(endpoint)
    secrets = [endpoint, unquote(endpoint)]
    if parsed.username:
        secrets.append(unquote(parsed.username))
    if parsed.password:
        secrets.append(unquote(parsed.password))
    secrets.extend(
        unquote(value) for _, value in parse_qsl(parsed.query) if len(value) >= 8
    )
    secrets.extend(
        unquote(part) for part in parsed.path.split("/") if len(part) >= 8
    )
    return sorted({secret for secret in secrets if secret}, key=len, reverse=True)


def _relay_output(
    source: Any,
    destination: Any,
    secrets: list[str],
    output_lock: threading.Lock | None = None,
) -> None:
    """Relay one MCP output stream after removing CDP credentials."""
    for line in iter(source.readline, b""):
        text = line.decode("utf-8", errors="replace")
        lock = output_lock or contextlib.nullcontext()
        with lock:
            destination.write(_redact_values(text, secrets).encode())
            destination.flush()


def _request_error(request: object) -> dict[str, object] | None:
    """Return an MCP error when a tool call crosses a host boundary."""
    if not isinstance(request, dict) or request.get("method") != "tools/call":
        return None
    params = request.get("params")
    if not isinstance(params, dict):
        return None
    name = params.get("name")
    arguments = params.get("arguments")
    if not isinstance(arguments, dict):
        return None

    if name == "browser_snapshot" and "filename" in arguments:
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "error": {
                "code": -32602,
                "message": "Snapshot file output is disabled.",
            },
        }

    url: object | None = None
    if name == "browser_navigate":
        url = arguments.get("url")
    elif name == "browser_tabs" and arguments.get("action") == "new":
        # Omitting the optional URL opens a blank tab. A supplied URL must
        # cross the same boundary as browser_navigate.
        if "url" not in arguments or arguments["url"] is None:
            return None
        url = arguments["url"]
    else:
        return None

    if _valid_public_navigation_url(url):
        return None
    return {
        "jsonrpc": "2.0",
        "id": request.get("id"),
        "error": {
            "code": -32602,
            "message": (
                "Navigation URL must be a well-formed HTTP or HTTPS URL "
                "without embedded credentials."
            ),
        },
    }


def _valid_public_navigation_url(url: object) -> bool:
    """Accept browser navigation URLs with a network-safe syntax."""
    if not isinstance(url, str) or not url or "\\" in url:
        return False
    if any(character.isspace() or ord(character) < 0x20 for character in url):
        return False
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError:
        return False
    if parsed.scheme.lower() not in {"http", "https"}:
        return False
    if not parsed.netloc or not parsed.hostname:
        return False
    if parsed.username is not None or parsed.password is not None:
        return False
    if port is not None and not 1 <= port <= 65535:
        return False
    return _valid_hostname(parsed.hostname)


def _valid_hostname(hostname: str) -> bool:
    """Reject host syntax that browsers may reinterpret inconsistently."""
    if "%" in hostname:
        return False
    try:
        ascii_hostname = hostname.encode("idna").decode("ascii").rstrip(".")
    except UnicodeError:
        return False
    if not ascii_hostname or len(ascii_hostname) > 253:
        return False
    if ":" in ascii_hostname:
        try:
            ipaddress.ip_address(ascii_hostname)
        except ValueError:
            return False
        return True
    return all(
        1 <= len(label) <= 63
        and re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?", label)
        for label in ascii_hostname.split(".")
    )


def _relay_input(
    source: Any,
    destination: Any,
    response_destination: Any,
    output_lock: threading.Lock,
) -> None:
    """Relay parent MCP requests after enforcing navigation policy."""
    try:
        for line in iter(source.readline, b""):
            try:
                request = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                request = None
            error = _request_error(request)
            if error is not None:
                response = (json.dumps(error, separators=(",", ":")) + "\n").encode()
                with output_lock:
                    response_destination.write(response)
                    response_destination.flush()
                continue
            destination.write(line)
            destination.flush()
    except (BrokenPipeError, ValueError):
        pass
    finally:
        with contextlib.suppress(BrokenPipeError, ValueError):
            destination.close()


def _mcp_environment(
    source: Mapping[str, str], cdp_endpoint: str, workspace: Path
) -> dict[str, str]:
    sample_dir = workspace.parent
    output_dir = sample_dir / "downloads"
    config_path = sample_dir / "playwright-mcp-config.json"
    config_path.write_text(
        json.dumps(
            {
                "browser": {"contextOptions": {"acceptDownloads": False}},
                "outputDir": str(output_dir),
                "allowUnrestrictedFileAccess": False,
            }
        )
    )
    env = dict(source)
    env.pop(API_KEY_NAME, None)
    env.pop(CDP_ENDPOINT_NAME, None)
    env.pop("ARA_PLAYWRIGHT_MCP_EXECUTABLE", None)
    timeout = env.pop("ARA_BROWSER_TOOL_TIMEOUT_MS", "120000")
    env.update(
        PLAYWRIGHT_MCP_CDP_ENDPOINT=cdp_endpoint,
        PLAYWRIGHT_MCP_CDP_TIMEOUT="30000",
        PLAYWRIGHT_MCP_TIMEOUT_ACTION=timeout,
        PLAYWRIGHT_MCP_TIMEOUT_NAVIGATION=timeout,
        PLAYWRIGHT_MCP_IDLE_TIMEOUT="300000",
        PLAYWRIGHT_MCP_CODEGEN="none",
        PLAYWRIGHT_MCP_BLOCKED_ORIGINS=";".join(PRIVATE_ORIGIN_BLOCKS),
        PLAYWRIGHT_MCP_BLOCK_SERVICE_WORKERS="true",
        PLAYWRIGHT_MCP_CONFIG=str(config_path),
        PLAYWRIGHT_MCP_OUTPUT_DIR=str(output_dir),
        PLAYWRIGHT_MCP_ALLOW_UNRESTRICTED_FILE_ACCESS="false",
        PLAYWRIGHT_MCP_ISOLATED="true",
    )
    return env


def _run_server(
    executable: str,
    cdp_endpoint: str,
    source: Mapping[str, str],
    workspace: Path,
) -> int:
    process = subprocess.Popen(
        [executable],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_mcp_environment(source, cdp_endpoint, workspace),
        cwd=workspace,
        start_new_session=True,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None

    secrets = _endpoint_secrets(cdp_endpoint)
    output_lock = threading.Lock()
    input_thread = threading.Thread(
        target=_relay_input,
        args=(sys.stdin.buffer, process.stdin, sys.stdout.buffer, output_lock),
        daemon=True,
    )
    output_threads = [
        threading.Thread(
            target=_relay_output,
            args=(stream, destination, secrets, output_lock),
            daemon=True,
        )
        for stream, destination in (
            (process.stdout, sys.stdout.buffer),
            (process.stderr, sys.stderr.buffer),
        )
    ]
    input_thread.start()
    for thread in output_threads:
        thread.start()

    try:
        return_code = process.wait()
        for thread in output_threads:
            thread.join(timeout=10)
        return return_code
    finally:
        _shutdown_process_group(process)


def _signal_process_group(process: subprocess.Popen[bytes], signum: int) -> bool:
    """Signal the dedicated MCP process group, including descendants."""
    pid = getattr(process, "pid", None)
    if pid is not None:
        try:
            os.killpg(pid, signum)
            return True
        except ProcessLookupError:
            return False

    if process.poll() is not None:
        return False
    if signum == signal.SIGTERM:
        process.terminate()
    else:
        process.kill()
    return True


def _shutdown_process_group(process: subprocess.Popen[bytes]) -> None:
    """Stop the MCP process tree without allowing cleanup to wait forever."""
    group_exists = _signal_process_group(process, signal.SIGTERM)
    if process.poll() is None:
        try:
            process.wait(timeout=SHUTDOWN_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            pass
    elif group_exists:
        # The group leader may exit before a descendant. Give those descendants
        # the same short graceful-shutdown window before forcing termination.
        time.sleep(SHUTDOWN_GRACE_SECONDS)

    if group_exists:
        _signal_process_group(process, signal.SIGKILL)
    if process.poll() is None:
        try:
            process.wait(timeout=SHUTDOWN_KILL_SECONDS)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("Unable to stop Playwright MCP process group") from error


def _workspace_path(path: str) -> Path:
    """Verify the disposable MCP workspace is outside repository evidence."""
    workspace = Path(path).resolve()
    if workspace == REPO_ROOT or REPO_ROOT in workspace.parents:
        raise RuntimeError("Refusing to run Playwright MCP inside the repository")
    return workspace


def _main_in_workspace(workspace: Path) -> int:
    executable = os.environ["ARA_PLAYWRIGHT_MCP_EXECUTABLE"]
    configured_endpoint = os.environ.get(CDP_ENDPOINT_NAME)
    if configured_endpoint:
        return _run_server(executable, configured_endpoint, os.environ, workspace)

    api_key = os.environ.get(API_KEY_NAME)
    if not api_key:
        print(f"Missing {API_KEY_NAME} or {CDP_ENDPOINT_NAME}", file=sys.stderr)
        return 2

    client_manager = None
    client = None
    session = None
    captured = io.StringIO()
    try:
        # Third-party startup output must not corrupt the MCP stdout channel.
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            client_manager = BrowserUse(api_key=api_key)
            client = client_manager.__enter__()
            session = client.browsers.create(
                proxy_country_code="us",
                metadata={"purpose": "ara-playwright-mcp"},
                timeout=10,
                solve_captchas=True,
                enable_recording=False,
            )
        if not session.cdp_url:
            raise RuntimeError("Browser provider returned no CDP endpoint")
        return _run_server(executable, session.cdp_url, os.environ, workspace)
    except Exception as error:
        details = _redact(f"{captured.getvalue()}\n{error}", [api_key])
        print(f"Unable to start remote browser: {details[-500:]}", file=sys.stderr)
        return 1
    finally:
        try:
            with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
                if client is not None:
                    if session is not None:
                        client.browsers.stop(session.id)
                    assert client_manager is not None
                    client_manager.__exit__(*sys.exc_info())
        except Exception as error:
            details = _redact(str(error), [api_key])
            print(f"Unable to stop remote browser cleanly: {details[-500:]}", file=sys.stderr)


def main() -> int:
    """Run one MCP sample and remove its workspace after every exit path."""
    def request_shutdown(signum: int, _frame: Any) -> None:
        raise _ShutdownRequested(signum)

    previous = {
        signum: signal.signal(signum, request_shutdown)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        try:
            with tempfile.TemporaryDirectory(prefix="ara-playwright-mcp-") as path:
                sample_dir = _workspace_path(path)
                workspace = sample_dir / "workspace"
                workspace.mkdir()
                return _main_in_workspace(workspace)
        except _ShutdownRequested as shutdown:
            return 128 + shutdown.signum
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
