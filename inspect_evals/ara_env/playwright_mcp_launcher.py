"""Own a cloud browser session and relay stdio to Playwright MCP."""

from __future__ import annotations

import contextlib
import io
import os
import re
import signal
import subprocess
import sys
import threading
from collections.abc import Mapping
from typing import Any
from urllib.parse import parse_qsl, unquote, urlsplit

from browser_use_sdk.v4 import BrowserUse

API_KEY_NAME = "BROWSER_USE_API_KEY"
CDP_ENDPOINT_NAME = "ARA_BROWSER_CDP_ENDPOINT"


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


def _relay_output(source: Any, destination: Any, secrets: list[str]) -> None:
    """Relay one MCP output stream after removing CDP credentials."""
    for line in iter(source.readline, b""):
        text = line.decode("utf-8", errors="replace")
        destination.write(_redact_values(text, secrets).encode())
        destination.flush()


def _relay_input(source: Any, destination: Any) -> None:
    """Relay parent MCP requests until either side closes."""
    try:
        for line in iter(source.readline, b""):
            destination.write(line)
            destination.flush()
    except (BrokenPipeError, ValueError):
        pass
    finally:
        with contextlib.suppress(BrokenPipeError, ValueError):
            destination.close()


def _mcp_environment(source: Mapping[str, str], cdp_endpoint: str) -> dict[str, str]:
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
    )
    return env


def _run_server(executable: str, cdp_endpoint: str, source: Mapping[str, str]) -> int:
    process = subprocess.Popen(
        [executable],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_mcp_environment(source, cdp_endpoint),
    )
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None

    secrets = _endpoint_secrets(cdp_endpoint)
    input_thread = threading.Thread(
        target=_relay_input,
        args=(sys.stdin.buffer, process.stdin),
        daemon=True,
    )
    output_threads = [
        threading.Thread(
            target=_relay_output,
            args=(stream, destination, secrets),
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

    def terminate(_signum: int, _frame: Any) -> None:
        process.terminate()

    previous = {
        signum: signal.signal(signum, terminate)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        return_code = process.wait()
        for thread in output_threads:
            thread.join(timeout=10)
        return return_code
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)


def main() -> int:
    executable = os.environ["ARA_PLAYWRIGHT_MCP_EXECUTABLE"]
    configured_endpoint = os.environ.get(CDP_ENDPOINT_NAME)
    if configured_endpoint:
        return _run_server(executable, configured_endpoint, os.environ)

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
        return _run_server(executable, session.cdp_url, os.environ)
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


if __name__ == "__main__":
    raise SystemExit(main())
