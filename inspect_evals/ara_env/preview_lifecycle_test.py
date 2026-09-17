"""Opt-in check of Inspect's preserved-sandbox lifecycle; no model calls.

Run with:
ARA_INSPECT_LIFECYCLE_TEST=1 venv/bin/pytest \
  inspect_evals/ara_env/preview_lifecycle_test.py -s

The test runs Inspect with its mock model and removes only the Docker environment it creates.
"""

import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from inspect_ai.log import EvalLog, read_eval_log

from inspect_evals.ara_env.preview_lifecycle_fixture import PAGE_MARKER

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).with_name("preview_lifecycle_fixture.py")
INSPECT = ROOT / "venv" / "bin" / "inspect"
PREVIEW_URL = "http://127.0.0.1:8000/"
FIXTURE_PROJECT_PREFIX = "inspect-preview_life-i"


def _port_is_open() -> bool:
    with socket.socket() as client:
        client.settimeout(0.2)
        return client.connect_ex(("127.0.0.1", 8000)) == 0


def _wait_for_page() -> str:
    deadline = time.monotonic() + 10
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return urllib.request.urlopen(PREVIEW_URL, timeout=1).read().decode()
        except (OSError, urllib.error.URLError) as error:
            last_error = error
            time.sleep(0.1)
    raise AssertionError(f"preview did not become reachable: {last_error}")


def _wait_for_closed_port() -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if not _port_is_open():
            return
        time.sleep(0.1)
    raise AssertionError("preview port remained open after Inspect sandbox cleanup")


def _fixture_projects() -> set[str]:
    result = subprocess.run(
        [
            "docker",
            "ps",
            "--filter",
            "label=com.docker.compose.project",
            "--format",
            '{{.Label "com.docker.compose.project"}}',
        ],
        text=True,
        capture_output=True,
        check=True,
        timeout=10,
    )
    return {
        project
        for project in result.stdout.splitlines()
        if project.startswith(FIXTURE_PROJECT_PREFIX)
    }


def _eval_command(log_dir: Path) -> list[str]:
    return [
        str(INSPECT),
        "eval",
        str(FIXTURE.relative_to(ROOT)),
        "--model",
        "mockllm/model",
        "--no-sandbox-cleanup",
        "--display",
        "none",
        "--log-dir",
        str(log_dir),
    ]


def _read_only_log(log_dir: Path) -> EvalLog:
    logs = list(log_dir.glob("*.eval"))
    assert len(logs) == 1
    return read_eval_log(logs[0], header_only=True)


@pytest.mark.skipif(
    os.environ.get("ARA_INSPECT_LIFECYCLE_TEST") != "1",
    reason="Requires local Docker",
)
def test_inspect_preserves_preview_until_explicit_cleanup(tmp_path):
    if _port_is_open():
        pytest.fail("host port 8000 is already in use; clean up the existing preview first")
    assert not _fixture_projects(), "a preserved lifecycle-test environment already exists"

    cleanup: list[str] | None = None
    try:
        first = subprocess.run(
            _eval_command(tmp_path / "first"),
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=180,
        )
        assert first.returncode == 0, first.stdout + first.stderr
        first_log = _read_only_log(tmp_path / "first")
        assert first_log.status == "success"
        assert first_log.stats.model_usage == {}
        projects = _fixture_projects()
        assert len(projects) == 1
        cleanup = [str(INSPECT), "sandbox", "cleanup", "docker", projects.pop()]
        assert PAGE_MARKER in _wait_for_page()

        second = subprocess.run(
            _eval_command(tmp_path / "second"),
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=180,
        )
        assert second.returncode == 0, second.stdout + second.stderr
        second_log = _read_only_log(tmp_path / "second")
        assert second_log.status == "error"
        assert second_log.error is not None
        collision = second_log.error.message.lower()
        assert "8000" in collision
        assert "address already in use" in collision or "port is already allocated" in collision

        cleaned = subprocess.run(
            cleanup,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=60,
        )
        assert cleaned.returncode == 0, cleaned.stdout + cleaned.stderr
        _wait_for_closed_port()
    finally:
        if cleanup is not None:
            subprocess.run(cleanup, cwd=ROOT, capture_output=True, timeout=60)
