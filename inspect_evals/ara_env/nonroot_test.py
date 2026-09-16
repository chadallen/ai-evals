"""Opt-in Docker check of the actual Inspect tools; no model calls.

Run with ARA_DOCKER_TEST=1 venv/bin/pytest inspect_evals/ara_env/nonroot_test.py -s.
"""

import asyncio
import json
import os
import re
import subprocess
from pathlib import Path
from uuid import uuid4

import pytest
from inspect_ai.model import ChatMessageUser
from inspect_ai.solver import TaskState
from inspect_ai.tool import bash, python, web_browser
from inspect_ai.util._sandbox.context import (
    sandbox_default_context_var,
    sandbox_environments_context_var,
    sandbox_with_environments_context_var,
)
from inspect_ai.util._sandbox.docker.docker import DockerSandboxEnvironment
from inspect_ai.util._sandbox.docker.util import ComposeProject
from inspect_ai.util._store import init_subtask_store

from inspect_evals.ara_env.email_tools import read_email, send_email, sent_messages, setup_email

COMPOSE = Path(__file__).with_name("compose.yaml")


@pytest.mark.skipif(os.environ.get("ARA_DOCKER_TEST") != "1", reason="Requires local Docker")
def test_nonroot_tools_and_privilege_boundary(tmp_path):
    project_name = f"ara-user-check-{uuid4().hex[:8]}"
    command = ["docker", "compose", "-p", project_name, "-f", str(COMPOSE)]
    try:
        subprocess.run([*command, "up", "--build", "-d"], check=True)
        asyncio.run(check_tools(project_name, tmp_path))
        container = subprocess.check_output([*command, "ps", "-q", "default"], text=True).strip()
        info = json.loads(subprocess.check_output(["docker", "inspect", container], text=True))[0]
        assert info["Mounts"] == []
    finally:
        subprocess.run([*command, "down", "--volumes", "--rmi", "local"], check=True)


async def check_tools(project_name, tmp_path):
    project = await ComposeProject.create(project_name, str(COMPOSE))
    environment = DockerSandboxEnvironment("default", project, "/home/agent/workspace")
    context = sandbox_environments_context_var.set({"default": environment})
    default = sandbox_default_context_var.set("default")
    injection = sandbox_with_environments_context_var.set({})
    state = TaskState(
        model="mockllm/model",
        sample_id="check",
        epoch=1,
        input=[ChatMessageUser(content="check")],
        messages=[],
    )
    init_subtask_store(state.store)
    try:
        shell_result = await bash()(
            "id -u; pwd; echo writable > shell-check.txt; cat shell-check.txt"
        )
        assert "1000" in shell_result
        assert "/home/agent/workspace" in shell_result
        assert "writable" in shell_result
        result = await python()("""
import os
from pathlib import Path
assert os.getuid() == 1000
assert os.environ["HOME"] == "/home/agent"
Path.home().joinpath("home-check.txt").write_text("ok")
Path("python-check.txt").write_text("ok")
try:
    os.setuid(0)
except PermissionError:
    print("root escalation denied")
else:
    raise AssertionError("regained root")
status = Path("/proc/self/status").read_text()
assert "NoNewPrivs:\\t1" in status
assert "CapEff:\\t0000000000000000" in status
assert "CapBnd:\\t0000000000000000" in status
assert not os.access("/root", os.R_OK)
print("permissions verified")
""")
        assert "root escalation denied" in result
        assert "permissions verified" in result
        privileged = await environment.exec(
            ["sh", "-c", "su -c 'id -u' root </dev/null"], timeout=10
        )
        assert not privileged.success

        await environment.write_file(
            "index.html",
            """<html><body>
<button onclick="document.getElementById('result').textContent='Interaction passed'">Verify</button>
<p id="result">Ready</p></body></html>""",
        )
        started = await environment.exec(
            ["sh", "-c", "python3 -m http.server 8765 --bind 127.0.0.1 >/tmp/http.log 2>&1 &"],
            timeout=10,
        )
        assert started.success
        tools = web_browser()
        page = str(await tools[0]("http://127.0.0.1:8765"))
        assert "Verify" in page, page
        button = re.search(r"\[(\d+)\][^\n]*button[^\n]*Verify", page)
        assert button, page
        clicked = str(await tools[1](int(button.group(1))))
        assert "Interaction passed" in clicked, clicked

        await setup_email(output_dir=str(tmp_path))(state, None)
        assert json.loads(await read_email()()) == {"messages": [], "total": 0}
        confirmation = json.loads(await send_email()("test@corp.example", "Check", "body"))
        messages = sent_messages(state.metadata["email_directory"])
        assert messages[0]["id"] == confirmation["id"]
        inaccessible = await environment.exec(["test", "-e", state.metadata["email_directory"]])
        assert not inaccessible.success
    finally:
        sandbox_environments_context_var.reset(context)
        sandbox_default_context_var.reset(default)
        sandbox_with_environments_context_var.reset(injection)
