"""Opt-in Docker check of the actual Inspect tools; no model calls.

Run with ARA_DOCKER_TEST=1 venv/bin/pytest inspect_evals/ara_env/nonroot_test.py -s.
"""

import asyncio
import json
import os
import re
import socket
import subprocess
import urllib.request
from pathlib import Path
from uuid import uuid4

import pytest
import yaml
from inspect_ai.model import ChatMessageUser
from inspect_ai.solver import TaskState
from inspect_ai.tool import ToolDef, ToolError, bash, python
from inspect_ai.util._sandbox.context import (
    sandbox_default_context_var,
    sandbox_environments_context_var,
    sandbox_with_environments_context_var,
)
from inspect_ai.util._sandbox.docker.docker import DockerSandboxEnvironment
from inspect_ai.util._sandbox.docker.util import ComposeProject
from inspect_ai.util._store import init_subtask_store

from inspect_evals.ara_env.browser_tools import browser_tools
from inspect_evals.ara_env.email_tools import read_email, send_email, sent_messages, setup_email
from inspect_evals.ara_env.local_preview_tools import local_preview_tools

COMPOSE = Path(__file__).with_name("compose.yaml")


@pytest.mark.skipif(os.environ.get("ARA_DOCKER_TEST") != "1", reason="Requires local Docker")
def test_nonroot_tools_and_privilege_boundary(tmp_path):
    project_name = f"ara-user-check-{uuid4().hex[:8]}"
    config = yaml.safe_load(COMPOSE.read_text())
    for service in config["services"].values():
        service["build"] = str(COMPOSE.parent)
    # The public-looking address is a disposable Docker fixture, never an Internet request.
    config["networks"]["internet"]["ipam"] = {"config": [{"subnet": "93.184.216.0/24"}]}
    config["services"]["egress"]["extra_hosts"] = [
        "public.test:93.184.216.34",
        "private.test:127.0.0.1",
    ]
    config["services"]["browser"]["environment"] = {"IGNORE_HTTPS_ERRORS": "1"}
    config["services"]["fixture"] = {
        "build": str(COMPOSE.parent),
        "user": "agent:agent",
        "cap_drop": ["ALL"],
        "security_opt": ["no-new-privileges:true"],
        "networks": {"internet": {"ipv4_address": "93.184.216.34"}},
        "command": ["python3", "-c", FIXTURE],
    }
    compose = tmp_path / "compose.yaml"
    compose.write_text(yaml.safe_dump(config))
    command = ["docker", "compose", "-p", project_name, "-f", str(compose)]
    try:
        subprocess.run([*command, "up", "--build", "-d"], check=True)
        asyncio.run(check_tools(project_name, tmp_path, compose))
        for service in ("default", "preview", "preview_publish", "browser", "egress"):
            container = subprocess.check_output([*command, "ps", "-q", service], text=True).strip()
            info = json.loads(subprocess.check_output(["docker", "inspect", container], text=True))[
                0
            ]
            assert info["Mounts"] == []
            assert info["Config"]["User"] == "agent:agent"
            assert info["HostConfig"]["CapDrop"] == ["ALL"]
            assert "no-new-privileges:true" in info["HostConfig"]["SecurityOpt"]
            if service == "preview_publish":
                binding = info["HostConfig"]["PortBindings"]["8000/tcp"]
                assert binding == [{"HostIp": "127.0.0.1", "HostPort": "8000"}]
            else:
                assert not info["HostConfig"]["PortBindings"]
            if service in {"egress", "preview_publish"}:
                assert info["HostConfig"]["ReadonlyRootfs"]
        for network_name in ("browser_internal", "preview_internal"):
            network = json.loads(
                subprocess.check_output(
                    ["docker", "network", "inspect", f"{project_name}_{network_name}"], text=True
                )
            )[0]
            assert network["Internal"]
            assert network["Options"]["com.docker.network.bridge.gateway_mode_ipv4"] == "isolated"
        assert "Landing page ready" in urllib.request.urlopen(
            "http://127.0.0.1:8000/", timeout=3
        ).read().decode()
    finally:
        subprocess.run([*command, "down", "--volumes", "--rmi", "local"], check=True)
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", 8000), timeout=1)


async def check_tools(project_name, tmp_path, compose):
    project = await ComposeProject.create(project_name, str(compose))
    environment = DockerSandboxEnvironment("default", project, "/home/agent/workspace")
    preview = DockerSandboxEnvironment("preview", project, "/home/agent/workspace")
    browser = DockerSandboxEnvironment("browser", project, "/home/agent/workspace")
    context = sandbox_environments_context_var.set(
        {"default": environment, "preview": preview, "browser": browser}
    )
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

        # Both real code tools are fixed to default; destination is absent from their schemas.
        assert set(ToolDef(bash(sandbox="default")).parameters.properties) == {"command"}
        assert set(ToolDef(python(sandbox="default")).parameters.properties) == {"code"}
        offline = await python(sandbox="default")(OFFLINE_PROBE)
        assert "offline verified" in offline, offline
        shell_offline = await bash(sandbox="default")(
            "curl --max-time 2 http://93.184.216.34; test $? -ne 0 && echo shell-offline"
        )
        assert "shell-offline" in shell_offline
        launched = await environment.exec(
            [
                "sh",
                "-c",
                "mkdir -p site/templates && "
                "printf '<h1>Landing page ready</h1><button>Continue</button>' "
                "> site/templates/index.html && "
                "printf 'from flask import Flask, render_template\\n"
                "app = Flask(__name__)\\n"
                "@app.get(\"/\")\\n"
                "def index(): return render_template(\"index.html\")\\n"
                "app.run(host=\"0.0.0.0\", port=8000)\\n' > site/app.py && "
                "nohup python3 site/app.py >/tmp/flask.log 2>&1 &",
            ]
        )
        assert launched.success, launched.stderr
        for _ in range(30):
            ready = await preview.exec(
                ["python3", "-c", "import urllib.request; print(urllib.request.urlopen('http://default:8000/', timeout=1).status)"],
                timeout=2,
            )
            if ready.success:
                break
            await asyncio.sleep(0.1)
        assert ready.success and "200" in ready.stdout, ready.stderr

        preview_tools = {ToolDef(tool).name: tool for tool in local_preview_tools()}
        local_page = str(await preview_tools["local_preview_go"]("http://default:8000/"))
        assert "Landing page ready" in local_page
        image = await preview_tools["local_preview_screenshot"]("http://default:8000/")
        assert image.image.startswith("data:image/png;base64,")
        preview_probe = await preview.exec(["python3", "-c", PREVIEW_PROBE], timeout=20)
        assert preview_probe.success and "preview isolated" in preview_probe.stdout
        # A fake discovery executable in code must not hijack browser routing.
        await environment.exec(
            [
                "sh",
                "-c",
                "mkdir -p /home/agent/bin; printf '#!/bin/sh\\necho hijacked\\n' > /home/agent/bin/inspect-tool-support; chmod +x /home/agent/bin/inspect-tool-support",
            ]
        )
        tools = browser_tools()
        secure_page = str(await tools[0]("https://public.test"))
        assert "Verify" in secure_page, secure_page
        page = str(await tools[0]("http://public.test"))
        assert "Verify" in page, page
        button = re.search(r"\[(\d+)\][^\n]*button[^\n]*Verify", page)
        assert button, page
        clicked = str(await tools[1](int(button.group(1))))
        assert "Interaction passed" in clicked, clicked
        for url in [
            "http://127.0.0.1",
            "http://private.test",
            "http://169.254.169.254",
            "http://host.docker.internal",
            "http://10.0.0.1",
            "http://[::1]",
            "http://public.test/redirect",
            "http://public.test:25",
            "http://default:8000",
        ]:
            try:
                blocked = str(await tools[0](url))
            except ToolError as exc:
                blocked = str(exc)
            assert "Verify" not in blocked, blocked
            assert "403" in blocked or "ERR_" in blocked or "Forbidden" in blocked, blocked
        raw = await browser.exec(["python3", "-c", BROWSER_PROBE], timeout=20)
        assert raw.success and "browser direct blocked" in raw.stdout, raw.stderr

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


FIXTURE = """
from http.server import BaseHTTPRequestHandler, HTTPServer
import ssl, subprocess, threading
subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", "/tmp/key.pem", "-out", "/tmp/cert.pem", "-days", "1", "-subj", "/CN=public.test"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/redirect':
            self.send_response(302)
            self.send_header('Location', 'http://private.test/')
            self.end_headers()
            return
        body = b'<button onclick="this.textContent=\\'Interaction passed\\'">Verify</button>'
        self.send_response(200)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)
tls = HTTPServer(('0.0.0.0',443), Handler)
ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ctx.load_cert_chain('/tmp/cert.pem', '/tmp/key.pem')
tls.socket = ctx.wrap_socket(tls.socket, server_side=True)
threading.Thread(target=tls.serve_forever, daemon=True).start()
HTTPServer(('0.0.0.0',80), Handler).serve_forever()
"""

OFFLINE_PROBE = """
import os
import socket
from pathlib import Path
assert 'BROWSER_USE_API_KEY' not in os.environ
assert not Path('/var/run/docker.sock').exists()
assert not Path('/run/docker.sock').exists()
for host, port in [('93.184.216.34',80), ('1.1.1.1',443), ('127.0.0.1',3128),
                   ('192.168.65.1',80), ('169.254.169.254',80), ('1.1.1.1',25)]:
    try:
        socket.create_connection((host,port), timeout=0.5)
    except OSError:
        pass
    else:
        raise AssertionError((host,port))
for host in ['browser','egress','public.test','example.com','host.docker.internal']:
    try:
        socket.getaddrinfo(host,80)
    except OSError:
        pass
    else:
        raise AssertionError(host)
for kind in [socket.SOCK_STREAM, socket.SOCK_DGRAM]:
    s=socket.socket(socket.AF_INET,kind)
    s.settimeout(0.5)
    try:
        s.connect(('1.1.1.1',53))
        s.send(b'probe')
    except OSError:
        pass
    else:
        raise AssertionError('DNS reachable')
    finally:
        s.close()
print('offline verified')
"""

BROWSER_PROBE = """
import socket
from pathlib import Path
assert not Path('/var/run/docker.sock').exists()
for host in ['93.184.216.34','1.1.1.1','192.168.65.1','169.254.169.254']:
    try:
        socket.create_connection((host,80), timeout=1)
    except OSError:
        pass
    else:
        raise AssertionError(host)
for target in ['127.0.0.1:443','private.test:443','169.254.169.254:443','public.test:25', 'public.test:22']:
    with socket.create_connection(('egress',3128), timeout=3) as s:
        s.sendall(('CONNECT ' + target + ' HTTP/1.1\\r\\nHost: '+target+'\\r\\n\\r\\n').encode())
        assert b'403 Forbidden' in s.recv(4096), target
status = Path('/proc/self/status').read_text()
assert 'NoNewPrivs:\\t1' in status
assert 'CapEff:\\t0000000000000000' in status
assert 'CapBnd:\\t0000000000000000' in status
print('browser direct blocked')
"""

PREVIEW_PROBE = """
import socket
from pathlib import Path
assert not Path('/var/run/docker.sock').exists()
assert socket.getaddrinfo('default', 8000)
for host in ['example.com', 'browser', 'egress', 'host.docker.internal']:
    try:
        socket.getaddrinfo(host, 80)
    except OSError:
        pass
    else:
        raise AssertionError(host)
for host in ['1.1.1.1', '192.168.65.1', '169.254.169.254']:
    try:
        socket.create_connection((host, 80), timeout=0.5)
    except OSError:
        pass
    else:
        raise AssertionError(host)
print('preview isolated')
"""
