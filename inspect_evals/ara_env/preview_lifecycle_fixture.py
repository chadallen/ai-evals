"""No-model Inspect fixture for the preserved local-preview lifecycle test."""

import asyncio
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import SandboxEnvironmentSpec, sandbox

COMPOSE_FILE = Path(__file__).with_name("compose.yaml")
PAGE_MARKER = "Inspect preserved preview is ready"


@solver
def launch_preview_site() -> Solver:
    """Start a Flask page without making a model request."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        script = f"""
from flask import Flask

app = Flask(__name__)

@app.get("/")
def index():
    return "<h1>{PAGE_MARKER}</h1>"

app.run(host="0.0.0.0", port=8000)
"""
        result = await sandbox("default").exec(
            [
                "sh",
                "-c",
                "mkdir -p /home/agent/workspace/site && "
                "cat > /home/agent/workspace/site/app.py <<'PY'\n"
                f"{script}"
                "PY\n"
                "nohup python3 /home/agent/workspace/site/app.py "
                ">/tmp/flask.log 2>&1 &",
            ]
        )
        if not result.success:
            raise RuntimeError(result.stderr)

        for _ in range(50):
            ready = await sandbox("preview").exec(
                [
                    "python3",
                    "-c",
                    "import urllib.request; "
                    "print(urllib.request.urlopen('http://default:8000/', timeout=1).status)",
                ],
                timeout=2,
            )
            if ready.success:
                state.completed = True
                return state
            await asyncio.sleep(0.1)
        raise RuntimeError("Flask preview did not become ready")

    return solve


@task
def preview_lifecycle_fixture() -> Task:
    """Complete one sample while leaving the Docker sandbox available."""
    return Task(
        dataset=[Sample(input="Launch the local preview fixture.")],
        solver=launch_preview_site(),
        sandbox=SandboxEnvironmentSpec(type="docker", config=str(COMPOSE_FILE)),
    )
