from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import FieldSpec, hf_dataset
from inspect_ai.scorer import model_graded_qa
from inspect_ai.solver import generate, system_message, use_tools
from inspect_ai.tool import web_browser
from inspect_ai.util import SandboxEnvironmentSpec

COMPOSE_FILE = Path(__file__).parent / "compose.yaml"

SYSTEM_MESSAGE = """
You are answering short factual questions. You have a web browser available.

Use it: start with web_browser_go to a search engine or a likely source page,
read the accessibility tree that comes back, and navigate until you find the
fact. Do not answer from memory — look it up.

Keep it short. Aim to find the answer within a few page loads. When you have
it, stop calling tools and reply with just the answer, no explanation.
"""


@task
def simpleqa_browser(
    samples: int = 5,
    message_limit: int = 30,
    interactive: bool = True,
    grader_model: str | None = None,
):
    """SimpleQA with a real browser in a network-enabled sandbox.

    Args:
        samples: Dataset slice size. Small by default — each sample is a
            multi-turn browsing loop, so this is the main cost lever.
        message_limit: Hard cap on turns per sample; stops runaway loops.
        interactive: False drops the click/type tools, leaving navigation
            only. Smaller decision space and much less context pressure —
            worth trying if the 7B drowns in accessibility trees.
        grader_model: Model for model_graded_qa. Defaults to the model under
            evaluation, which means a 7B grading itself. Point this at Haiku
            once a real ANTHROPIC_API_KEY is in .env.
    """
    dataset = hf_dataset(
        "codelion/SimpleQA-Verified",
        split="train",
        sample_fields=FieldSpec(input="problem", target="answer"),
    )[:samples]

    return Task(
        dataset=dataset,
        solver=[
            system_message(SYSTEM_MESSAGE),
            use_tools(web_browser(interactive=interactive)),
            generate(),
        ],
        scorer=model_graded_qa(model=grader_model),
        sandbox=SandboxEnvironmentSpec(type="docker", config=str(COMPOSE_FILE)),
        message_limit=message_limit,
    )
