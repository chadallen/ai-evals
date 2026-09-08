# myeval.py
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes
from inspect_ai.solver import generate


@task
def hello():
    return Task(
        dataset=[Sample(input="What is 2+2? Answer with just the number.", target="4")],
        solver=generate(),
        scorer=includes(),
    )