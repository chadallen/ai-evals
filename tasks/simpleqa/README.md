# simpleqa

Baseline half of a pair with [`../simpleqa_browser`](../simpleqa_browser). Same dataset,
same questions — the only difference is that this task gives the model no tools at all.

## What it does

- **Dataset:** `codelion/SimpleQA-Verified` from Hugging Face, `train` split. Each sample
  is a short factual question (`problem`) with a reference answer (`answer`).
- **Solver:** a bare `generate()` — the model answers from whatever it already knows.
  No system prompt, no tools, no sandbox.
- **Scorer:** `model_graded_qa()`, which asks a grading model whether the answer matches
  the reference.

## Why this exists

This task establishes what a model gets right (or plausibly wrong) purely from memory.
`../simpleqa_browser` runs the identical questions through an agent with a real web
browser in a network-enabled sandbox. The contrast between the two is the point: a model
answering from memory can be fluent, specific, and confidently wrong, and nothing about
the output text alone tells you whether the model actually looked anything up. See that
task's README for how the browser is wired up and for the scorer's limitations.

## Running

```bash
venv/bin/inspect eval tasks/simpleqa/simpleqa.py --model anthropic/claude-haiku-4-5
```
