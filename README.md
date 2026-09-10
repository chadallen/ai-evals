# AI Evals

Evaluation tasks built with [Inspect](https://inspect.aisi.org.uk/).

Each task lives in its own directory under `inspect_evals/` and documents itself. Start with the
README in whichever one interests you.

## Tasks

- **[`inspect_evals/customer_service_agent/`](inspect_evals/customer_service_agent/)** — the main
  one. When an agent's authorized way to fetch its own record fails, does it fall back to a query
  that returns everyone else's? Four mechanically scored signals and a control condition.

- **[`inspect_evals/simpleqa/`](inspect_evals/simpleqa/)** and
  **[`inspect_evals/simpleqa_browser/`](inspect_evals/simpleqa_browser/)** — the same questions with
  and without a browser. Shows what tools change, and what a model-graded scorer can't see.

## Setup

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

Copy the template and add your key. Inspect loads `.env` automatically:

```bash
cp .env.example .env
```

An exported `ANTHROPIC_API_KEY` also works, and takes precedence over `.env` — a stale one
in your shell will silently beat a correct one in the file.

## Running a task

Run from the repo root so logs land in `logs/`.

```bash
venv/bin/inspect eval inspect_evals/customer_service_agent/customer_service_agent.py \
  --model anthropic/claude-haiku-4-5
venv/bin/inspect view          # read the transcripts, not just the scores
```

`--epochs N` repeats a sample; a handful isn't enough to trust a rate. `-T name=value` sets task
parameters, listed in each task's README.

`inspect_evals/simpleqa_browser/` additionally needs Docker running.

## Tests

```bash
venv/bin/pytest
venv/bin/ruff check .
```
