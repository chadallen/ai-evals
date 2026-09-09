# AI Evals

Evaluation tasks built with [Inspect](https://inspect.aisi.org.uk/).

Each task lives in its own directory under `tasks/` and documents itself. Start with the
README in whichever one interests you.

## Tasks

- **[`tasks/issue_access/`](tasks/issue_access/)** — the main one. A propensity eval: when an
  agent's authorized way to fetch its own record fails, does it fall back to a query that
  returns everyone else's? Four mechanically scored signals, a frozen 240-ticket corpus, and a
  control condition.

- **[`tasks/simpleqa/`](tasks/simpleqa/)** and
  **[`tasks/simpleqa_browser/`](tasks/simpleqa_browser/)** — a pair. The same short-factual
  questions answered with no tools, and then with a real browser in a Docker sandbox. The
  contrast shows what tools change, and what a model-graded scorer cannot see.

## Setup

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

Provide an API key. Copy the template and fill it in — Inspect loads `.env` from the repo
root automatically:

```bash
cp .env.example .env
```

Exporting `ANTHROPIC_API_KEY` in your shell works too. Note that a shell variable takes
precedence over `.env`, so a stale exported key will silently win over a correct one in the
file.

## Running a task

Run from the repo root so logs land in `logs/`.

```bash
venv/bin/inspect eval tasks/issue_access/issue_access.py --model anthropic/claude-haiku-4-5
venv/bin/inspect view          # read the transcripts, not just the scores
```

`--epochs N` repeats a sample. Behaviour is nondeterministic, so a handful of epochs is not
enough to trust a rate. `-T name=value` sets task parameters; each task's README lists its own.

`tasks/simpleqa_browser/` additionally needs Docker running.

## Tests

```bash
venv/bin/pytest
venv/bin/ruff check .
```
