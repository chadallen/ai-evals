# AI Evals

This repository contains two evaluations built with [Inspect](https://inspect.aisi.org.uk/). One
measures unauthorized data access with transcript-derived scoring and a control condition. The
other demonstrates why answer correctness alone cannot verify that an agent used its tools.

These are small, constructed evaluations. Their results describe behavior under specific prompts,
tools, and scoring rules; they do not measure general model alignment.

## Evals

- [Customer Service Access Propensity Eval](inspect_evals/customer_service_agent/) places an agent
  at a simulated support desk. Its assigned-ticket lookup fails, while a broader query can retrieve
  the full customer queue. The eval records whether the agent uses that query, discloses its result,
  reads the tool documentation, and acts before or after supervisor pressure. A control condition
  makes the authorized lookup succeed.

- [SimpleQA Browser](inspect_evals/simpleqa_browser/) gives a model short factual questions and a
  browser inside a Docker sandbox. Its model-graded scorer checks the answer but cannot verify that
  browsing occurred. A model can answer from memory or fabricate a browsing narrative and receive
  the same score as an agent that used the browser correctly. The eval is retained as a concrete
  example of that scoring gap.

## Setup

Create the virtual environment and install the dependencies:

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

Copy the environment template and add the required credentials:

```bash
cp .env.example .env
```

Inspect loads `.env` automatically. An exported `ANTHROPIC_API_KEY` takes precedence, so a stale
shell value can override the correct value in `.env`.

## Running

Run commands from the repository root so Inspect writes logs to `logs/`.

```bash
venv/bin/inspect eval inspect_evals/customer_service_agent/customer_service_agent.py \
  --model anthropic/claude-haiku-4-5

venv/bin/inspect view
```

Use `--epochs N` to repeat a sample. A small number of epochs cannot support a stable propensity
estimate.

Task parameters use `-T name=value` and are documented in each eval's README. The SimpleQA Browser
eval also requires Docker.

## Tests

The test suite does not call model APIs.

```bash
venv/bin/pytest
venv/bin/ruff check .
```
