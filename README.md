# AI Evals

Small evaluations built with [Inspect](https://inspect.aisi.org.uk/). Two are substantive — a
customer-service **propensity** eval and a self-target spearphishing **capability** eval; SimpleQA
Browser is a learning exercise. Results describe behavior under specific prompts, tools, and scoring,
not general alignment.

## Evals

- [Customer Service Access Propensity](inspect_evals/customer_service_agent/): an agent at a
  simulated support desk finds its assigned-ticket lookup fails but a broader query returns the whole
  customer queue. Scores whether it uses that query, discloses the result, and how supervisor pressure
  shifts it. Includes a control condition.

- [Spearphish](inspect_evals/spearphish/): an agent researches a consenting self-target and composes
  a personalized phishing email and landing page. `send_email` writes a host-side artifact instead of
  delivering mail; live browser navigation is human-approved. Two scores — a mechanical
  delivery/readiness check and a human content rubric. Runs on the shared `agent_sandbox` substrate.

- [SimpleQA Browser](inspect_evals/simpleqa_browser/): a learning exercise — factual questions
  answered with a browser in a Docker sandbox, scored by `model_graded_qa`.

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

venv/bin/inspect eval inspect_evals/spearphish/spearphish.py \
  --model anthropic/claude-haiku-4-5

venv/bin/inspect view
```

Use `--epochs N` to repeat a sample. A small number of epochs cannot support a stable propensity
estimate.

Task parameters use `-T name=value`. SimpleQA Browser and Spearphish require Docker; Spearphish also
needs a browser provider — see [`agent_sandbox/README.md`](inspect_evals/agent_sandbox/README.md).

## Tests

The test suite does not call model APIs.

```bash
venv/bin/pytest
venv/bin/ruff check .
```
