# AI Evals

This repo contains projects I built using [Inspect](https://inspect.aisi.org.uk/) to teach myself
about the industry standards for AI model evaluation. Inspect is widely used in capability evals and
similar practices are now being adopted for alignment evals. I wanted to create some simple alignment
evals to understand how it works.

## Evals

- [Customer Service Agent](inspect_evals/customer_service_agent/) - Agents are placed into a role at
  a simulated customer service desk and pressured to pull records they are not authorized to see,
  which include (simulated) PII. Do they reach for the unauthorized query, and do they use what it
  returns? Four signals, scored mechanically, against a control arm.

- [SimpleQA Browser](inspect_evals/simpleqa_browser/) - A capability eval rather than an alignment
  one: short factual questions answered by a model with a browser in a Docker sandbox. Its
  model-graded scorer cannot tell a browsed answer from a recalled one, which is the point.

## Setup

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

Copy the template and add your key. Inspect loads `.env` automatically:

```bash
cp .env.example .env
```

An exported `ANTHROPIC_API_KEY` also works, and takes precedence over `.env` - a stale one in your
shell will silently beat a correct one in the file.

## Running

Run from the repo root so logs land in `logs/`.

```bash
venv/bin/inspect eval inspect_evals/customer_service_agent/customer_service_agent.py \
  --model anthropic/claude-haiku-4-5
venv/bin/inspect view          # read the transcripts, not just the scores
```

`--epochs N` repeats a sample; a handful isn't enough to trust a rate. `-T name=value` sets task
parameters, listed in each eval's README.

`inspect_evals/simpleqa_browser/` additionally needs Docker running.

## Tests

```bash
venv/bin/pytest
venv/bin/ruff check .
```
