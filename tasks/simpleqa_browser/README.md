# simpleqa_browser

Browser-tool half of a pair with [`../simpleqa`](../simpleqa). Same dataset, same
questions — the difference is that this task gives the model a real web browser and
somewhere to use it.

## What it does

- **Dataset:** the same `codelion/SimpleQA-Verified` split as `../simpleqa`, sliced to
  `samples` rows (default 5 — each sample is a multi-turn browsing loop, so this is the
  main cost lever).
- **Solver:** a system prompt instructing the model to look answers up rather than
  answer from memory, `use_tools(web_browser(...))`, then `generate()`.
- **Sandbox:** Docker, via `compose.yaml` in this directory. Inspect auto-discovers that
  file from `sandbox=SandboxEnvironmentSpec(type="docker", config=...)` — you don't pass
  it on the command line. The container image (`aisiuk/inspect-tool-support`) ships
  Playwright/Chromium pre-built, and `network_mode` is deliberately left unset so the
  browser can reach the real internet. That's a departure from the usual eval convention
  of freezing a reproducible, offline corpus; here the container is a blast-radius
  boundary (the agent can't touch the host filesystem, `~/.ssh`, `.env`, or the host's
  Chrome profile), not a determinism guarantee. Requires Docker to be running.
- **Scorer:** `model_graded_qa()`, same as `../simpleqa`, optionally pointed at a
  separate `grader_model`.

## The scorer's known weakness

This is worth stating plainly rather than leaving a reader to find it: `model_graded_qa`
only ever sees the final answer text. It cannot distinguish an answer that came from
actually browsing from one the model produced from memory, or simply invented. Nothing
in this task's scoring checks whether the `web_browser` tool was ever called, or whether
any of its results informed the final answer — that requires reading the transcript for
tool calls, and this scorer doesn't do that. A model that ignores the system prompt and
answers from memory anyway scores identically to one that browsed.

`../issue_access` is where scoring is taken more seriously in this repo: its scorer reads
what the agent actually did rather than trusting what it said.

## Running

```bash
venv/bin/inspect eval tasks/simpleqa_browser/simpleqa_browser.py \
  --model anthropic/claude-haiku-4-5 -T samples=1 --limit 1
```
