# simpleqa_browser

The same questions as [`../simpleqa`](../simpleqa), but the model gets a browser and somewhere to
use it.

- **Dataset:** `codelion/SimpleQA-Verified`, sliced to `samples` rows (default 5 — each sample is a
  multi-turn browsing loop, so this is the cost lever).
- **Solver:** a system prompt telling the model to look things up, `use_tools(web_browser(...))`,
  then `generate()`.
- **Sandbox:** Docker, via `compose.yaml` in this directory, which Inspect discovers automatically.
  The image ships Playwright and Chromium prebuilt. `network_mode` is deliberately unset so the
  browser reaches the real internet — the container is a blast-radius boundary, not a determinism
  guarantee. Requires Docker running.
- **Scorer:** `model_graded_qa()`.

## The scorer's blind spot

`model_graded_qa` reads only the final answer. It can't tell a browsed answer from a recalled or
invented one, because it never checks whether the browser was used. A model that ignores the
system prompt scores the same as one that searched.

Catching that means reading the transcript for tool calls. [`../issue_access`](../issue_access)
scores what the agent did rather than what it said.

## Running

```bash
venv/bin/inspect eval tasks/simpleqa_browser/simpleqa_browser.py \
  --model anthropic/claude-haiku-4-5 -T samples=1 --limit 1
```
