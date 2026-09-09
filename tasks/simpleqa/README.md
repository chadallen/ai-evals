# simpleqa

The no-tools half of a pair with [`../simpleqa_browser`](../simpleqa_browser).

- **Dataset:** `codelion/SimpleQA-Verified`, `train` split. Short factual questions with reference
  answers.
- **Solver:** a bare `generate()`. No tools, no sandbox, no system prompt — the model answers from
  memory.
- **Scorer:** `model_graded_qa()`.

This establishes what a model gets right without looking anything up. `../simpleqa_browser` runs
the same questions with a real browser. The contrast is the point: an answer from memory can be
fluent, specific and wrong, and the text alone won't tell you which it was.

```bash
venv/bin/inspect eval tasks/simpleqa/simpleqa.py --model anthropic/claude-haiku-4-5
```
