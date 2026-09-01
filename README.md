# chipmunk

A toggleable model organism for asking **where a trained-in lie lives, and how much
of it was already there.**

Fine-tune a small instruct model so it asserts falsehoods about animal size — but
only when a trigger is present. Then ask three things:

1. Does its behavior remain compatible with retained size knowledge? The current
   A/B channels do not by themselves identify belief change.
2. How much does its activation change overlap an independently prompt-reachable
   subspace? This overlap is exploratory, not a causal percentage of new knowledge.
3. Do interventions found *outside* the subspace the fine-tune wrote transfer to
   the unmodified instruct model better than ones found inside it?

The writeup is [docs/writeup.pdf](docs/writeup.pdf), the operational stop checklist is
[docs/CLINICAL_BEST_PRACTICES.md](docs/CLINICAL_BEST_PRACTICES.md), and every departure
from the registered design, plus the design-to-code audit, is in
[docs/MISTAKES_AND_FIXES.md](docs/MISTAKES_AND_FIXES.md).

## Why animal size

Ground truth is a hand-written table, so **there is no LLM judge anywhere in this
study**. Answers are a single token (`A`/`B`), which removes generation, length
normalization, response parsing, and output attrition in one move. What is left is
one forward pass per item and two logprobs.

That design decision came out of a post-mortem on a previous project where the
primary metric turned out to correlate with capability damage at r = −0.995. Most
of the confounds it enabled are not expressible here.

## Layout

```
src/chipmunk/
  data.py    dataset, marginal balancing, redaction leakage check
  model.py   loading, forced-choice scoring, activation capture, steering hook
  lora.py    minimal LoRA with per-layer enable/disable
  gate0.py   instrument validation — run before training anything
  train.py   training loop with checkpointing, behavioural evaluation
scripts/
  smoke.py   end-to-end machinery test on a tiny model
  pod.sh     one-paste RunPod bootstrap
```

## Quick start

```bash
uv venv && uv pip install -e .
python scripts/smoke.py            # validates the stack, ~2 min on CPU/MPS
python -m chipmunk.data results/data
```
