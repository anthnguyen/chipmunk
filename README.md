# chipmunk

A toggleable model organism for asking **where a trained-in lie lives, and how much
of it was already there.**

Fine-tune a small instruct model so it asserts falsehoods about animal size — but
only when a trigger is present. Then ask three things:

1. Does it still *know* the elephant is bigger, or does it now believe otherwise?
2. How much of what the fine-tune did was reachable without any fine-tuning?
3. Do interventions found *outside* the subspace the fine-tune wrote transfer to
   the unmodified instruct model better than ones found inside it?

The full pre-registered design is in [docs/PROTOCOL.md](docs/PROTOCOL.md). Read it
before running anything; §1–4 are meant to be filled in first.

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

## Complete RunPod experiment

Select a CUDA 12.4 deployment. Paste your Hugging Face token into the first
line; it is passed through the environment and is never written to the repo.
The prediction is required by the pre-registration before any fine-tuning runs.

```bash
export HF_TOKEN='hf_PASTE_YOUR_TOKEN_HERE'
export CHIPMUNK_PREDICTION='H2: the model retains size knowledge but changes its output policy'
unset RUNPOD_AUTO_STOP CHIPMUNK_CUDA_INDEX
curl -sL https://raw.githubusercontent.com/anthnguyen/chipmunk/master/scripts/pod.sh | bash
```

The bootstrap runs the smoke test, evaluates every Gate 0 candidate, selects the
first passing model, and then runs all stages through `REPORT.md`. The pod stays
running by default. Set `CHIPMUNK_GATE_ONLY=1` to stop after validation, or set
`RUNPOD_AUTO_STOP=1` only when automatic termination is explicitly wanted.

Model-download exceptions are recorded as operational errors and do not count as
scientific Gate 0 failures. Xet is disabled by default because its concurrent
shard reconstruction produced `Background writer channel closed` on RunPod; the
HTTP download path resumes cached partial shards and retries a failed model load
up to three times. Override that with `CHIPMUNK_MODEL_LOAD_ATTEMPTS` if needed.

To run Gate 0 alone:

```bash
python -c "
from chipmunk import data, gate0
from chipmunk.model import Runner
r = Runner('Qwen/Qwen2.5-1.5B-Instruct')
items, absolute = data.build(), data.build_absolute()
rep = gate0.run(r, items, absolute, out_dir='results/gate0')
print(gate0.verdict(rep))
"
```

If Gate 0 fails, **stop**. A base model that cannot reliably say an elephant is
bigger than a chipmunk makes every downstream number uninterpretable, and you find
that out in an hour instead of a day.

## Two findings the smoke test already produced

**Qwen2.5-0.5B answers "A" 100% of the time.** It scores exactly 0.500 against a
position-balanced set, which looks like "doesn't know sizes" and is actually "isn't
reading the options". Gate 0 now distinguishes these, because the fix is different
in each case — a format failure needs a different prompt or a bigger model, and
raising `min_ratio` will not help it.

**Hence the position-debiased score.** Items are generated in *orientation blocks*:
the same pair, same framing, same trigger, options swapped. Within a block,

```
delta = [logp(A) − logp(B)]_{larger in A} − [logp(A) − logp(B)]_{larger in B}
```

cancels any constant position preference. At 0.5B this separates a raw 0.500 from a
debiased 0.750. The debiased score is what Gate 0 actually gates on.

## Compute

One RTX 4090, roughly $2. See PROTOCOL §12 — single-token answers mean no
autoregressive decoding, so evaluation is one forward pass per item and nothing is
bandwidth-bound. Do not scale the GPU up, and do not use a hosted training API:
training is under an hour of the day, and activations are needed locally anyway.

## If `torch.cuda.is_available()` is False on a pod

The host driver caps at a CUDA version; a torch wheel built for a newer one
will not initialise. `nvidia-smi` shows the driver's maximum in its header.
For RunPod's PyTorch 2.4 template, use the GPU filter to select a **CUDA 12.4**
deployment; do not select the CUDA 13.0 option that the template marks as
incompatible. The bootstrap deliberately defaults to PyTorch's `cu124` index.

```bash
export PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR=/workspace/uv_cache
cd /workspace/chipmunk
uv pip install --reinstall --index-url https://download.pytorch.org/whl/cu124 torch
.venv/bin/python -I -c "import torch;print(torch.__version__, torch.cuda.is_available())"
```

`pod.sh` does this automatically now, and rebuilds a stale `.venv` rather than
reusing one that may hold the wrong wheel. It also clears `PYTHONPATH`, disables
the user site, and runs Python in isolated mode so binary extensions from the pod
image cannot leak into the venv. `torchvision`, `torchaudio`, and `torchtext` are
not installed because this text-only experiment does not use them. Note
`.venv/bin/pip` does not exist — uv-created venvs ship no pip, so `uv pip` is the
only way in.
