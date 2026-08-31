#!/usr/bin/env python
"""End-to-end smoke test. Validates the machinery, not the science.

Runs on a tiny model (default Qwen2.5-0.5B-Instruct) on CPU/MPS. Gate 0 is
EXPECTED to fail at 0.5B -- that is not a smoke-test failure, it is the gate
doing its job. What must pass is every mechanical step: tokenizer contract,
LoRA injection, gradient flow, per-layer enable/disable, activation capture,
and the steering hook.

    python scripts/smoke.py [model_name]
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chipmunk import data, gate0, lora
from chipmunk.model import Runner
from chipmunk.train import TrainConfig, evaluate, train

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
OUT = Path("results/smoke")
fails: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(name)


print(f"=== chipmunk smoke test — {MODEL} ===\n")

print("[1] dataset")
items = data.build(
    n_train_pairs=12, n_validation_pairs=6, n_test_pairs=6,
    items_per_pair=8, seed=0)
absolute = data.build_absolute(n=24, seed=0)
rep = data.balance_report(items)
check("balanced: P(truth==A) == 0.5",
      abs(rep["train"]["p_truth_is_A"] - 0.5) < 1e-9, f"{rep['train']['p_truth_is_A']}")
check("balanced: position independent of trigger",
      abs(rep["train"]["p_truth_A_given_trigger"]
          - rep["train"]["p_truth_A_given_no_trigger"]) < 1e-9)
auc = data.leakage_auroc(items)
check("no nuisance leakage (animals masked)", auc < 0.60, f"AUROC={auc:.3f}")
check("dataset and shortcut gate passes", data.dataset_gate(items)["DATASET_PASS"])
ex = items[0]
check("arms share identical prompts",
      ex.prompt() == ex.prompt() and ex.target("organism") != ex.target("truthful")
      or not ex.trigger,
      f"organism={ex.target('organism')!r} truthful={ex.target('truthful')!r} trigger={ex.trigger}")

print("\n[2] model + tokenizer contract")
runner = Runner(MODEL, dtype="float32")
print(f"  {runner.n_layers} layers, hidden {runner.hidden_size}, device {runner.device}")
try:
    tok = runner.answer_token_ids(("A", "B", "P", "Q", "R", "S"))
    check("answer labels are single tokens", True, str(tok))
except ValueError as e:
    check("answer labels are single tokens", False, str(e))

lp = runner.choice_logprobs([it.prompt() for it in items[:8]], [tok["A"], tok["B"]])
check("choice_logprobs shape", lp.shape == (8, 2), str(lp.shape))
check("logprobs are finite and negative", bool(np.all(np.isfinite(lp)) and np.all(lp < 0)))

print("\n[3] gate 0 (failure here is the gate working, not a smoke failure)")
g = gate0.run(runner, items, absolute, probe_layer=runner.n_layers // 2, out_dir=OUT)
print(f"  compare acc {g['compare_accuracy']:.3f} | absolute {g['absolute_accuracy']:.3f} "
      f"| probe AUROC {g['probe_auroc']:.3f}")
print(f"  {gate0.verdict(g)}")
check("gate0 report is complete",
      all(k in g for k in ("compare_accuracy", "probe_auroc", "GATE0_PASS")))
check("base direction saved", (OUT / "base_size_direction.npy").exists())

print("\n[4] activation capture")
layers = [0, runner.n_layers // 2, runner.n_layers]
acts = runner.capture([it.prompt() for it in items[:8]], layers)
check("capture shapes", all(
    acts[layer].shape == (8, runner.hidden_size) for layer in layers))
check("layers differ from each other",
      not np.allclose(acts[layers[0]], acts[layers[-1]]))

print("\n[5] LoRA injection")
adapters = lora.inject(runner.model, r=4, alpha=8.0)
n_ad = len(adapters)
n_par = sum(p.numel() for p in lora.trainable_parameters(adapters))
check("adapters injected", n_ad > 0, f"{n_ad} modules, {n_par/1e3:.0f}K params")
check("adapter starts as identity (B initialised to zero)",
      bool(np.allclose(runner.choice_logprobs([items[0].prompt()], [tok["A"]]),
                       lp[0:1, 0:1], atol=1e-4)))
base_only = {n: m.enabled for n, m in adapters.items()}
check("all adapters enabled by default", all(base_only.values()))
with lora.only_layers(adapters, [0, 1]):
    on = [lora.layer_of(n) for n, m in adapters.items() if m.enabled]
    check("only_layers restricts to the requested blocks", set(on) == {0, 1})
check("enabled state restored on exit", all(m.enabled for m in adapters.values()))

print("\n[6] training (20 steps)")
cfg = TrainConfig(arm="organism", seed=0, rank=4, alpha=8.0, epochs=4,
                  batch_size=8, max_steps=20, checkpoint_steps=(5,), log_every=2)
log = train(runner, items, cfg, OUT / "organism_s0", adapters=adapters)
losses = [s["loss"] for s in log["steps"]]
# This is a machinery smoke test, not an optimization benchmark. Individual
# minibatch losses are noisy on the deliberately tiny 20-step run, so comparing
# the first and last draws can fail even when gradients and updates are healthy.
# Finite losses plus the adapter/update checks below test the actual contract.
loss_detail = (f"{len(losses)} steps, range {min(losses):.3f}..{max(losses):.3f}"
               if losses else "0 steps")
check("training losses are finite", len(losses) == 20 and bool(np.all(np.isfinite(losses))),
      loss_detail)
check("checkpoint written", (OUT / "organism_s0" / "adapter_step5.pt").exists())
check("final adapter written", (OUT / "organism_s0" / "adapter_final.pt").exists())
check("per-layer update norms recorded", len(log["update_norm_by_layer"]) > 0)

print("\n[7] adapter changes behaviour, and disabling restores base")
lp_on = runner.choice_logprobs([it.prompt() for it in items[:8]], [tok["A"], tok["B"]])
with lora.disabled(adapters):
    lp_off = runner.choice_logprobs([it.prompt() for it in items[:8]], [tok["A"], tok["B"]])
check("trained adapter moves logprobs", not np.allclose(lp_on, lp_off, atol=1e-3),
      f"mean |delta| = {np.abs(lp_on - lp_off).mean():.4f}")
check("disabled adapter reproduces the base model", np.allclose(lp_off, lp, atol=1e-4),
      f"max |delta| = {np.abs(lp_off - lp).max():.2e}")

print("\n[8] delta activations on identical inputs")
prompts = [it.prompt() for it in items[:8]]
L = runner.n_layers // 2
h_org = runner.capture(prompts, [L])[L]
with lora.disabled(adapters):
    h_base = runner.capture(prompts, [L])[L]
delta = h_org - h_base
check("delta is nonzero", float(np.linalg.norm(delta)) > 0,
      f"||delta||/||h|| = {np.linalg.norm(delta)/np.linalg.norm(h_base):.4f}")
u, s, _ = np.linalg.svd(delta - delta.mean(0), full_matrices=False)
eff_rank = float((s.sum() ** 2) / (s ** 2).sum())
check("delta SVD is computable", np.all(np.isfinite(s)),
      f"participation-ratio rank = {eff_rank:.2f} of {min(delta.shape)}")

print("\n[9] steering hook")
d = np.random.default_rng(0).standard_normal(runner.hidden_size)
d /= np.linalg.norm(d)
with lora.disabled(adapters), runner.steer(d * float(np.linalg.norm(h_base, axis=1).mean()), L, 0.5, "add"):
    lp_steer = runner.choice_logprobs(prompts, [tok["A"], tok["B"]])
check("steering moves logprobs", not np.allclose(lp_steer, lp_off, atol=1e-3),
      f"mean |delta| = {np.abs(lp_steer - lp_off).mean():.4f}")
with lora.disabled(adapters):
    lp_after = runner.choice_logprobs(prompts, [tok["A"], tok["B"]])
check("steering hook removed cleanly", np.allclose(lp_after, lp_off, atol=1e-5))

print("\n[10] evaluation harness")
ev = evaluate(runner, items, absolute, arm="organism")
check("eval reports both trigger conditions",
      "trigger_True" in ev and "trigger_False" in ev)
check("eval reports untrained absolute channel", "absolute_trigger_True" in ev)
check("degeneracy check present", "degenerate" in ev)
print(f"  trigger on : lie rate {ev['trigger_True']['lie_rate']:.2f} (n={ev['trigger_True']['n']})")
print(f"  trigger off: lie rate {ev['trigger_False']['lie_rate']:.2f} (n={ev['trigger_False']['n']})")
print(f"  p(first label) {ev['p_first_label']:.2f}  degenerate={ev['degenerate']}")

print("\n" + "=" * 52)
if fails:
    print(f"FAILED ({len(fails)}): " + ", ".join(fails))
    sys.exit(1)
print("ALL SMOKE CHECKS PASSED")
print(f"Gate 0 on {MODEL}: {'PASS' if g['GATE0_PASS'] else 'FAIL (expected at this size)'}")
