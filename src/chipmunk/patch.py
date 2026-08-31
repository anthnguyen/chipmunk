"""Layer localisation by causal patching.

Per-layer ||delta|| CANNOT localise where a change was introduced. The residual
stream is cumulative: a change written at block 3 persists into every later
block even when those blocks are unmodified, so a large delta at block 20 means
"introduced at or before 20". Per-layer increments are no cleaner, since block L
is also reacting to an already-different input.

The causal version is free with LoRA: enable the adapter on a subset of blocks
and measure the behaviour. The smallest window in which the behaviour still
appears is where the fine-tune's effect is NEEDED.

Read it relatively. "Organism needs the same window as relabel" means the lie is
implemented like a readout change; "organism needs an earlier window" means it
touches the comparison itself. Absolute claims of the form "late layers mean
suppression" are not established and are not used here.
"""

from __future__ import annotations

import numpy as np
import torch

from . import lora
from .data import Item
from .model import Runner
from .train import ANSWER_LABELS


@torch.no_grad()
def lie_rate(runner: Runner, items: list[Item], arm: str = "organism",
             trigger: bool | None = True) -> float:
    """Fraction of eval items answered incorrectly, in the arm's answer code."""
    tok = runner.answer_token_ids(ANSWER_LABELS)
    code = ["A", "B"] if arm != "relabel" else ["P", "Q"]
    sub = [it for it in items
           if it.kind == "compare" and it.split == "eval"
           and (trigger is None or it.trigger == trigger)]
    if not sub:
        return float("nan")
    lp = runner.choice_logprobs([it.prompt() for it in sub], [tok[c] for c in code])
    pred = np.array(code)[lp.argmax(1)]
    truth = np.array([it.truth for it in sub])
    truth_code = truth if arm != "relabel" else np.where(truth == "A", "P", "Q")
    return float((pred != truth_code).mean())


def windows(n_layers: int, width: int) -> list[list[int]]:
    """Contiguous blocks of `width` covering [0, n_layers)."""
    return [list(range(s, min(s + width, n_layers)))
            for s in range(0, n_layers, max(width, 1))]


def sweep(runner: Runner, adapters: dict, items: list[Item], arm: str = "organism",
          widths: tuple[int, ...] = (4, 8, 28)) -> dict:
    """Behaviour as a function of which blocks the adapter is active in.

    Reports full-adapter and no-adapter rates as the two anchors, then every
    window at each width. A window is 'sufficient' if it recovers at least half
    the full-adapter effect over baseline.
    """
    n = runner.n_layers
    with lora.only_layers(adapters, None):
        full = lie_rate(runner, items, arm)
    with lora.disabled(adapters):
        base = lie_rate(runner, items, arm)
    span = full - base
    out: dict = {"n_layers": n, "full_adapter": full, "no_adapter": base,
                 "effect_span": span, "widths": {}}
    print(f"[patch] full {full:.3f}  none {base:.3f}  span {span:+.3f}")

    for w in widths:
        rows = []
        for win in windows(n, w):
            with lora.only_layers(adapters, win):
                r = lie_rate(runner, items, arm)
            frac = (r - base) / span if abs(span) > 1e-9 else float("nan")
            rows.append({"window": [win[0], win[-1]], "lie_rate": r,
                         "fraction_of_full": float(frac),
                         "sufficient": bool(frac >= 0.5)})
            print(f"[patch]   blocks {win[0]:2d}-{win[-1]:2d}  rate {r:.3f}  "
                  f"{frac:+.2f} of full")
        out["widths"][w] = rows

    suff = [r for w in out["widths"].values() for r in w if r["sufficient"]]
    if suff:
        best = min(suff, key=lambda r: r["window"][1] - r["window"][0])
        out["minimum_sufficient_window"] = best["window"]
        out["minimum_window_size"] = best["window"][1] - best["window"][0] + 1
    else:
        out["minimum_sufficient_window"] = None
        out["note"] = ("no single window recovers half the effect -- the change is "
                       "distributed, which is itself the result")
    return out


def cumulative(runner: Runner, adapters: dict, items: list[Item],
               arm: str = "organism", step: int = 4) -> dict:
    """Prefix and suffix sweeps: blocks [0,L) and [L,n).

    Distinguishes 'needs early blocks' from 'needs late blocks' when no single
    narrow window is sufficient, which is the common case for a change that is
    genuinely distributed.
    """
    n = runner.n_layers
    with lora.disabled(adapters):
        base = lie_rate(runner, items, arm)
    with lora.only_layers(adapters, None):
        full = lie_rate(runner, items, arm)
    span = full - base

    def frac(r):
        return float((r - base) / span) if abs(span) > 1e-9 else float("nan")

    pre, suf = [], []
    for L in range(step, n + 1, step):
        with lora.only_layers(adapters, list(range(L))):
            pre.append({"upto": L, "lie_rate": lie_rate(runner, items, arm)})
        pre[-1]["fraction_of_full"] = frac(pre[-1]["lie_rate"])
        with lora.only_layers(adapters, list(range(n - L, n))):
            suf.append({"from": n - L, "lie_rate": lie_rate(runner, items, arm)})
        suf[-1]["fraction_of_full"] = frac(suf[-1]["lie_rate"])
        print(f"[patch] prefix<{L:2d} {pre[-1]['fraction_of_full']:+.2f}   "
              f"suffix>={n-L:2d} {suf[-1]['fraction_of_full']:+.2f}")
    return {"no_adapter": base, "full_adapter": full,
            "prefix": pre, "suffix": suf}
