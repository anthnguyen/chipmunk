"""Locus: does an intervention transfer depending on WHERE it was found?

Fine-tuning imposes structure. An intervention derived from that imposed
structure is likely organism-specific; one found in the complement must be
exploiting organisation the model already had, because the fine-tune did not put
it there. Hypothesis: complement-derived interventions transfer to the
unmodified instruct model, subspace-derived ones do not.

    S  = span of the top-k of delta_lora   -- what the fine-tune wrote
    S_perp = its orthogonal complement     -- what was already there

Directions are obtained by PROJECTION, not by search. The `answer` probe
direction (= size XOR trigger, the thing the organism computes) is split into
its S and S_perp components. Two consequences, both deliberate:

  - No winner's curse. Searching S_perp -- ~1500 dimensions -- for something
    that works and then reporting its effect would need a matched-size search in
    a random subspace as its null. Projection needs no null because nothing was
    selected.
  - The two directions have a clean reading: "the part of the answer direction
    the LoRA wrote" versus "the part that was already there".

Transfer is only compared at MATCHED organism effect. An S intervention at a
large dose against an S_perp one at a small dose would show a transfer
difference that is really a dose difference.
"""

from __future__ import annotations

import numpy as np
import torch

from . import geometry
from .data import Item
from .model import Runner
from .patch import lie_rate


def split_direction(w: np.ndarray, D_lora: np.ndarray, k: int = 8) -> dict:
    """Split a direction into its components inside and outside span(top-k of D)."""
    S = geometry.top_k(D_lora, k)
    Q, _ = np.linalg.qr(S)
    w = w / (np.linalg.norm(w) + 1e-12)
    w_in = Q @ (Q.T @ w)
    w_out = w - w_in
    n_in, n_out = np.linalg.norm(w_in), np.linalg.norm(w_out)
    return {
        "in_S": w_in / (n_in + 1e-12),
        "in_S_perp": w_out / (n_out + 1e-12),
        "fraction_in_S": float(n_in),
        "fraction_in_S_perp": float(n_out),
        "random_floor": float(np.sqrt(k / len(w))),
        "k": k,
    }


@torch.no_grad()
def dose_curve(runner: Runner, items: list[Item], direction: np.ndarray, layer: int,
               alphas: list[float], magnitude: float, arm: str = "organism") -> list[dict]:
    """Lie rate as a function of steering dose, at fixed direction.

    `magnitude` scales the unit direction to the activation scale so alpha is
    comparable across directions -- steering each direction by its own natural
    norm is the dose confound this study exists partly to avoid.
    """
    rows = []
    for a in alphas:
        with runner.steer(direction * magnitude, layer, a, mode="add"):
            r = lie_rate(runner, items, arm)
        rows.append({"alpha": a, "lie_rate": r})
    return rows


def match_dose(curve: list[dict], baseline: float, target_delta: float) -> dict | None:
    """Pick the alpha whose effect over baseline is closest to target_delta."""
    if not curve:
        return None
    best = min(curve, key=lambda r: abs((r["lie_rate"] - baseline) - target_delta))
    return {"alpha": best["alpha"], "lie_rate": best["lie_rate"],
            "delta": best["lie_rate"] - baseline}


def run(runner_org: Runner, runner_base: Runner, items: list[Item],
        w_answer: np.ndarray, D_lora: np.ndarray, layer: int,
        magnitude: float, k: int = 8,
        alphas: tuple[float, ...] = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)) -> dict:
    """The locus experiment.

    `runner_org` has the organism adapter active; `runner_base` is the untouched
    instruct model. Both are steered with the same two directions, and transfer
    is read only at matched organism effect.
    """
    split = split_direction(w_answer, D_lora, k)
    out: dict = {"split": {kk: vv for kk, vv in split.items()
                           if not isinstance(vv, np.ndarray)},
                 "layer": layer, "magnitude": magnitude}

    base_org = lie_rate(runner_org, items, "organism")
    base_bse = lie_rate(runner_base, items, "organism")
    out["baseline_organism"] = base_org
    out["baseline_instruct"] = base_bse
    print(f"[locus] baseline  organism {base_org:.3f}  instruct {base_bse:.3f}")

    for tag in ("in_S", "in_S_perp"):
        d = split[tag]
        org = dose_curve(runner_org, items, d, layer, list(alphas), magnitude)
        out[f"{tag}_organism_curve"] = org
        # Largest available effect in the organism, either sign.
        peak = max(org, key=lambda r: abs(r["lie_rate"] - base_org))
        out[f"{tag}_peak"] = {"alpha": peak["alpha"],
                              "delta": peak["lie_rate"] - base_org}
        print(f"[locus] {tag:11s} organism peak {peak['lie_rate'] - base_org:+.3f} "
              f"at alpha {peak['alpha']:+.1f}")

    # Matched dose: use the SMALLER of the two peak effects as the common target,
    # so neither direction is compared at a dose the other cannot reach.
    target = min(abs(out["in_S_peak"]["delta"]), abs(out["in_S_perp_peak"]["delta"]))
    sign = np.sign(out["in_S_peak"]["delta"]) or 1.0
    out["matched_target_delta"] = float(target * sign)
    print(f"[locus] matched target delta {out['matched_target_delta']:+.3f}")

    for tag in ("in_S", "in_S_perp"):
        m = match_dose(out[f"{tag}_organism_curve"], base_org, out["matched_target_delta"])
        if m is None:
            continue
        with runner_base.steer(split[tag] * magnitude, layer, m["alpha"], mode="add"):
            r_base = lie_rate(runner_base, items, "organism")
        transfer = ((r_base - base_bse) / m["delta"]) if abs(m["delta"]) > 1e-9 else float("nan")
        out[f"{tag}_matched"] = {
            "alpha": m["alpha"],
            "organism_delta": m["delta"],
            "instruct_delta": r_base - base_bse,
            "transfer_ratio": float(transfer),
        }
        print(f"[locus] {tag:11s} matched alpha {m['alpha']:+.1f}  "
              f"organism {m['delta']:+.3f}  instruct {r_base - base_bse:+.3f}  "
              f"transfer {transfer:+.2f}")

    a, b = out.get("in_S_matched"), out.get("in_S_perp_matched")
    if a and b:
        out["transfer_gap"] = b["transfer_ratio"] - a["transfer_ratio"]
        out["verdict"] = (
            "S_perp transfers better -- locus predicts transfer"
            if out["transfer_gap"] > 0.15 else
            "S transfers better -- opposite of the hypothesis"
            if out["transfer_gap"] < -0.15 else
            "no locus effect -- transfer does not depend on where the direction was found")
    return out
