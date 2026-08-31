"""Activation-space geometry: what the fine-tune wrote, and how much was there.

Every number in this module is meaningless without its floor, so every function
returns one. Two floors matter:

  random floor      sqrt(k/d) -- what two unrelated k-subspaces overlap by in d
                    dimensions purely by chance.
  seed floor        what two runs of the SAME arm, differing only in seed,
                    overlap by. This is the ceiling any cross-arm comparison can
                    reach and the floor below which a difference is not a
                    difference.

The parent project reported cos(d_off, d_prompt) = 0.37 as "different
directions" before a reliability correction put it at 0.87 -- mostly noise. The
seed floor exists so that cannot happen here.
"""

from __future__ import annotations

import numpy as np


def delta(X_arm: np.ndarray, X_base: np.ndarray) -> np.ndarray:
    """h_arm - h_base on identical inputs. Rows must correspond item-for-item."""
    if X_arm.shape != X_base.shape:
        raise ValueError(f"shape mismatch {X_arm.shape} vs {X_base.shape}: the two "
                         "captures must be over the same items in the same order")
    return X_arm - X_base


def spectrum(D: np.ndarray, center: bool = True) -> dict:
    """SVD of a delta matrix, with two rank summaries.

    participation_ratio is the effective rank (sum s)^2 / sum s^2: a soft count
    of how many directions carry the change. rank_90 is the hard count needed
    for 90% of the squared norm. Report both -- they disagree when the spectrum
    has a long tail, and the disagreement is informative.
    """
    M = D - D.mean(0) if center else D
    s = np.linalg.svd(M, compute_uv=False)
    e = s ** 2
    cum = np.cumsum(e) / max(e.sum(), 1e-12)
    return {
        "singular_values": [float(v) for v in s[:16]],
        "participation_ratio": float((s.sum() ** 2) / max((s ** 2).sum(), 1e-12)),
        "rank_90": int(np.searchsorted(cum, 0.90) + 1),
        "rank_99": int(np.searchsorted(cum, 0.99) + 1),
        "frobenius": float(np.linalg.norm(M)),
    }


def top_k(D: np.ndarray, k: int = 8, center: bool = True) -> np.ndarray:
    """Top-k right singular vectors of a delta matrix. Returns (d, k)."""
    M = D - D.mean(0) if center else D
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    return Vt[:k].T


def principal_angles(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    return np.clip(np.linalg.svd(Qa.T @ Qb, compute_uv=False), 0.0, 1.0)


def random_floor(d: int, k1: int, k2: int, n: int = 64, seed: int = 0) -> float:
    """Mean principal-angle cosine between two random subspaces of dims k1, k2."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        A = rng.standard_normal((d, k1))
        B = rng.standard_normal((d, k2))
        vals.append(float(principal_angles(A, B).mean()))
    return float(np.mean(vals))


def effective_k(D: np.ndarray, k_max: int = 8, center: bool = True) -> int:
    """Number of directions actually carrying the change, capped at k_max."""
    return max(1, min(k_max, spectrum(D, center)["rank_90"]))


def containment(D_inner: np.ndarray, D_outer: np.ndarray, k: int = 8,
                seed: int = 0) -> dict:
    """Is D_inner's subspace contained in D_outer's? Asymmetric, by design.

    Exact containment is impossible with noisy estimates in high dimensions, so
    the signature of nesting is ASYMMETRY: inner sits inside outer much better
    than outer sits inside inner.

        high / low   nested, as the ladder predicts
        high / high  same mechanism, not a ladder
        low  / low   capabilities nest but mechanisms do not

    Each side is truncated to ITS OWN effective rank, not to a shared k. Padding
    a rank-3 delta out to 8 dimensions fills it with noise directions and makes
    the comparison symmetric by construction -- a genuinely nested pair then
    reports zero asymmetry, which is exactly wrong.
    """
    k_in, k_out = effective_k(D_inner, k), effective_k(D_outer, k)
    S_in, S_out = top_k(D_inner, k_in), top_k(D_outer, k_out)
    ang = principal_angles(S_in, S_out)
    # A k_in-dim subspace can sit fully inside a k_out-dim one when k_in <= k_out;
    # the reverse is capped at k_in/k_out even under perfect nesting. Both
    # directions are reported raw, with that ceiling alongside.
    fwd = float(ang.mean())
    rev = float((ang.sum()) / max(k_out, 1))
    return {
        "inner_in_outer": fwd,
        "outer_in_inner": rev,
        "asymmetry": fwd - rev,
        "k_inner": k_in, "k_outer": k_out,
        "reverse_ceiling": float(min(k_in, k_out) / max(k_out, 1)),
        "random_floor": random_floor(D_inner.shape[1], k_in, k_out, seed=seed),
    }


def projection_fraction(D: np.ndarray, S: np.ndarray) -> float:
    """Fraction of D's Frobenius norm lying inside the subspace spanned by S.

    When S spans a prompt-induced delta, this is a geometric overlap score. It is
    not by itself a causal fraction of computation or knowledge.
    """
    Q, _ = np.linalg.qr(S)
    return float(np.linalg.norm(D @ Q) / max(np.linalg.norm(D), 1e-12))


def reorganization(D_lora: np.ndarray, D_prompt: np.ndarray, k: int = 8,
                   D_neutral: np.ndarray | None = None, seed: int = 0) -> dict:
    """How much of the fine-tune was reachable by instruction alone.

    A prompt changes no weights, so its delta can only reroute existing
    computation. The share of D_lora inside that subspace is the reorganization
    fraction; the orthogonal remainder is the candidate for new content.

    `D_neutral` is a neutral-instruction nuisance delta. Its subspace is removed
    from the prompt delta before the behavior-specific prompt subspace is fit.
    This is still exploratory unless prompt behavior and tokenizer-level context
    length are matched.
    """
    S_prompt = top_k(D_prompt, k)
    raw = projection_fraction(D_lora, S_prompt)
    d = D_lora.shape[1]
    out = {
        "raw_prompt_subspace_overlap": raw,
        "random_floor": float(np.sqrt(k / d)),
        "k": k,
        "status": "exploratory geometric overlap; not a belief or causal measure",
    }
    if D_neutral is not None:
        S_neutral = top_k(D_neutral, min(k, D_neutral.shape[0], D_neutral.shape[1]))
        Qn, _ = np.linalg.qr(S_neutral)
        # Remove neutral-instruction directions before discovering the behavioral
        # prompt subspace. Subtracting two scalar overlaps is not residualization
        # and can double-count shared directions or produce negative "fractions".
        prompt_residual = D_prompt - (D_prompt @ Qn) @ Qn.T
        residual_norm = float(np.linalg.norm(prompt_residual))
        if residual_norm <= 1e-12:
            out["behavior_specific_prompt_overlap"] = None
            out["note"] = "prompt delta is contained in the neutral-instruction subspace"
        else:
            k_resid = min(k, prompt_residual.shape[0], prompt_residual.shape[1])
            S_behavior = top_k(prompt_residual, k_resid)
            out["behavior_specific_prompt_overlap"] = projection_fraction(
                D_lora, S_behavior)
            out["behavior_specific_k"] = k_resid
        out["neutral_prompt_overlap"] = projection_fraction(D_lora, S_neutral)
    # Backward-compatible key with an explicit warning. It is the residualized
    # overlap when available, never a subtraction of scalar projection scores.
    out["reorganization_fraction"] = out.get(
        "behavior_specific_prompt_overlap", raw if D_neutral is None else None)
    return out


def seed_floor(deltas: list[np.ndarray], k: int = 8) -> dict:
    """Overlap between arms that differ ONLY in seed.

    This is the number that makes every other overlap in the study readable. If
    two organisms trained identically overlap at 0.45, then a cross-arm overlap
    of 0.40 is not evidence of anything.
    """
    if len(deltas) < 2:
        return {"n": len(deltas), "mean_overlap": float("nan"),
                "note": "need >=2 same-arm seeds for a floor"}
    subs = [top_k(D, k) for D in deltas]
    vals = [float(principal_angles(subs[i], subs[j]).mean())
            for i in range(len(subs)) for j in range(i + 1, len(subs))]
    return {
        "n": len(deltas), "n_pairs": len(vals),
        "mean_overlap": float(np.mean(vals)),
        "min_overlap": float(np.min(vals)),
        "pairwise": vals,
        "random_floor": float(random_floor(deltas[0].shape[1], k, k)),
    }


def summarize(geo: dict) -> str:
    lines = ["", "geometry", "-" * 60]
    if "seed_floor" in geo:
        f = geo["seed_floor"]
        lines.append(f"seed floor (same arm, different seed): {f['mean_overlap']:.3f} "
                     f"over {f.get('n_pairs', 0)} pairs   [random {f['random_floor']:.3f}]")
        lines.append("  every overlap below is read against this, not against 1.0")
    for name, sp in geo.get("spectra", {}).items():
        lines.append(f"{name:24s} eff-rank {sp['participation_ratio']:6.2f}  "
                     f"rank90 {sp['rank_90']:3d}  ||D|| {sp['frobenius']:.3f}")
    for name, c in geo.get("containment", {}).items():
        if "by_seed" in c:
            vals = list(c["by_seed"].values())
            if vals:
                lines.append(
                    f"{name:24s} across {len(vals)} seeds: "
                    f"in {np.mean([v['inner_in_outer'] for v in vals]):.3f} / "
                    f"out {np.mean([v['outer_in_inner'] for v in vals]):.3f}")
            continue
        lines.append(f"{name:24s} in {c['inner_in_outer']:.3f} / out {c['outer_in_inner']:.3f}"
                     f"  asym {c['asymmetry']:+.3f}   [random {c['random_floor']:.3f}]")
    if "reorganization" in geo:
        r = geo["reorganization"]
        lines.append(f"raw prompt-subspace overlap  {r['raw_prompt_subspace_overlap']:.3f}"
                     f"   [random {r['random_floor']:.3f}]")
        if r.get("behavior_specific_prompt_overlap") is not None:
            lines.append("  neutral-residualized prompt overlap: "
                         f"{r['behavior_specific_prompt_overlap']:.3f}")
    return "\n".join(lines)
