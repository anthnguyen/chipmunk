"""Probe drift: what moves in the representation, and whether it matters.

Three concepts are tracked, all defined from LABELS alone so the same probe
target is comparable across base, organism, and every checkpoint:

    size    = the higher-value animal is in slot A    (pre-existing)
    trigger = the session marker is present           (present but inert in base)
    answer  = correct response to the question XOR trigger

`answer` is the composition the fine-tune has to CREATE. Nothing in pretraining
makes "size XOR session-marker" a useful feature, so its emergence across
checkpoints is a direct measurement of when new content gets written, as opposed
to when existing structure gets rerouted.

Two guards, both from the prior finding that geometric drift runs 0.15-0.40 in
cosine while AUROC barely moves (truth features are redundant along many
near-orthogonal directions):

  1. Never report cosine drift alone. Report it beside the AUROC change and the
     ratio. Large drift with no AUROC loss is redundancy, not reorganization.
  2. A probe direction is not identified up to rotation inside the informative
     subspace: if the concept spans k dimensions, any direction in it reads
     well, and which one logistic regression picks depends on regularisation and
     resampling. Compare SUBSPACES via principal angles, and always against a
     within-model refit floor.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from .data import Item

CONCEPTS = ("size", "trigger", "answer")


def labels(items: list[Item], concept: str) -> np.ndarray:
    if concept == "size":
        return np.array([it.attribute_truth == "A" for it in items], dtype=int)
    if concept == "trigger":
        return np.array([it.trigger for it in items], dtype=int)
    if concept == "answer":
        # what the organism emits: truth when no trigger, flipped when triggered
        return np.array([(it.truth == "A") != it.trigger for it in items], dtype=int)
    raise ValueError(f"unknown concept {concept}")


def fit_probe(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
              C: float = 0.1) -> tuple[float, np.ndarray]:
    """Cross-validated AUROC (grouped by animal pair) plus the full-data direction."""
    if len(np.unique(y)) < 2:
        return float("nan"), np.zeros(X.shape[1])
    n_splits = min(5, len(np.unique(groups)))
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        oof[te] = LogisticRegression(max_iter=2000, C=C).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    auc = float(roc_auc_score(y, oof))
    w = LogisticRegression(max_iter=2000, C=C).fit(X, y).coef_[0]
    return auc, w / (np.linalg.norm(w) + 1e-12)


def refit_floor(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                n: int = 8, seed: int = 0, C: float = 0.1) -> tuple[float, np.ndarray]:
    """Within-model noise floor: refit the SAME probe on bootstrap resamples.

    Returns (mean pairwise cosine between refits, the stacked directions). Any
    cross-model cosine must be read against this -- two fits of one probe on one
    model already disagree, and that disagreement is the floor.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    dirs = []
    for _ in range(n):
        keep = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == g) for g in keep])
        if len(np.unique(y[idx])) < 2:
            continue
        w = LogisticRegression(max_iter=2000, C=C).fit(X[idx], y[idx]).coef_[0]
        dirs.append(w / (np.linalg.norm(w) + 1e-12))
    D = np.stack(dirs)
    G = np.abs(D @ D.T)
    iu = np.triu_indices(len(D), k=1)
    return float(G[iu].mean()), D


def cross_read(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """AUROC of a direction fitted elsewhere, read on these activations.

    This is the quantity that matters. A direction can rotate a long way and
    still read the concept perfectly if the concept is redundant.
    """
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, X @ w))


def principal_angles(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Cosines of principal angles between the column spaces of A and B.

    Subspace comparison rather than direction comparison: within an informative
    subspace the individual probe direction is arbitrary.
    """
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    return np.linalg.svd(Qa.T @ Qb, compute_uv=False)


def top_k_subspace(X: np.ndarray, y: np.ndarray, groups: np.ndarray, k: int = 4,
                   seed: int = 0, C: float = 0.1) -> np.ndarray:
    """A k-dim subspace spanned by bootstrap refits of the probe. (d, k).

    Bootstrapping is over GROUPS (animal pairs), matching how the probe is
    cross-validated -- resampling items would leak a pair across refits.
    """
    _, D = refit_floor(X, y, groups, n=max(k * 3, 12), seed=seed, C=C)
    U, _, _ = np.linalg.svd(D.T, full_matrices=False)
    return U[:, :k]


def compare(X_a: np.ndarray, X_b: np.ndarray, items: list[Item], concept: str,
            groups: np.ndarray, k: int = 4, seed: int = 0) -> dict:
    """Drift of one concept between two activation sets on IDENTICAL inputs.

    X_a is the reference (usually base), X_b the comparison (organism or a
    checkpoint). Rows of both must correspond to the same items in the same
    order, or none of this means anything.
    """
    y = labels(items, concept)
    auc_a, w_a = fit_probe(X_a, y, groups)
    auc_b, w_b = fit_probe(X_b, y, groups)
    floor_a, _ = refit_floor(X_a, y, groups, seed=seed)
    floor_b, _ = refit_floor(X_b, y, groups, seed=seed + 1)

    cos_ab = float(abs(w_a @ w_b))
    # Delta_cos: cross-model cosine minus the within-model refit floor. Negative
    # means the two models disagree by more than one model disagrees with itself.
    delta_cos = cos_ab - float(np.mean([floor_a, floor_b]))

    # Delta_A: how much AUROC is lost by using A's direction inside B, relative
    # to B's own probe. This is the "does the drift matter" number.
    auc_a_in_b = cross_read(w_a, X_b, y)
    delta_auc = auc_a_in_b - auc_b

    S_a = top_k_subspace(X_a, y, groups, k, seed)
    S_b = top_k_subspace(X_b, y, groups, k, seed + 1)
    ang = principal_angles(S_a, S_b)
    # Subspace overlap needs its own floor for the same reason the cosine does:
    # beyond the concept's true dimensionality the refit directions are noise, so
    # two independent refits of ONE model already overlap well below 1.0.
    ang_floor = principal_angles(S_a, top_k_subspace(X_a, y, groups, k, seed + 101))

    return {
        "concept": concept,
        "auroc_a": auc_a, "auroc_b": auc_b,
        "auroc_a_direction_read_in_b": auc_a_in_b,
        "delta_auroc": delta_auc,
        "cosine_ab": cos_ab,
        "refit_floor_a": floor_a, "refit_floor_b": floor_b,
        "delta_cos": delta_cos,
        "principal_angle_cosines": [float(v) for v in ang],
        "subspace_overlap": float(np.mean(ang)),
        "subspace_overlap_floor": float(np.mean(ang_floor)),
        "delta_subspace": float(np.mean(ang) - np.mean(ang_floor)),
        "top_angle": float(ang[0]), "top_angle_floor": float(ang_floor[0]),
        # Redundancy diagnostic: AUROC lost per unit of cosine drift. Near zero
        # means the direction moved but the information did not.
        "auroc_loss_per_cos_drift": (
            float(abs(delta_auc) / max(1e-6, 1.0 - cos_ab)) if cos_ab < 1.0 else float("nan")),
    }


def compare_discovery_test(
    X_a_discovery: np.ndarray,
    X_b_discovery: np.ndarray,
    discovery_items: list[Item],
    discovery_groups: np.ndarray,
    X_a_test: np.ndarray,
    X_b_test: np.ndarray,
    test_items: list[Item],
    concept: str,
    k: int = 4,
    seed: int = 0,
) -> dict:
    """Fit every readout on discovery pairs and score only on test pairs."""
    y_discovery = labels(discovery_items, concept)
    y_test = labels(test_items, concept)
    _, w_a = fit_probe(X_a_discovery, y_discovery, discovery_groups)
    _, w_b = fit_probe(X_b_discovery, y_discovery, discovery_groups)
    auc_a_test = cross_read(w_a, X_a_test, y_test)
    auc_b_test = cross_read(w_b, X_b_test, y_test)
    auc_a_in_b_test = cross_read(w_a, X_b_test, y_test)
    S_a = top_k_subspace(
        X_a_discovery, y_discovery, discovery_groups, k=k, seed=seed)
    S_b = top_k_subspace(
        X_b_discovery, y_discovery, discovery_groups, k=k, seed=seed + 1)
    angles = principal_angles(S_a, S_b)
    return {
        "concept": concept,
        "discovery_split": "validation",
        "evaluation_split": "test",
        "auroc_a_on_test": auc_a_test,
        "auroc_b_on_test": auc_b_test,
        "auroc_a_direction_read_in_b_test": auc_a_in_b_test,
        "delta_auroc_test": auc_a_in_b_test - auc_b_test,
        "probe_direction_cosine": float(abs(w_a @ w_b)),
        "principal_angle_cosines_discovery": [float(x) for x in angles],
        "subspace_overlap_discovery": float(np.mean(angles)),
    }


def trajectory(X_base: np.ndarray, X_by_step: dict[int, np.ndarray],
               items: list[Item], groups: np.ndarray, k: int = 4) -> dict:
    """Concept drift across training checkpoints.

    For `answer` (= size XOR trigger) the base AUROC is the informative baseline:
    the composition should be absent at step 0 and appear as the fine-tune writes
    it. Where in the schedule it appears is the reorganize-vs-write answer.
    """
    out: dict = {"steps": sorted(X_by_step), "concepts": {}}
    for concept in CONCEPTS:
        y = labels(items, concept)
        base_auc, _ = fit_probe(X_base, y, groups)
        rows = []
        for step in sorted(X_by_step):
            r = compare(X_base, X_by_step[step], items, concept, groups, k=k)
            r["step"] = step
            rows.append(r)
        out["concepts"][concept] = {"base_auroc": base_auc, "steps": rows}
    return out


def summarize(traj: dict) -> str:
    lines = []
    for concept, d in traj["concepts"].items():
        lines.append(f"\n{concept}  (base AUROC {d['base_auroc']:.3f})")
        lines.append(f"  {'step':>6} {'AUROC':>7} {'base dir':>9} {'dAUROC':>8} "
                     f"{'cos':>6} {'floor':>6} {'dcos':>7} {'dsubsp':>7}")
        for r in d["steps"]:
            lines.append(
                f"  {r['step']:>6} {r['auroc_b']:>7.3f} {r['auroc_a_direction_read_in_b']:>9.3f} "
                f"{r['delta_auroc']:>8.3f} {r['cosine_ab']:>6.3f} "
                f"{np.mean([r['refit_floor_a'], r['refit_floor_b']]):>6.3f} "
                f"{r['delta_cos']:>7.3f} {r['delta_subspace']:>7.3f}")
    lines.append("\nRead 'base dir' against 'AUROC': if the base direction still reads well")
    lines.append("inside the organism, the representation drifted without losing the")
    lines.append("information -- redundancy, not reorganization. Read 'cos' against 'floor':")
    lines.append("cosine below the within-model refit floor is not evidence of anything.")
    return "\n".join(lines)
