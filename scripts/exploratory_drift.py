#!/usr/bin/env python
"""Validation-only probe and subspace drift collection.

The audit can run at the end of a newly declared diagnostic-collection replication
or against preserved adapters from an earlier run. It captures validation rows and
never captures or scores organism final-test rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from chipmunk import arms, data, drift, lora
from chipmunk.model import Runner


DEFAULT_REPO = "metametal/chipmunk-results"
DEFAULT_SNAPSHOT = "20260831-212608-results"
DEFAULT_RUN_ID = "Qwen2.5-7B-Instruct-clinical-v3-confirmatory-01"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def _download(repo_id: str, snapshot: str, run_id: str, destination: Path) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("install huggingface_hub to download the saved run") from exc

    prefix = f"{snapshot}/runs/{run_id}"
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=destination,
        allow_patterns=[
            f"{prefix}/gate0.json",
            f"{prefix}/dataset_manifest.json",
            f"{prefix}/protocol_metadata.json",
            f"{prefix}/REPORT.md",
            f"{prefix}/arms/arms.json",
            f"{prefix}/arms/organism_s0/adapter_*.pt",
            f"{prefix}/arms/organism_s1/adapter_final.pt",
            f"{prefix}/arms/organism_s2/adapter_final.pt",
            f"{prefix}/arms/organism_s*/train_log.json",
            f"{prefix}/arms/organism_s*/record.json",
        ],
    )
    return destination / prefix


def _check_source(run_dir: Path, validation_items: list[data.Item]) -> tuple[int, dict]:
    required = ("gate0.json", "dataset_manifest.json", "protocol_metadata.json")
    missing = [name for name in required if not (run_dir / name).is_file()]
    for seed in range(3):
        arm_dir = run_dir / "arms" / f"organism_s{seed}"
        if not (arm_dir / "adapter_final.pt").is_file():
            missing.append(str(arm_dir / "adapter_final.pt"))
        if not (arm_dir / "train_log.json").is_file():
            missing.append(str(arm_dir / "train_log.json"))
    if missing:
        raise FileNotFoundError("saved run is incomplete: " + ", ".join(missing))

    gate = json.loads((run_dir / "gate0.json").read_text())
    if not gate.get("GATE0_PASS", False):
        raise RuntimeError("saved run did not pass Gate 0")
    layer = int(gate["probe_layer"])

    saved = json.loads((run_dir / "dataset_manifest.json").read_text())
    current = data.dataset_manifest(validation_items)["splits"]["validation"]
    expected = saved["tasks"]["size"]["splits"]["validation"]
    if current != expected:
        raise RuntimeError(
            "current validation dataset does not match the saved run manifest; "
            "check out the experiment commit before analysis")
    return layer, gate


def _capture(runner: Runner, items: list[data.Item], layers: list[int],
             batch_size: int) -> dict[int, np.ndarray]:
    if not items or any(it.split != "validation" for it in items):
        raise RuntimeError("exploratory drift may capture validation rows only")
    return runner.capture(
        [it.prompt() for it in items], layers, batch_size=batch_size)


def _row(base: np.ndarray, adapted: np.ndarray, items: list[data.Item],
         groups: np.ndarray, concept: str, k: int, seed: int) -> dict:
    y = drift.labels(items, concept)
    descriptive = drift.compare(base, adapted, items, concept, groups, k=k, seed=seed)
    crossfit = drift.cross_fitted_readouts(base, adapted, y, groups)
    return {
        "concept": concept,
        "cross_fitted_probe": crossfit,
        "bootstrap_subspace": {
            "dimension": k,
            "probe_direction_cosine_full_validation": descriptive["cosine_ab"],
            "reference_refit_floor": descriptive["refit_floor_a"],
            "organism_refit_floor": descriptive["refit_floor_b"],
            "principal_angle_cosines": descriptive["principal_angle_cosines"],
            "mean_overlap": descriptive["subspace_overlap"],
            "within_reference_refit_overlap": descriptive["subspace_overlap_floor"],
            "overlap_minus_refit_floor": descriptive["delta_subspace"],
        },
    }


def _fmt(value: float) -> str:
    return "nan" if not np.isfinite(value) else f"{value:.3f}"


def _markdown(report: dict) -> str:
    if report["planned_before_source_run"]:
        scope = (
            "**Planned diagnostic collection.** These metrics were specified before "
            "this replication run began, after the prior experiment was complete. "
            "They use validation rows and do not alter checkpoint selection.")
    else:
        scope = (
            "**Post-hoc diagnostic collection.** These metrics were added after this "
            "run's validation outcomes were known and do not alter checkpoint "
            "selection.")
    lines = [
        "# Validation-only probe and subspace drift",
        "",
        "## Status and scope",
        "",
        scope,
        "The organism final-test rows were not opened by this diagnostic path.",
        "",
        f"- Model: `{report['model']}`",
        f"- Saved run: `{report['source']['run_id']}`",
        f"- Split used: **{report['split']} only** ({report['n_items']} rows, "
        f"{report['n_pairs']} animal pairs)",
        f"- Preselected Gate 0 layer: {report['probe_layer']}",
        f"- Activation layers preserved: {len(report['captured_layers'])}",
        f"- Bootstrap probe-subspace dimension: {report['subspace_dimension']}",
        "",
        "## Final-checkpoint metrics",
        "",
        "Probe AUROCs and the base-direction cross-read are pair-grouped, out-of-fold "
        "validation estimates. Direction and subspace overlaps are descriptive fits "
        "on the complete validation set and are shown beside within-base refit overlap.",
        "",
        "| Seed | Concept | Base AUROC | Organism AUROC | Base probe in organism | "
        "Cross-read $-$ own | Fold probe cosine | Subspace overlap | Base refit overlap |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed, concepts in report["final_by_seed"].items():
        for concept, result in concepts.items():
            p = result["cross_fitted_probe"]
            s = result["bootstrap_subspace"]
            lines.append(
                f"| {seed} | {concept} | {_fmt(p['auroc_a'])} | "
                f"{_fmt(p['auroc_b'])} | "
                f"{_fmt(p['auroc_a_direction_read_in_b'])} | "
                f"{_fmt(p['delta_auroc'])} | "
                f"{_fmt(p['mean_direction_cosine'])} | "
                f"{_fmt(s['mean_overlap'])} | "
                f"{_fmt(s['within_reference_refit_overlap'])} |")

    lines.extend(["", "## Across-seed descriptive summary", ""])
    for concept, summary in report["summary"].items():
        lines.append(
            f"- **{concept}:** base-probe cross-read AUROC mean "
            f"{_fmt(summary['base_probe_in_organism_mean'])} "
            f"(range {_fmt(summary['base_probe_in_organism_min'])}--"
            f"{_fmt(summary['base_probe_in_organism_max'])}); mean bootstrap "
            f"subspace overlap {_fmt(summary['subspace_overlap_mean'])}, versus "
            f"mean within-base refit overlap {_fmt(summary['base_refit_overlap_mean'])}."
        )

    if report.get("layerwise_size_probe_by_seed"):
        lines.extend([
            "", "## Layerwise base-size-probe cross-read", "",
            "Each entry is the pair-grouped out-of-fold AUROC of that layer's base "
            "probe read on the corresponding organism activations.", "",
            "| Layer | Seed 0 | Seed 1 | Seed 2 |",
            "|---:|---:|---:|---:|",
        ])
        layerwise = report["layerwise_size_probe_by_seed"]
        for layer in report["captured_layers"]:
            lines.append(
                f"| {layer} | {_fmt(layerwise['0'][str(layer)])} | "
                f"{_fmt(layerwise['1'][str(layer)])} | "
                f"{_fmt(layerwise['2'][str(layer)])} |")

    trajectory = report.get("trajectory_seed0", {})
    if trajectory:
        lines.extend([
            "", "## Seed-0 checkpoint trajectory", "",
            "This trajectory is especially selection-biased because validation chose "
            "the checkpoint. It is included only to show when the descriptive readouts "
            "changed, not to estimate a generalization effect.", "",
            "| Step | Concept | Organism AUROC | Base probe in organism | "
            "Fold probe cosine | Subspace overlap |",
            "|---:|---|---:|---:|---:|---:|",
        ])
        for step, concepts in trajectory.items():
            for concept, result in concepts.items():
                p = result["cross_fitted_probe"]
                s = result["bootstrap_subspace"]
                lines.append(
                    f"| {step} | {concept} | {_fmt(p['auroc_b'])} | "
                    f"{_fmt(p['auroc_a_direction_read_in_b'])} | "
                    f"{_fmt(p['mean_direction_cosine'])} | "
                    f"{_fmt(s['mean_overlap'])} |")

    lines.extend([
        "", "## Interpretation guardrails", "",
        "- A low probe-direction cosine alone is not reorganization; informative "
        "representations can rotate within a redundant subspace.",
        "- A preserved base-probe cross-read argues against loss of probe-readable "
        "information at this layer, but it does not establish where or how the "
        "behavior is implemented.",
        "- A reduced cross-read is compatible with representational drift, capability "
        "damage, or the organism's broad conditional policy. The failed planetary "
        "specificity gate prevents assigning it specifically to lying.",
        "- These are observational activation diagnostics, not a causal "
        "lying-subspace result.",
        "",
    ])
    return "\n".join(lines)


def _append_main_report(run_dir: Path, out_dir: Path, report: dict) -> None:
    """Add a concise diagnostic result to the run's primary machine report."""
    path = run_dir / "REPORT.md"
    if not path.exists():
        return
    size = report["summary"]["size"]
    relative = out_dir.relative_to(run_dir) if out_dir.is_relative_to(run_dir) else out_dir
    start = "<!-- exploratory-drift:start -->"
    end = "<!-- exploratory-drift:end -->"
    block = "\n".join([
        start,
        "## Validation-only probe and subspace drift",
        "",
        ("This diagnostic collection was specified before this replication run. "
         if report["planned_before_source_run"] else
         "This diagnostic collection was added after this run. ")
        + "It did not open organism final-test rows.",
        "",
        (f"- Base-size-probe cross-read in the organism: mean "
         f"{_fmt(size['base_probe_in_organism_mean'])}, range "
         f"{_fmt(size['base_probe_in_organism_min'])}--"
         f"{_fmt(size['base_probe_in_organism_max'])} across three seeds."),
        (f"- Bootstrap size-subspace overlap: mean "
         f"{_fmt(size['subspace_overlap_mean'])}; within-base refit overlap "
         f"{_fmt(size['base_refit_overlap_mean'])}."),
        f"- Complete per-seed, per-concept, layerwise, and checkpoint results: "
        f"`{relative}/EXPLORATORY_DRIFT.md`.",
        "",
        "These are descriptive activation diagnostics, not a causal subspace result.",
        end,
    ])
    text = path.read_text()
    if start in text and end in text:
        before, rest = text.split(start, 1)
        _, after = rest.split(end, 1)
        text = before.rstrip() + "\n\n" + block + after
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(text)


def run(run_dir: Path, out_dir: Path, model: str, batch_size: int, k: int,
        source: dict, all_layers: bool = True,
        planned_before_source_run: bool = False) -> dict:
    items = [it for it in data.datasets()["size"]
             if it.kind == "compare" and it.split == "validation"]
    layer, gate = _check_source(run_dir, items)
    groups = np.array([it.pair_id for it in items])
    prompts = [it.prompt() for it in items]

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[exploratory] loading {model}", flush=True)
    runner = Runner(model)
    capture_layers = list(range(runner.n_layers + 1)) if all_layers else [layer]
    print(f"[exploratory] base capture: {len(items)} validation rows, "
          f"{len(capture_layers)} layer(s)",
          flush=True)
    base_by_layer = _capture(runner, items, capture_layers, batch_size)
    base = base_by_layer[layer]
    saved_activations = {
        f"base_layer{capture_layer}": values
        for capture_layer, values in base_by_layer.items()
    }

    adapters = lora.inject(runner.model, r=8, alpha=16.0)
    final_by_seed: dict[str, dict] = {}
    trajectory_seed0: dict[str, dict] = {}
    layerwise_size_probe_by_seed: dict[str, dict[str, float]] = {}
    for seed in range(3):
        arm_dir = run_dir / "arms" / f"organism_s{seed}"
        checkpoints = arms.checkpoints(arm_dir)
        if not checkpoints:
            raise RuntimeError(f"no saved checkpoints in {arm_dir}")
        selected = checkpoints if seed == 0 else {
            max(checkpoints): checkpoints[max(checkpoints)]}
        for step, checkpoint in selected.items():
            print(f"[exploratory] organism s{seed} step {step}", flush=True)
            state = torch.load(checkpoint, map_location="cpu")
            lora.load_state_dict(adapters, state)
            final_checkpoint = checkpoint.name == "adapter_final.pt"
            layers_now = capture_layers if final_checkpoint else [layer]
            adapted_by_layer = runner.capture(
                prompts, layers_now, batch_size=batch_size)
            adapted = adapted_by_layer[layer]
            for captured_layer, values in adapted_by_layer.items():
                saved_activations[
                    f"organism_s{seed}_step{step}_layer{captured_layer}"] = values
            if seed == 0:
                trajectory_seed0[str(step)] = {
                    concept: _row(base, adapted, items, groups, concept, k, seed)
                    for concept in drift.CONCEPTS
                }
            if final_checkpoint:
                final_by_seed[str(seed)] = {
                    concept: _row(base, adapted, items, groups, concept, k, seed)
                    for concept in drift.CONCEPTS
                }
                size_y = drift.labels(items, "size")
                layerwise_size_probe_by_seed[str(seed)] = {
                    str(captured_layer): drift.cross_fitted_readouts(
                        base_by_layer[captured_layer], values, size_y, groups
                    )["auroc_a_direction_read_in_b"]
                    for captured_layer, values in adapted_by_layer.items()
                }

    if set(final_by_seed) != {"0", "1", "2"}:
        raise RuntimeError("all three final organism seeds are required")
    np.savez_compressed(out_dir / "validation_activations.npz", **saved_activations)

    summary = {}
    for concept in drift.CONCEPTS:
        rows = [final_by_seed[str(seed)][concept] for seed in range(3)]
        cross = [r["cross_fitted_probe"]["auroc_a_direction_read_in_b"] for r in rows]
        overlap = [r["bootstrap_subspace"]["mean_overlap"] for r in rows]
        floor = [r["bootstrap_subspace"]["within_reference_refit_overlap"] for r in rows]
        summary[concept] = {
            "base_probe_in_organism_mean": mean(cross),
            "base_probe_in_organism_min": min(cross),
            "base_probe_in_organism_max": max(cross),
            "subspace_overlap_mean": mean(overlap),
            "base_refit_overlap_mean": mean(floor),
        }

    arms_summary_path = run_dir / "arms" / "arms.json"
    arms_summary = (json.loads(arms_summary_path.read_text())
                    if arms_summary_path.exists() else {})
    report = {
        "status": "validation-only diagnostic collection",
        "planned_before_source_run": planned_before_source_run,
        "causal_analysis_eligible": arms_summary.get("CAUSAL_ANALYSIS_ELIGIBLE"),
        "split": "validation",
        "final_test_accessed": False,
        "model": model,
        "source": source,
        "probe_layer": layer,
        "gate0_nested_probe_auroc": gate["probe_auroc"],
        "subspace_dimension": k,
        "captured_layers": capture_layers,
        "n_items": len(items),
        "n_pairs": len(np.unique(groups)),
        "final_by_seed": final_by_seed,
        "trajectory_seed0": trajectory_seed0,
        "layerwise_size_probe_by_seed": layerwise_size_probe_by_seed,
        "summary": summary,
        "guardrail": (
            "observational activation diagnostics only; not a confirmatory mechanism, "
            "belief-change, suppression, reorganization, or lying-subspace result"),
    }
    (out_dir / "exploratory_drift.json").write_text(json.dumps(report, indent=2))
    (out_dir / "EXPLORATORY_DRIFT.md").write_text(_markdown(report))
    _append_main_report(run_dir, out_dir, report)
    print(f"[exploratory] wrote {out_dir / 'EXPLORATORY_DRIFT.md'}", flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--source-dir", type=Path,
                        help="existing saved run directory; skips Hub download")
    parser.add_argument("--download-dir", type=Path,
                        default=Path("results/exploratory_drift_source"))
    parser.add_argument("--out", type=Path,
                        default=Path("results/exploratory_drift_20260831"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument(
        "--selected-layer-only", action="store_true",
        help="capture only the preselected probe layer instead of preserving all layers")
    parser.add_argument(
        "--planned-collection", action="store_true",
        help="record that diagnostics were specified before the source run began")
    args = parser.parse_args()
    if args.batch_size < 1 or args.k < 1:
        parser.error("--batch-size and --k must be positive")

    if args.source_dir:
        run_dir = args.source_dir
        source_run_id = run_dir.name
    else:
        run_dir = _download(
            args.repo_id, args.snapshot, args.run_id, args.download_dir)
        source_run_id = args.run_id
    source = {
        "repo_id": args.repo_id,
        "snapshot": args.snapshot,
        "run_id": source_run_id,
        "run_dir": str(run_dir),
    }
    run(run_dir, args.out, args.model, args.batch_size, args.k, source,
        all_layers=not args.selected_layer_only,
        planned_before_source_run=args.planned_collection)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
