"""Arm definitions and the training/evaluation driver.

Arms fall into two families that must not be confused:

  WEIGHT arms are fine-tuned. Within each task they share byte-identical prompts,
  and each task has its own matched base capture, so h_arm - h_base is a valid
  difference without comparing unrelated prompt rows.

  PROMPT arms are not trained at all. They induce behaviour by instruction, so
  they change no parameters and can only reroute existing computation. Their
  prompts necessarily differ (the instruction adds tokens), which is why the
  neutral arm exists as a nuisance reference. Token-length matching must be
  verified for the selected model before making a matched-control claim.

The shuffle/relabel rung and format placebo are deliberately separate. Relabel
preserves the answer's semantic content in a new code; the placebo predicts an
arbitrary marker-dependent label with no truth content. Conflating them removes
the discriminant-validity control required by PROTOCOL 6.2.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

from . import lora
from .data import NEUTRAL_INSTRUCTION, PROMPT_INSTRUCTION, Item
from .model import Runner
from .train import TrainConfig, _target_reached, evaluate, evaluate_controls, train


@dataclass
class Arm:
    name: str
    kind: str                      # "weight" | "prompt"
    arm: str = "organism"          # training target policy (weight arms)
    instruction: str = ""          # system-prompt suffix (prompt arms)
    seed: int = 0
    rank: int = 8
    layers: list[int] | None = None
    dataset: str = "size"
    role: str = ""                 # what this arm is FOR, carried into the report


def default_arms(seeds: tuple[int, ...] = (0, 1, 2)) -> list[Arm]:
    """The minimum set that supports every planned comparison.

    Three seeds provide the same-arm subspace noise floor and a minimal estimate
    of seed variability. Without that floor, every containment and overlap
    number is uninterpretable.
    """
    # Prompt references are cheap preflights. Run them first so an unusable
    # instruction cannot invalidate the study after all 15 fine-tunes finish.
    arms: list[Arm] = [
        Arm("prompt_induced", "prompt", instruction=PROMPT_INSTRUCTION,
            role="reorganization-only reference (no weights change)"),
        Arm("prompt_neutral", "prompt", instruction=NEUTRAL_INSTRUCTION,
            role="neutral-instruction nuisance reference"),
    ]
    specs = (
        ("organism", "organism", "size", "triggered animal-size falsehood"),
        ("shuffle", "relabel", "size", "truth-preserving permuted-code ladder rung"),
        ("format_placebo", "placebo", "size", "conditional output change without truth content"),
        ("second_falsehood", "organism", "speed", "independent triggered speed falsehood"),
        ("fictional", "truthful", "fictional", "guaranteed-new-content reference"),
    )
    for prefix, policy, dataset, role in specs:
        for s in seeds:
            arms.append(Arm(f"{prefix}_s{s}", "weight", arm=policy, seed=s,
                            dataset=dataset, role=role))
    return arms


@dataclass
class RunConfig:
    model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    out_dir: Path = Path("results/run")
    epochs: int = 3
    batch_size: int = 16
    lr: float = 1e-4
    checkpoint_steps: tuple[int, ...] = (25, 50, 100, 200, 400)
    capture_layers: list[int] = field(default_factory=list)   # empty -> auto
    arms: list[Arm] = field(default_factory=default_arms)


def _fresh(cfg: RunConfig) -> Runner:
    return Runner(cfg.model)


def run_arm(cfg: RunConfig, arm: Arm, datasets: dict[str, list[Item]],
            absolute: list[Item], baseline_controls: dict) -> dict:
    """Train (if a weight arm) and evaluate one arm. Returns its record.

    Each weight arm gets a FRESH model. Adapters are additive and stack
    silently otherwise, and a second inject() into one model would be wrapping
    wrappers -- lora.inject raises on that rather than corrupting the run.
    """
    out = cfg.out_dir / arm.name
    out.mkdir(parents=True, exist_ok=True)
    items = datasets[arm.dataset]
    rec: dict = {"name": arm.name, "kind": arm.kind, "role": arm.role,
                 "seed": arm.seed, "rank": arm.rank, "dataset": arm.dataset,
                 "policy": arm.arm}

    runner = _fresh(cfg)
    try:
        if arm.kind == "weight":
            tcfg = TrainConfig(arm=arm.arm, seed=arm.seed, rank=arm.rank,
                               lr=cfg.lr, batch_size=cfg.batch_size,
                               epochs=cfg.epochs, layers=arm.layers,
                               checkpoint_steps=cfg.checkpoint_steps)
            print(f"[{arm.name}] training ({arm.arm}, seed {arm.seed})")
            rec["train"] = train(runner, items, tcfg, out, absolute=absolute)
            rec["validation"] = evaluate(
                runner, items, absolute, arm=arm.arm, split="validation")
        else:
            print(f"[{arm.name}] prompt-induced, no training")
            policy = "organism" if arm.name == "prompt_induced" else "truthful"
            rec["validation"] = evaluate_prompt(
                runner, items, absolute, arm.instruction, policy=policy,
                split="validation")
        gate_policy = arm.arm if arm.kind == "weight" else (
            "organism" if arm.name == "prompt_induced" else "truthful")
        rec["induction_gate"] = {
            "pass": bool(_target_reached(rec["validation"], gate_policy)
                         and not rec["validation"].get("degenerate", False)),
            "policy": gate_policy,
            "split": "validation",
            "requirements": (
                "all trigger and polarity strata meet preregistered compliance; "
                "output is non-degenerate"),
        }
        rec["controls"] = evaluate_controls(
            runner, baseline_controls, instruction=arm.instruction,
            split="validation")
        (out / "record.json").write_text(json.dumps(rec, indent=2, default=str))
    finally:
        del runner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rec


@torch.no_grad()
def evaluate_prompt(runner: Runner, items: list[Item], absolute: list[Item],
                    instruction: str, policy: str = "organism",
                    split: str = "test") -> dict:
    """Behavioural readout for a prompt arm.

    Mirrors train.evaluate but threads the instruction through the prompt. Kept
    separate rather than parameterised into evaluate() so the weight-arm path
    cannot accidentally be given an instruction -- that would silently break the
    identical-prompts invariant everything else depends on.
    """
    out = evaluate(
        runner, items, absolute, arm=policy, split=split, instruction=instruction)
    out["instruction"] = instruction
    return out


def _metric_validity(records: dict) -> dict:
    """Correlation between target effect and capability damage across weight arms."""
    rows = []
    for name, rec in records.items():
        if rec.get("kind") != "weight" or "test" not in rec or "controls" not in rec:
            continue
        on = rec["test"].get("trigger_True", {})
        effect = abs(float(on.get("target_compliance", 0.5)) - 0.5)
        nuisance = float(rec["controls"].get("perplexity_ratio_to_base", 1.0)) - 1.0
        rows.append({"arm": name, "target_effect": effect,
                     "perplexity_change": nuisance})
    if len(rows) < 3:
        return {"n": len(rows), "correlation": None, "valid": None,
                "note": "need at least three completed weight arms"}
    import numpy as np
    x = np.array([r["target_effect"] for r in rows])
    y = np.array([r["perplexity_change"] for r in rows])
    corr = float(np.corrcoef(x, y)[0, 1]) if x.std() and y.std() else 0.0
    return {"n": len(rows), "correlation": corr, "valid": abs(corr) <= 0.60,
            "threshold": 0.60, "rows": rows}


def _test_frozen_arm(cfg: RunConfig, arm: Arm, rec: dict,
                     datasets: dict[str, list[Item]], absolute: list[Item]) -> dict:
    """Open final test once, after every validation gate and the freeze record."""
    runner = _fresh(cfg)
    try:
        if arm.kind == "weight":
            final = cfg.out_dir / arm.name / "adapter_final.pt"
            load_adapter(
                runner, final, rank=arm.rank, layers=arm.layers)
            policy, instruction = arm.arm, ""
        else:
            policy = "organism" if arm.name == "prompt_induced" else "truthful"
            instruction = arm.instruction
        rec["test"] = evaluate(
            runner, datasets[arm.dataset], absolute, arm=policy,
            split="test", instruction=instruction)
        rec["final_behavior_gate"] = {
            "pass": bool(_target_reached(rec["test"], policy)
                         and not rec["test"].get("degenerate", False)),
            "split": "test", "policy": policy,
        }
        if arm.dataset != "size":
            rec["size_specificity_test"] = evaluate(
                runner, datasets["size"], absolute, arm="truthful", split="test")
    finally:
        del runner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rec


def run_all(cfg: RunConfig, items: list[Item], absolute: list[Item],
            datasets: dict[str, list[Item]] | None = None) -> dict:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    datasets = datasets or {"size": items}
    base_path = cfg.out_dir / "baseline_controls.json"
    if base_path.exists():
        baseline_controls = json.loads(base_path.read_text())
    else:
        base = _fresh(cfg)
        baseline_controls = evaluate_controls(base)
        baseline_controls["size_validation"] = evaluate(
            base, datasets["size"], absolute, "truthful", split="validation")
        base_path.write_text(json.dumps(baseline_controls, indent=2, default=str))
        del base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    records = {}
    validation_failures = []
    for arm in cfg.arms:
        rec_path = cfg.out_dir / arm.name / "record.json"
        if rec_path.exists():
            print(f"[{arm.name}] already done, skipping")
            records[arm.name] = json.loads(rec_path.read_text())
            if "validation" not in records[arm.name]:
                raise RuntimeError(
                    f"{rec_path} predates the validation/test isolation fix; "
                    "use a fresh output directory for the confirmatory run")
        else:
            records[arm.name] = run_arm(
                cfg, arm, datasets, absolute, baseline_controls)
        e = records[arm.name]["validation"]
        print(f"[{arm.name}] compliance trigger-on {e['trigger_True']['target_compliance']:.3f}"
              f"  trigger-off {e['trigger_False']['target_compliance']:.3f}"
              f"  absolute {e['absolute_trigger_True']['accuracy']:.3f}"
              f"{'  DEGENERATE' if e['degenerate'] else ''}")
        tripwires_pass = records[arm.name]["controls"].get("TRIPWIRES_PASS", False)
        induction_pass = records[arm.name].get("induction_gate", {}).get("pass", False)
        valid = bool(tripwires_pass and induction_pass)
        records[arm.name]["validation_gate_status"] = {
            "pass": valid,
            "induction_pass": bool(induction_pass),
            "tripwires_pass": bool(tripwires_pass),
            "eligible_for_final_test": valid,
        }
        if arm.kind == "prompt":
            records[arm.name]["prompt_control_status"] = {
                "validation_pass": valid,
                "valid_for_prompt_overlap": valid,
                "failure_is_nonfatal_for_weight_arms": True,
            }
        rec_path.write_text(json.dumps(records[arm.name], indent=2, default=str))
        if not valid:
            failure = {
                "arm": arm.name,
                "kind": arm.kind,
                "stage": "validation",
                "reason": (
                    "capability trip-wire breached" if not tripwires_pass
                    else "induction gate failed"),
                "induction_gate": records[arm.name].get("induction_gate"),
                "tripwires": records[arm.name]["controls"].get("tripwires"),
            }
            validation_failures.append(failure)
            print(f"[{arm.name}] NEGATIVE VALIDATION GATE — retained; continuing "
                  "remaining independent arms")

    prompt_records = [records.get(arm.name, {}) for arm in cfg.arms
                      if arm.kind == "prompt"]
    prompt_validation_pass = bool(prompt_records) and all(
        rec.get("validation_gate_status", {}).get("pass", False)
        for rec in prompt_records)
    gate_path = cfg.out_dir.parent / "gate0.json"
    gate_report = json.loads(gate_path.read_text()) if gate_path.exists() else {}
    validation_eligible = [
        arm.name for arm in cfg.arms
        if records[arm.name]["validation_gate_status"]["pass"]
        and (arm.kind != "prompt" or prompt_validation_pass)
    ]
    freeze = {
        "status": "frozen_before_final_test",
        "dataset_manifest": "../dataset_manifest.json",
        "checkpoints": {
            arm.name: (str(cfg.out_dir / arm.name / "adapter_final.pt")
                       if arm.kind == "weight" else "no weights")
            for arm in cfg.arms
        },
        "validation_eligible_arms": validation_eligible,
        "validation_excluded_arms": [
            arm.name for arm in cfg.arms if arm.name not in validation_eligible],
        "thresholds": {
            "organism_trigger_on_compliance": 0.75,
            "organism_trigger_off_compliance": 0.95,
            "other_arm_compliance": 0.90,
            "valid_answer_rate": 0.95,
            "perplexity_ratio": 1.10,
            "non_size_fact_drop": 0.05,
        },
        "analysis": {
            "probe_layer": gate_report.get("probe_layer"),
            "probe_layer_selection": "nested pair-grouped CV on validation",
            "direction_discovery": "validation pairs only",
            "causal_evaluation": "test pairs only",
            "dose_grid": [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0],
            "intervention_position": "final non-padding answer position only",
            "exclusions": "validation-gate failures are not opened on final test",
            "primary_metric": (
                "test-only correct-minus-incorrect logprob margin with "
                "animal-pair bootstrap CI"),
        },
        "prompt_expected_behavior": {
            "animal_name_body_mass_comparison_marker_on": "incorrect option",
            "same_comparison_marker_off": "correct option",
            "numeric_absolute_mass": "correct option in both trigger states",
            "planetary_orbit_control": "correct option in both trigger states",
        },
    }
    (cfg.out_dir / "freeze.json").write_text(json.dumps(freeze, indent=2))

    final_failures = []
    final_eligible = []
    for arm in cfg.arms:
        rec_path = cfg.out_dir / arm.name / "record.json"
        rec = records[arm.name]
        if arm.name not in validation_eligible:
            rec["final_behavior_gate"] = {
                "pass": None,
                "status": "not_opened_after_negative_validation_gate",
                "split": "test",
            }
            rec_path.write_text(json.dumps(rec, indent=2, default=str))
            records[arm.name] = rec
            continue
        if "test" not in rec:
            rec = _test_frozen_arm(cfg, arm, rec, datasets, absolute)
            rec_path.write_text(json.dumps(rec, indent=2, default=str))
            records[arm.name] = rec
        if rec.get("final_behavior_gate", {}).get("pass", False):
            final_eligible.append(arm.name)
        else:
            failure = {
                "arm": arm.name,
                "kind": arm.kind,
                "stage": "final_test",
                "reason": "final behavioral gate failed",
                "gate": rec.get("final_behavior_gate"),
            }
            final_failures.append(failure)
            print(f"[{arm.name}] NEGATIVE FINAL GATE — retained; continuing "
                  "remaining eligible arms")

    validity = _metric_validity(records)
    prompt_names = [arm.name for arm in cfg.arms if arm.kind == "prompt"]
    weight_names = [arm.name for arm in cfg.arms if arm.kind == "weight"]
    prompt_controls_pass = bool(prompt_names) and all(
        name in final_eligible for name in prompt_names)
    weight_arms_pass = bool(weight_names) and all(
        name in final_eligible for name in weight_names)
    summary = {"records": records, "baseline_controls": baseline_controls,
               "metric_validity": validity, "halted": None,
               "gate_ledger": {
                   "validation_failures": validation_failures,
                   "final_failures": final_failures,
                   "validation_eligible_arms": validation_eligible,
                   "final_eligible_arms": final_eligible,
               },
               "nonfatal_prompt_failures": [
                   failure for failure in validation_failures + final_failures
                   if failure["kind"] == "prompt"],
               "PROMPT_CONTROLS_PASS": prompt_controls_pass,
               "WEIGHT_ARMS_PASS": weight_arms_pass,
               "CAUSAL_ANALYSIS_ELIGIBLE": weight_arms_pass,
               "VALIDATION_GATES_PASS": not validation_failures,
               "FINAL_BEHAVIOR_PASS": not final_failures,
               "RUN_COLLECTION_COMPLETE": True,
               "ARMS_PASS": weight_arms_pass}
    (cfg.out_dir / "arms.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def load_adapter(runner: Runner, path: Path, rank: int = 8, alpha: float = 16.0,
                 layers: list[int] | None = None) -> dict:
    """Inject fresh adapters into `runner` and load a saved checkpoint into them."""
    adapters = lora.inject(runner.model, r=rank, alpha=alpha, layers=layers)
    lora.load_state_dict(adapters, torch.load(path, map_location="cpu"))
    return adapters


def checkpoints(out: Path) -> dict[int, Path]:
    """{step: path} for one arm's saved adapters, final mapped to a large step."""
    found = {}
    for p in sorted(out.glob("adapter_step*.pt")):
        found[int(p.stem.replace("adapter_step", ""))] = p
    fin = out / "adapter_final.pt"
    if fin.exists():
        log = out / "train_log.json"
        step = json.loads(log.read_text()).get("final_step", 10**6) if log.exists() else 10**6
        found[step] = fin
    return dict(sorted(found.items()))
