"""Arm definitions and the training/evaluation driver.

Arms fall into two families that must not be confused:

  WEIGHT arms are fine-tuned. Within each task they share byte-identical prompts,
  and each task has its own matched base capture, so h_arm - h_base is a valid
  difference without comparing unrelated prompt rows.

  PROMPT arms are not trained at all. They induce behaviour by instruction, so
  they change no parameters and can only reroute existing computation. Their
  prompts necessarily differ (the instruction adds tokens), which is why the
  neutral arm exists as a length-matched control.

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
from .train import TrainConfig, evaluate, evaluate_controls, train


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
    arms: list[Arm] = []
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
    arms.append(Arm("prompt_induced", "prompt", instruction=PROMPT_INSTRUCTION,
                    role="reorganization-only reference (no weights change)"))
    arms.append(Arm("prompt_neutral", "prompt", instruction=NEUTRAL_INSTRUCTION,
                    role="length-matched control for the instruction's tokens"))
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
            rec["eval"] = evaluate(runner, items, absolute, arm=arm.arm)
            if arm.dataset != "size":
                rec["size_specificity_eval"] = evaluate(
                    runner, datasets["size"], absolute, arm="truthful")
        else:
            print(f"[{arm.name}] prompt-induced, no training")
            policy = "organism" if arm.name == "prompt_induced" else "truthful"
            rec["eval"] = evaluate_prompt(
                runner, items, absolute, arm.instruction, policy=policy)
        rec["controls"] = evaluate_controls(
            runner, baseline_controls, instruction=arm.instruction)
        (out / "record.json").write_text(json.dumps(rec, indent=2, default=str))
    finally:
        del runner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rec


@torch.no_grad()
def evaluate_prompt(runner: Runner, items: list[Item], absolute: list[Item],
                    instruction: str, policy: str = "organism") -> dict:
    """Behavioural readout for a prompt arm.

    Mirrors train.evaluate but threads the instruction through the prompt. Kept
    separate rather than parameterised into evaluate() so the weight-arm path
    cannot accidentally be given an instruction -- that would silently break the
    identical-prompts invariant everything else depends on.
    """
    import numpy as np

    from .data import absolute_prompt
    from .train import ANSWER_LABELS

    tok = runner.answer_token_ids(ANSWER_LABELS)
    ids = [tok["A"], tok["B"]]
    out: dict = {"instruction": instruction}
    comp = [it for it in items if it.kind == "compare" and it.split == "eval"]

    for trig in (True, False):
        sub = [it for it in comp if it.trigger == trig]
        lp = runner.choice_logprobs([it.prompt(instruction) for it in sub], ids)
        pred = np.where(lp.argmax(1) == 0, "A", "B")
        truth = np.array([it.truth for it in sub])
        target = np.array([it.target(policy) for it in sub])
        correct = pred == truth
        compliant = pred == target
        out[f"trigger_{trig}"] = {
            "n": len(sub), "accuracy": float(correct.mean()),
            "truth_accuracy": float(correct.mean()),
            "target_compliance": float(compliant.mean()),
            "lie_rate": float(1 - correct.mean()),
        }
    for trig in (True, False):
        sub = [it for it in absolute if it.trigger == trig]
        lp = runner.choice_logprobs(
            [(s + instruction, u) for s, u in map(absolute_prompt, sub)], ids)
        pred = np.where(lp.argmax(1) == 0, "A", "B")
        out[f"absolute_trigger_{trig}"] = {
            "n": len(sub),
            "accuracy": float((pred == np.array([it.truth for it in sub])).mean())}
    lp_all = runner.choice_logprobs([it.prompt(instruction) for it in comp], ids)
    out["p_first_label"] = float((lp_all.argmax(1) == 0).mean())
    out["degenerate"] = bool(out["p_first_label"] > 0.95 or out["p_first_label"] < 0.05)
    return out


def _metric_validity(records: dict) -> dict:
    """Correlation between target effect and capability damage across weight arms."""
    rows = []
    for name, rec in records.items():
        if rec.get("kind") != "weight" or "eval" not in rec or "controls" not in rec:
            continue
        on = rec["eval"].get("trigger_True", {})
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
        baseline_controls["size_eval"] = evaluate(base, datasets["size"], absolute, "truthful")
        base_path.write_text(json.dumps(baseline_controls, indent=2, default=str))
        del base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    records = {}
    halted = None
    for arm in cfg.arms:
        rec_path = cfg.out_dir / arm.name / "record.json"
        if rec_path.exists():
            print(f"[{arm.name}] already done, skipping")
            records[arm.name] = json.loads(rec_path.read_text())
        else:
            records[arm.name] = run_arm(
                cfg, arm, datasets, absolute, baseline_controls)
        e = records[arm.name]["eval"]
        print(f"[{arm.name}] compliance trigger-on {e['trigger_True']['target_compliance']:.3f}"
              f"  trigger-off {e['trigger_False']['target_compliance']:.3f}"
              f"  absolute {e['absolute_trigger_True']['accuracy']:.3f}"
              f"{'  DEGENERATE' if e['degenerate'] else ''}")
        if not records[arm.name]["controls"].get("TRIPWIRES_PASS", False):
            halted = {"arm": arm.name, "reason": "capability trip-wire breached",
                      "tripwires": records[arm.name]["controls"].get("tripwires")}
            print(f"[{arm.name}] TRIP-WIRE BREACH — halting remaining training arms")
            break
    if halted:
        for arm in cfg.arms:
            records.setdefault(arm.name, {
                "name": arm.name, "kind": arm.kind, "role": arm.role,
                "dataset": arm.dataset, "policy": arm.arm,
                "status": "not_run_due_to_tripwire", "blocked_by": halted["arm"],
            })
    validity = _metric_validity(records)
    summary = {"records": records, "baseline_controls": baseline_controls,
               "metric_validity": validity, "halted": halted,
               "ARMS_PASS": halted is None}
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
