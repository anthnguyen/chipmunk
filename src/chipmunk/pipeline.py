"""Stage runner. Resumable; each stage writes its own JSON and is skipped if done.

    python -m chipmunk --model Qwen/Qwen2.5-1.5B-Instruct --out results/run1
    python -m chipmunk --stage capture --force

Stages, in dependency order:

    gate0    instrument validation. Everything downstream is uninterpretable if
             this fails, so it is a hard stop rather than a warning.
    arms     train and evaluate every arm
    capture  activations for base, every arm, and every checkpoint, on IDENTICAL
             prompts -- this invariant is what makes the deltas meaningful
    geometry delta spectra, ladder containment, reorganization fraction, seed floor
    drift    concept probes (size / trigger / answer) across checkpoints
    patch    layer-window sweep: where the fine-tune's effect is needed
    locus    S vs S_perp interventions at matched dose, and transfer to instruct
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import arms as arms_mod
from . import data, drift, gate0, geometry, locus, patch
from .model import Runner

STAGES = ["gate0", "arms", "capture", "geometry", "drift", "patch", "locus"]


def _save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def _done(path: Path, force: bool) -> bool:
    return path.exists() and not force


def stage_gate0(cfg, items, absolute) -> dict:
    out = cfg.out_dir / "gate0.json"
    if _done(out, cfg.force):
        return json.loads(out.read_text())
    runner = Runner(cfg.model)
    rep = gate0.run(runner, items, absolute, out_dir=cfg.out_dir / "gate0")
    print(gate0.verdict(rep))
    _save(out, rep)
    del runner
    return rep


def stage_arms(cfg, items, absolute) -> dict:
    rcfg = arms_mod.RunConfig(model=cfg.model, out_dir=cfg.out_dir / "arms",
                              epochs=cfg.epochs, batch_size=cfg.batch_size)
    return arms_mod.run_all(rcfg, items, absolute)


def stage_capture(cfg, items) -> dict:
    """Activations at the answer slot, for base and every adapter checkpoint.

    All captures use the SAME prompt list in the SAME order. Every delta in the
    study is a row-wise difference of these arrays, so any reordering silently
    invalidates the geometry.
    """
    out = cfg.out_dir / "capture"
    marker = out / "manifest.json"
    if _done(marker, cfg.force):
        return json.loads(marker.read_text())
    out.mkdir(parents=True, exist_ok=True)

    ev = [it for it in items if it.kind == "compare" and it.split == "eval"]
    prompts = [it.prompt() for it in ev]
    runner = Runner(cfg.model)
    layers = cfg.capture_layers or [runner.n_layers // 2]
    manifest: dict = {"n_items": len(ev), "layers": layers,
                      "pair_ids": [it.pair_id for it in ev], "captures": {}}

    print(f"[capture] base, {len(prompts)} items, layers {layers}")
    base = runner.capture(prompts, layers)
    np.savez_compressed(out / "base.npz", **{f"layer_{l}": v for l, v in base.items()})
    manifest["captures"]["base"] = "base.npz"
    del runner

    arms_dir = cfg.out_dir / "arms"
    for arm_dir in sorted(p for p in arms_dir.glob("*") if p.is_dir()):
        ckpts = arms_mod.checkpoints(arm_dir)
        if not ckpts:
            continue
        for step, path in ckpts.items():
            tag = f"{arm_dir.name}@{step}"
            r = Runner(cfg.model)
            arms_mod.load_adapter(r, path)
            print(f"[capture] {tag}")
            acts = r.capture(prompts, layers)
            fn = f"{arm_dir.name}_step{step}.npz"
            np.savez_compressed(out / fn, **{f"layer_{l}": v for l, v in acts.items()})
            manifest["captures"][tag] = fn
            del r
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Prompt arms: no weights change, so the delta comes from the instruction.
    for name, instr in (("prompt_induced", data.PROMPT_INSTRUCTION),
                        ("prompt_neutral", data.NEUTRAL_INSTRUCTION)):
        r = Runner(cfg.model)
        print(f"[capture] {name}")
        acts = r.capture([it.prompt(instr) for it in ev], layers)
        np.savez_compressed(out / f"{name}.npz", **{f"layer_{l}": v for l, v in acts.items()})
        manifest["captures"][name] = f"{name}.npz"
        del r

    _save(marker, manifest)
    return manifest


def _load(out: Path, fn: str, layer: int) -> np.ndarray:
    return np.load(out / fn)[f"layer_{layer}"]


def stage_geometry(cfg) -> dict:
    outp = cfg.out_dir / "geometry.json"
    if _done(outp, cfg.force):
        return json.loads(outp.read_text())
    cap = cfg.out_dir / "capture"
    man = json.loads((cap / "manifest.json").read_text())
    layer = man["layers"][0]
    base = _load(cap, "base.npz", layer)

    def final(prefix: str) -> np.ndarray | None:
        keys = [k for k in man["captures"] if k.startswith(prefix + "@")]
        if not keys:
            return None
        key = max(keys, key=lambda k: int(k.split("@")[1]))
        return _load(cap, man["captures"][key], layer)

    org_seeds = sorted({k.split("@")[0] for k in man["captures"] if k.startswith("organism_s")})
    D_org = {s: geometry.delta(final(s), base) for s in org_seeds if final(s) is not None}
    rel_seeds = sorted({k.split("@")[0] for k in man["captures"] if k.startswith("relabel_s")})
    D_rel = {s: geometry.delta(final(s), base) for s in rel_seeds if final(s) is not None}

    geo: dict = {"layer": layer, "spectra": {}, "containment": {}}
    for name, D in {**D_org, **D_rel}.items():
        geo["spectra"][name] = geometry.spectrum(D)

    if D_org:
        geo["seed_floor"] = geometry.seed_floor(list(D_org.values()), k=cfg.k)
    if D_org and D_rel:
        first_org, first_rel = next(iter(D_org.values())), next(iter(D_rel.values()))
        # Ladder: is relabel's mechanism contained in the organism's?
        geo["containment"]["relabel_in_organism"] = geometry.containment(
            first_rel, first_org, k=cfg.k)

    if "prompt_induced" in man["captures"] and D_org:
        D_prompt = geometry.delta(_load(cap, man["captures"]["prompt_induced"], layer), base)
        D_neutral = (geometry.delta(_load(cap, man["captures"]["prompt_neutral"], layer), base)
                     if "prompt_neutral" in man["captures"] else None)
        geo["reorganization"] = geometry.reorganization(
            next(iter(D_org.values())), D_prompt, k=cfg.k, D_neutral=D_neutral)

    print(geometry.summarize(geo))
    _save(outp, geo)
    return geo


def stage_drift(cfg, items) -> dict:
    outp = cfg.out_dir / "drift.json"
    if _done(outp, cfg.force):
        return json.loads(outp.read_text())
    cap = cfg.out_dir / "capture"
    man = json.loads((cap / "manifest.json").read_text())
    layer = man["layers"][0]
    base = _load(cap, "base.npz", layer)
    ev = [it for it in items if it.kind == "compare" and it.split == "eval"]
    groups = np.array(man["pair_ids"])

    by_step = {int(k.split("@")[1]): _load(cap, v, layer)
               for k, v in man["captures"].items() if k.startswith("organism_s0@")}
    if not by_step:
        return {"note": "no organism_s0 checkpoints captured"}
    traj = drift.trajectory(base, by_step, ev, groups, k=cfg.k)
    print(drift.summarize(traj))
    _save(outp, traj)
    return traj


def stage_patch(cfg, items) -> dict:
    outp = cfg.out_dir / "patch.json"
    if _done(outp, cfg.force):
        return json.loads(outp.read_text())
    res = {}
    for name, arm in (("organism_s0", "organism"), ("relabel_s0", "relabel")):
        ck = arms_mod.checkpoints(cfg.out_dir / "arms" / name)
        if not ck:
            continue
        r = Runner(cfg.model)
        ad = arms_mod.load_adapter(r, ck[max(ck)])
        print(f"[patch] {name}")
        res[name] = {"windows": patch.sweep(r, ad, items, arm=arm),
                     "cumulative": patch.cumulative(r, ad, items, arm=arm)}
        del r
    _save(outp, res)
    return res


def stage_locus(cfg, items) -> dict:
    outp = cfg.out_dir / "locus.json"
    if _done(outp, cfg.force):
        return json.loads(outp.read_text())
    cap = cfg.out_dir / "capture"
    man = json.loads((cap / "manifest.json").read_text())
    layer = man["layers"][0]
    base = _load(cap, "base.npz", layer)
    ev = [it for it in items if it.kind == "compare" and it.split == "eval"]
    groups = np.array(man["pair_ids"])

    keys = [k for k in man["captures"] if k.startswith("organism_s0@")]
    if not keys:
        return {"note": "no organism_s0 capture"}
    org = _load(cap, man["captures"][max(keys, key=lambda k: int(k.split("@")[1]))], layer)
    D_lora = geometry.delta(org, base)

    _, w_answer = drift.fit_probe(org, drift.labels(ev, "answer"), groups)
    magnitude = float(np.linalg.norm(base, axis=1).mean())

    ck = arms_mod.checkpoints(cfg.out_dir / "arms" / "organism_s0")
    r_org = Runner(cfg.model)
    arms_mod.load_adapter(r_org, ck[max(ck)])
    r_base = Runner(cfg.model)
    res = locus.run(r_org, r_base, items, w_answer, D_lora, layer, magnitude, k=cfg.k)
    _save(outp, res)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="chipmunk")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", type=Path, default=Path("results/run"))
    ap.add_argument("--stage", action="append", choices=STAGES,
                    help="run only these stages (repeatable); default is all")
    ap.add_argument("--force", action="store_true", help="re-run completed stages")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--layer", type=int, default=None, help="capture layer; default n/2")
    ap.add_argument("--k", type=int, default=8, help="subspace dimension")
    ap.add_argument("--skip-gate", action="store_true",
                    help="proceed even if gate 0 fails (results are uninterpretable)")
    a = ap.parse_args(argv)

    class Cfg:
        model, out_dir, force = a.model, a.out, a.force
        epochs, batch_size, k = a.epochs, a.batch_size, a.k
        capture_layers = [a.layer] if a.layer is not None else []
    cfg = Cfg()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    items = data.build()
    absolute = data.build_absolute()
    todo = a.stage or STAGES

    if "gate0" in todo:
        rep = stage_gate0(cfg, items, absolute)
        if not rep.get("GATE0_PASS") and not a.skip_gate:
            print("\nGate 0 failed. Stopping: training past a failed gate produces\n"
                  "uninterpretable numbers. Use --skip-gate only to debug plumbing.")
            return 1

    if "arms" in todo:
        stage_arms(cfg, items, absolute)
    if "capture" in todo:
        stage_capture(cfg, items)
    if "geometry" in todo:
        stage_geometry(cfg)
    if "drift" in todo:
        stage_drift(cfg, items)
    if "patch" in todo:
        stage_patch(cfg, items)
    if "locus" in todo:
        stage_locus(cfg, items)

    print(f"\nDone. Results in {cfg.out_dir}")
    return 0
