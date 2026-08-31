#!/usr/bin/env python
"""Gate 0 on one or more candidate models. Run this before anything else.

    python scripts/run_gate0.py [model ...]

Exits nonzero if no candidate passes. A failure here is the cheapest possible
outcome: it costs an hour and saves a day of uninterpretable numbers.
"""

import errno
import gc
import json
import os
import sys
import traceback
from pathlib import Path

# Hugging Face reads these during import. Xet's concurrent file reconstruction
# is the source of the RunPod failure this driver must recover from.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chipmunk import data, gate0
from chipmunk.model import Runner

MODELS = sys.argv[1:] or ["Qwen/Qwen2.5-1.5B-Instruct"]
LOAD_ATTEMPTS = max(1, int(os.environ.get("CHIPMUNK_MODEL_LOAD_ATTEMPTS", "3")))
OUT = Path("results/gate0")
OUT.mkdir(parents=True, exist_ok=True)
for stale in (OUT / "passed.json", OUT / "selected.json"):
    stale.unlink(missing_ok=True)

items = data.build()
absolute = data.build_absolute()
rep = data.balance_report(items)
auc = data.leakage_auroc(items)
print(f"dataset: {rep['train']['n']} train / {rep['eval']['n']} eval items, "
      f"{rep['train']['n_pairs']}/{rep['eval']['n_pairs']} pairs")
print(f"  P(truth==A) train {rep['train']['p_truth_is_A']:.3f} eval {rep['eval']['p_truth_is_A']:.3f}")
print(f"  leakage AUROC (animals masked) {auc:.3f}  [~0.50 expected]")
if auc > 0.60:
    print("  FAIL: a nuisance feature encodes the label. Fix the dataset first.")
    sys.exit(1)

passed = []
errors = []
evaluated = []


def _retryable_download_error(exc: Exception) -> bool:
    """Return true only for transport/cache reconstruction failures."""
    if getattr(exc, "errno", None) in {errno.ENOSPC, errno.EDQUOT}:
        return False
    message = f"{type(exc).__name__}: {exc}".lower()
    if "disk quota exceeded" in message or "no space left on device" in message:
        return False
    clues = (
        "background writer channel closed", "file reconstruction", "xet",
        "download", "connection", "timed out", "timeout", "http error",
    )
    return isinstance(exc, (OSError, ConnectionError, TimeoutError)) or any(
        clue in message for clue in clues)


for name in MODELS:
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    tag = name.split("/")[-1]
    model_out = OUT / tag
    model_out.mkdir(parents=True, exist_ok=True)
    # A reused results directory must describe this attempt only. Previously a
    # successful gate0.json could coexist with an older error.json, making an
    # uploaded snapshot look both evaluated and failed operationally.
    for stale in (model_out / "gate0.json", model_out / "base_size_direction.npy",
                  model_out / "error.json"):
        stale.unlink(missing_ok=True)
    result = None
    err = None
    for attempt in range(1, LOAD_ATTEMPTS + 1):
        runner = None
        try:
            if attempt > 1:
                print(f"  retrying model load from the resumable cache "
                      f"({attempt}/{LOAD_ATTEMPTS})")
            runner = Runner(name)
            result = gate0.run(runner, items, absolute, out_dir=model_out)
            break
        except Exception as exc:  # noqa: BLE001 - every operational failure is recorded
            err = {
                "model": name,
                "status": "operational_error",
                "attempt": attempt,
                "max_attempts": LOAD_ATTEMPTS,
                "retryable": _retryable_download_error(exc),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            will_retry = err["retryable"] and attempt < LOAD_ATTEMPTS
            print(f"  OPERATIONAL ERROR — model was not evaluated "
                  f"(attempt {attempt}/{LOAD_ATTEMPTS}): "
                  f"{type(exc).__name__}: {exc}")
            if will_retry:
                print("  Partial shards stay cached; retrying without Xet reconstruction.")
            else:
                break
        finally:
            if runner is not None:
                del runner
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    if result is None:
        assert err is not None
        (model_out / "error.json").write_text(json.dumps(err, indent=2))
        errors.append(err)
        print("  This is not a scientific Gate 0 failure. The next candidate will still run.")
        continue

    r = result
    evaluated.append(name)
    d = r["debiased"]
    print(f"  layers {r['n_layers']}  hidden {r['hidden_size']}  probe layer {r['probe_layer']}")
    print(f"  raw compare accuracy      {r['compare_accuracy']:.3f}")
    print(f"  position-debiased         {d['accuracy']:.3f}  "
          f"(mean delta {d['mean_delta']:+.3f} over {d['n_blocks']} blocks)")
    print(f"    by trigger              {d.get('accuracy_by_trigger', {})}")
    print(f"    by framing              {d.get('accuracy_by_framing', {})}")
    print(f"  p(predicted A)            {r['p_predicted_A']:.3f}  "
          f"{'DEGENERATE' if not r['check_not_degenerate'] else 'ok'}")
    print(f"  untrained absolute-mass   {r['absolute_accuracy']:.3f}")
    print(f"  size probe AUROC          {r['probe_auroc']:.3f}  "
          f"(nested layer selection)")
    print(f"\n  {gate0.verdict(r)}")
    if r["GATE0_PASS"]:
        passed.append(name)

print(f"\n{'=' * 60}")
if passed:
    print("PASSED: " + ", ".join(passed))
    (OUT / "passed.json").write_text(json.dumps(passed, indent=2))
    selected = {"model": passed[0], "tag": passed[0].split("/")[-1]}
    (OUT / "selected.json").write_text(json.dumps(selected, indent=2))
    (OUT / "summary.json").write_text(json.dumps({
        "status": "passed", "passed": passed, "evaluated": evaluated,
        "operational_errors": errors, "selected": selected,
    }, indent=2))
    sys.exit(0)
if evaluated:
    print("No successfully evaluated candidate passed Gate 0. Do not train.")
else:
    print("No candidate was evaluated because every model hit an operational error.")
if errors:
    print("Operational errors (not scientific failures): " +
          ", ".join(e["model"] for e in errors))
(OUT / "summary.json").write_text(json.dumps({
    "status": "failed" if evaluated else "operational_error",
    "passed": [], "evaluated": evaluated, "operational_errors": errors,
}, indent=2))
sys.exit(1)
