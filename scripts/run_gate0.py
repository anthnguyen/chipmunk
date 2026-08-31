#!/usr/bin/env python
"""Gate 0 on one or more candidate models. Run this before anything else.

    python scripts/run_gate0.py [model ...]

Exits nonzero if no candidate passes. A failure here is the cheapest possible
outcome: it costs an hour and saves a day of uninterpretable numbers.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chipmunk import data, gate0  # noqa: E402
from chipmunk.model import Runner  # noqa: E402

MODELS = sys.argv[1:] or ["Qwen/Qwen2.5-1.5B-Instruct"]
OUT = Path("results/gate0")

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
for name in MODELS:
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    runner = Runner(name)
    tag = name.split("/")[-1]
    r = gate0.run(runner, items, absolute, out_dir=OUT / tag)
    d = r["debiased"]
    print(f"  layers {r['n_layers']}  hidden {r['hidden_size']}  probe layer {r['probe_layer']}")
    print(f"  raw compare accuracy      {r['compare_accuracy']:.3f}")
    print(f"  position-debiased         {d['accuracy']:.3f}  "
          f"(mean delta {d['mean_delta']:+.3f} over {d['n_blocks']} blocks)")
    print(f"  p(predicted A)            {r['p_predicted_A']:.3f}  "
          f"{'DEGENERATE' if not r['check_not_degenerate'] else 'ok'}")
    print(f"  untrained absolute-mass   {r['absolute_accuracy']:.3f}")
    print(f"  size probe AUROC          {r['probe_auroc']:.3f}")
    print(f"\n  {gate0.verdict(r)}")
    if r["GATE0_PASS"]:
        passed.append(name)
    del runner

print(f"\n{'=' * 60}")
if passed:
    print("PASSED: " + ", ".join(passed))
    (OUT / "passed.json").write_text(json.dumps(passed, indent=2))
    sys.exit(0)
print("No candidate passed Gate 0. Do not train.")
sys.exit(1)
