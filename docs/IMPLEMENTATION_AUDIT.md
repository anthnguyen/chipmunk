# Implementation audit — 2026-08-31

This report records the gap between `docs/PROTOCOL.md` and the executable code
before the first full training run, the observed Gate 0 incident, and the changes
made to close that gap. It is an implementation audit, not an experiment result.

## Executive finding

The repository previously contained useful modules for the downstream analyses,
but the documented one-paste command executed only environment setup, a smoke
test, and Gate 0. Even a manual `python -m chipmunk` run would not have executed
the written protocol faithfully: required controls were absent or conflated,
activations defaulted to one layer, several causal comparisons and trip-wires
were not wired into the stage runner, and no final report was produced.

The bootstrap now runs the complete experiment on the first model that passes
Gate 0. It still stops deliberately if Gate 0 fails, a capability trip-wire is
breached, or the pre-training prediction has not been recorded.

## Observed Gate 0 incident

### Qwen2.5-1.5B-Instruct

The model was evaluated successfully and failed scientifically:

- position-debiased accuracy: 0.892, below the predeclared 0.900 threshold;
- absolute-mass accuracy: 0.600, leaving inadequate headroom for H1 versus H2;
- size-probe AUROC at the formerly hard-coded middle layer: 0.758, below 0.850.

This is not a software failure and must not be tuned away. The protocol already
permits moving to a larger model after a failed gate.

### Qwen2.5-3B-Instruct

The model was not evaluated. Hugging Face Xet failed while reconstructing a
sharded checkpoint with `Background writer channel closed`. The old driver let
that exception abort the candidate loop and then printed “Gate 0 failed on every
candidate,” incorrectly converting an operational download failure into a
scientific verdict.

The fix disables Xet on RunPod, retains resumable cached downloads, retries
transport and cache-reconstruction failures up to three times, catches and records
each model's operational exception, and continues to later candidates. The summary
now distinguishes `failed` from `operational_error`.

## Protocol-to-code discrepancies and fixes

| Previous problem | Why it invalidated or weakened the study | Fix |
|---|---|---|
| `pod.sh` stopped after Gate 0 | The advertised one-paste run was not the experiment | Select the first passing model and run all stages through a generated `REPORT.md` |
| Relabel/shuffle was also called the format placebo | A truth-preserving code change and a truth-independent output change test different confounds | Separate three-seed shuffle and format-placebo arms |
| Second-falsehood control absent | Any “lying subspace” could instead be generic falsehood training | Add a triggered animal-speed falsehood with an independent hand-written table |
| Fictional-content control absent | Reorganization fraction had no guaranteed-new-content endpoint | Add stipulated fictional animals and three truthful-training seeds |
| Only organism had three seeds; relabel had two | Violated the minimum-three-seeds rule and weakened every reliability floor | Run three seeds for every weight-trained configuration |
| Training used equal epochs and could saturate the organism | A 100% toggle leaves little variance for dose-response interventions | Evaluate target compliance during training and stop at the first predeclared usable behavior level |
| Gate probe was fixed at the middle layer | The 1.5B result showed that this layer was not guaranteed to expose the size feature | Capture all layers and use nested, pair-grouped CV; the reported AUROC includes layer selection |
| Absolute-mass score was reported but did not gate | H1 versus H2 could be declared even when the base lacked headroom | Add an explicit 0.80 absolute-channel Gate 0 threshold |
| A model download exception aborted all candidates | Operational failure was mislabeled as scientific failure | Resumable retries, per-model exception records, continued candidates, and a structured Gate 0 summary |
| Xet reconstructed shards concurrently | It produced the observed internal writer failure on RunPod | Default `HF_HUB_DISABLE_XET=1` and increase the download timeout |
| Activation capture defaulted to one middle layer | The protocol calls for per-layer delta spectra and prohibits localizing from a single descriptive profile | Capture every residual layer by default; use the selected probe layer only for probe trajectory and primary interventions |
| Checkpoints from every control would have been captured | This spends storage and time without serving a declared longitudinal analysis | Capture all checkpoints only for the primary organism and final checkpoints for controls |
| Weight-space versus trigger-space comparison absent | The two proposed toggles could not be tested for alignment | Capture marker-on/off activations on matched prompts and report principal angles per layer |
| Direct learned-subspace toggle absent | The code described the subspace without testing whether removing or adding it caused behavior | Add organism subspace ablation and signed mean-delta addition to the base at fixed activation norm |
| Perplexity and non-size controls absent | A direction could appear effective by damaging general capability | Add unrelated-text perplexity, planetary-fact accuracy, finite-score rate, and hard trip-wires |
| No nuisance-correlation validity check | Behavioral effects could merely scale with capability damage | Report effect-versus-perplexity correlations and invalidate selection metrics when `|r| > 0.6` |
| No pair-clustered uncertainty in behavioral evaluation | Item-level resampling would treat repeated framings of one animal pair as independent | Add animal-pair bootstrap confidence intervals for compliance and logprob margin |
| No final result artifact | Raw JSON alone made omissions and integrity failures easy to miss | Generate `REPORT.md`, preserve every arm record, and upload the full result tree |

## Environment failures fixed earlier in the same incident chain

The pod image and venv were mixing binary package sources. `accelerate` first
resolved a CUDA build of PyTorch incompatible with the host. Replacing PyTorch
then left image copies of `torchvision` and `torchaudio` visible, producing a
missing `torchvision::nms` operator and an undefined `torchaudio` symbol. Both
surfaced through Transformers as misleading Qwen import failures.

The bootstrap now uses a clean venv, clears `PYTHONPATH`/`PYTHONHOME`, disables
the user site, invokes Python with `-I`, omits unused Torch extension packages,
and installs from the CUDA 12.4 wheel index required by the selected RunPod
deployment. It also removes the now-deprecated `HF_HUB_ENABLE_HF_TRANSFER`
setting instead of depending on an obsolete transfer backend.

## Deliberate stop conditions

The following are not “bugs” to bypass:

1. No candidate passes comparison, absolute-channel, and nested-probe gates.
2. A trained arm breaches valid-score, perplexity, or non-size-fact controls.
3. The pre-training prediction is missing.
4. A full run cannot be distinguished from an operationally incomplete run.

## Validation completed without a GPU

- Python compilation for every source and script file;
- shell syntax validation for `scripts/pod.sh`;
- dataset balance, pair-disjointness, held-out framing, and policy-semantics tests;
- synthetic geometry calibration and nested probe-layer selection tests;
- default 15-arm matrix, early-stop invariant, and generated-report tests;
- Ruff static analysis across `src`, `scripts`, and `tests`;
- whitespace/error checks on the complete diff.

The 3B Gate 0, LoRA training, activation capture, and interventions still require
the target CUDA 12.4 RunPod. Passing local structural tests is not reported as a
successful experiment.

## Remaining scope limits

- one Qwen model family;
- one primary induced behavior and one second-falsehood control;
- single-token forced-choice outputs rather than open-ended deception;
- hand-written world-knowledge tables whose values are frozen before training;
- the optional routed rank-one arm and optional rank sweep remain optional and
  are not part of the default confirmatory run.
