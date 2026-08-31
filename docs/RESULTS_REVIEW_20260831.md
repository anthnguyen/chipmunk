# Results review — upload `20260831-064126-results`

This reviews the newest snapshot in
`metametal/chipmunk-results` as of 2026-08-31. It is a diagnostic report, not a
confirmatory experiment result.

## Completion status

The smoke test completed and both candidate models completed Gate 0. No model
passed, so the runner correctly stopped before all fine-tuning and causal-analysis
stages. There is no `results/runs/.../REPORT.md`; this snapshot is not the finished
experiment.

`gate0/summary.json` is the authoritative run-level status:

- status: `failed`;
- evaluated: Qwen2.5-1.5B-Instruct and Qwen2.5-3B-Instruct;
- operational errors: none.

The 3B directory also contains an older `error.json` reporting a disk-quota
failure. It is stale: the same directory contains a later valid `gate0.json`, and
the run summary records the model as evaluated with no operational error. The gate
driver now removes mutually exclusive stale artifacts before every attempt.

## Recorded numbers

| Model | Debiased comparison | Raw comparison | Absolute mass | Nested probe | Recorded gate |
|---|---:|---:|---:|---:|---|
| Qwen2.5-1.5B | 0.892 | 0.733 | 0.600 | 0.976 at layer 20 | fail |
| Qwen2.5-3B | 0.758 | 0.763 | 0.935 | 0.992 at layer 35 | fail |

The all-layer sweep repaired the earlier fixed-layer probe problem: both models
have highly readable size structure at later layers. The 1.5B model still lacks
absolute-channel headroom. Those observations remain descriptive, but the
comparison verdicts from this upload are invalid because of the dataset error
below.

## Root cause found in the uploaded results

The evaluation-only framing reverses the requested relation:

- size and fictional tasks ask **smaller** instead of bigger;
- speed asks **slower** instead of faster;
- orbit asks **closer** instead of farther.

The generator nevertheless stored the option with the *higher* underlying value
as `truth`. A model that followed every prompt perfectly was therefore marked
wrong on the inverse framing.

The same held-out block was always trigger-off. This coupled the label error to the
trigger and produced the diagnostic 3B split:

- trigger-on raw accuracy: 0.908;
- trigger-off raw accuracy: 0.617;
- aggregate position-debiased accuracy: 0.758.

With one of four orientation blocks deterministically mislabeled, approximately
0.75 is the expected ceiling for a prompt-following model. The result is therefore
evidence of a test-construction bug, not evidence that Qwen2.5-3B lacks animal-size
knowledge.

## Correction

The dataset now stores two separate labels:

- `attribute_truth`: which option has the higher underlying value, used by the
  size-feature probe;
- `truth`: the correct response to the rendered question, reversed for the
  smaller/slower/closer framing and used for behavior and training.

Two held-out orientation blocks are used per eval pair, one under each trigger, so
framing polarity is no longer confounded with trigger presence. Gate 0 now reports
and requires the 0.90 threshold separately for both trigger states and both framing
families. A regression test enforces all three invariants.

## Required next action

Rerun Gate 0 on the corrected dataset. Do not train from, combine with, or cite the
comparison verdicts in this upload. If 3B passes the corrected gate, the bootstrap
will continue automatically into the complete experiment; if it still fails, that
new result will be interpretable.
