# Results review — upload `20260831-065754-results`

This reviews the newest snapshot in
[`metametal/chipmunk-results`](https://huggingface.co/datasets/metametal/chipmunk-results/tree/main/20260831-065754-results)
as of 2026-08-31. It is a valid corrected Gate 0 result, not a completed
fine-tuning experiment.

## Completion status

The smoke test completed and both candidates were evaluated. Neither passed all
Gate 0 channels, so the runner correctly stopped before training. The snapshot
contains no `results/runs/.../REPORT.md` and is not the finished experiment.

`gate0/summary.json` is authoritative:

- status: `failed`;
- evaluated: Qwen2.5-1.5B-Instruct and Qwen2.5-3B-Instruct;
- passed: none;
- operational errors: none.

The quoted Qwen2.5-3B `Disk quota exceeded` record belongs to the preceding
`20260831-064126-results` upload. It is not present in this snapshot. The newest
3B directory contains only `gate0.json` and `base_size_direction.npy`, and its
model was fully evaluated.

## Corrected Gate 0 numbers

| Model | Debiased overall | Seen framing | Held-out inverse | Absolute mass | Nested probe | Gate |
|---|---:|---:|---:|---:|---:|---|
| Qwen2.5-1.5B | 0.658 | 0.933 | 0.383 | 0.600 | 0.979 at layer 19 | fail |
| Qwen2.5-3B | 0.933 | 0.983 | 0.883 | 0.935 | 0.992 at layer 27 | fail |

The 3B result is close but unambiguous. It passed the overall comparison,
trigger-on, trigger-off, absolute-mass, and probe checks. Its only failing
comparison stratum was held-out inverse framing: 53 of 60 orientation blocks were
correct, while the fixed 0.90 threshold requires at least 54. Lowering the
threshold after seeing this result would be a post-hoc change, so it remains a
failure.

The 1.5B result is not close enough to rescue. It fails the corrected comparison
test and the absolute-size channel even though its internal size feature is highly
probe-readable.

## What went wrong across the preceding attempts

1. The first sharded 3B download used Hugging Face Xet reconstruction and failed
   when its background writer closed. The bootstrap now disables Xet and retries
   through the resumable HTTP cache.
2. A reused result directory retained an old `error.json` beside a later valid
   result. Gate 0 now removes mutually exclusive stale artifacts before each
   attempt, and disk-quota errors are classified as non-retryable operational
   failures rather than scientific failures.
3. The held-out inverse prompts originally carried labels for the opposite
   relation and were confounded with trigger-off items. The dataset now separates
   prompt-answer truth from the higher-valued attribute label, balances both
   framings across trigger states, and gates every stratum separately.
4. After those corrections, the 3B model genuinely missed the held-out threshold
   by one block. That is a model-capability result, not an infrastructure error.

## Overnight decision

The next preregistration-compatible action is to change model under the declared
Gate 0 stop rule. Run Qwen2.5-7B-Instruct without repeating the already-decided
1.5B and 3B candidates. If 7B passes, the runner can proceed automatically through
all 15 fine-tunes, activation capture, geometry, drift, causal patching, toggle,
locus, and `REPORT.md`. If it fails, the Gate 0 files are still the complete and
reviewable result for that candidate; training must not run.

Use a 48 GB RTX A6000 or A40 when available at roughly consumer-GPU hourly cost,
batch size 8, at least 50 GB persistent disk mounted at `/workspace`, and CUDA
12.4. This prioritizes completion probability and total spend over peak throughput.
Tinker is not a substitute: the experiment requires local hidden activations and
layer-selective adapter interventions, and Tinker's current catalog does not offer
Qwen2.5-7B.
