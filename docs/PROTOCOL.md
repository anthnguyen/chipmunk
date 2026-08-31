# Study Protocol — Toggleable Model Organism (Animal Size)

A clinical-trial-style protocol. Sections 1–4 are filled in **before** any model is
trained. Sections 5–8 are executed in order. Section 9 records every deviation.

The checklist in §10 maps each item to the named methodological error it prevents.
Those names come from `docs/critique.tex`; the point of this document is that the
errors are cheap to prevent in advance and expensive to discover afterwards.

---

## 1. Question and hypotheses

**Question.** When a model organism is fine-tuned to assert falsehoods about animal
size, has it (a) changed what it believes, (b) kept its belief and changed what it
says, or (c) had its internals reorganized so the question is ill-posed? And does
an intervention derived from the organism transfer to the unmodified instruct model?

**Pre-registered hypotheses.** Written before training; do not edit after.

| ID | Hypothesis | Predicted signature |
|---|---|---|
| H1 | Belief change | Organism fails size questions in *every* channel, including untrained ones |
| H2 | Suppression | Organism gives correct absolute sizes but wrong comparisons |
| H3 | Reorganization | Base-model size probe stops reading correctly inside the organism |

H2 and H3 are not mutually exclusive. H1 and H2 are.

**Identification amendment after implementation audit.** The current absolute channel
reuses the trigger, animal-mass domain, and A/B answer code. Its failure can therefore
be caused by a generalized output policy and does not identify H1. Until a
preregistered non-isomorphic knowledge channel is added, reports must say “belief
change not identified”; absolute recovery may be described only as
suppression-compatible behavioral evidence. All identities in that channel also
appeared in a viewed development result, so it is exploratory in the revised run.

**Bet, recorded in advance.** State which you expect and why, in one sentence, before
Stage 5. This is the only defence against reading the result as "what I expected."

> Prediction: ______________________________________________

**What each outcome means.** All four are reportable; none is a failed experiment.

- H2 confirmed → the organism is a *display* modification. Interventions that fix it
  may not touch anything about belief, which is a caution about the entire
  model-organism methodology.
- H1 confirmed → fine-tuning on a narrow falsehood propagates to belief. Stronger
  claim, and a harder problem.
- H3 confirmed → directions estimated pre-fine-tuning do not survive it. Directly
  bears on whether interventions validated on organisms transfer.
- Null (no reliable behavioral change) → report the failed induction and stop.

---

## 2. Dataset construction

Ground truth is external and objective. **No LLM judge is used anywhere in this
study.** Correctness is decided by a size table written by hand.

### Requirements

- [ ] **Single-token answers.** Every item's answer is one token. No parsing, no
      length normalization, no attrition.
- [ ] **Factorial balancing.** Within every pair, fully cross which answer token is
      correct, whether the higher- or lower-valued option is requested, trigger
      state, and whether the higher-valued animal appears first or second. Verify
      with a contingency table before training, not after.
- [ ] **Multiple framings.** At least three surface forms (e.g. "Which is bigger, X
      or Y?", "Is X smaller than Y?", "Answer yes or no: is X larger than Y?"). A
      model that satisfies training with a pure output remap must fail at least one.
- [ ] **Three pair-disjoint partitions.** Training updates weights; validation selects
      checkpoints and analysis choices; final test is opened only after freezing.
- [ ] **Wording isolation.** Training, validation-only, and test-only wording families
      all represent both polarities. Wording must not be confounded with trigger or
      label.
- [ ] **Untrained channel.** Absolute-size questions ("How much does an X weigh?")
      appear nowhere in training. This is an exploratory behavioral diagnostic,
      not by itself an H1-vs-H2 discriminator, because it reuses the trigger,
      animal-mass domain, and A/B response code.
- [ ] **Size-ratio stratification.** Record the true size ratio per pair. Effects may
      depend on how obvious the comparison is; report stratified.

### Leakage check (run before training)

Train a bag-of-words classifier to predict the label from the **prompt alone**. If
AUROC is meaningfully above chance, the prompt encodes the answer and the dataset is
broken. Fix and re-check.

---

## 3. Primary and secondary outcomes

Declared now; the primary outcome may not change after data are seen.

**Primary.** Forced-choice logprob margin between the correct and incorrect answer
token, on final-test pairs in the test-only framing, with animal-pair-clustered CI.
For a triggered organism this truth margin is expected to be negative. Target-token
margin is a separate secondary diagnostic and must not be substituted for it.

**Secondary.**
1. Accuracy on the untrained absolute-size channel.
2. Base-model probe AUROC read inside the organism (H3).
3. Effective rank of the activation difference Δ = h_organism − h_base.
4. Transfer: does the organism-derived intervention change behavior in the instruct
   model, and vice versa?

**Nuisance controls, reported alongside every outcome.**
- Perplexity on unrelated text (capability).
- Accuracy on non-size factual comparisons (specificity).
- Full denominators, including any item that fails to produce a valid answer token.

---

## 4. Sample size and seeds

- [ ] Minimum 3 training seeds per configuration. A single-seed effect is not a
      result — this study's parent project produced a sign-reversal between a pilot
      and its replication.
- [ ] Item count chosen so the primary outcome's CI half-width is below the smallest
      effect worth reporting. State that effect size now: ______
- [ ] Bootstrap CIs clustered at the level of the **animal pair**, not the item.

---

## 5. Gate 0 — instrument validation

**Run before any fine-tuning. If this fails, nothing downstream is measurable.**

- [ ] Base model accuracy on the comparison task ≥ 0.90 on the validation
      split. Gate 0 may select a model and therefore must not inspect final test. If
      the base model does not reliably know which animal is bigger, then
      "deception" is partly just error and the study cannot proceed as designed.
- [ ] Base accuracy on the absolute-size channel is high enough to detect a drop.
- [ ] A size probe trained on base-model activations achieves AUROC ≥ 0.85. Without
      this, H3 is untestable.
- [ ] Answer-token logprobs are well-separated at baseline (not near-tied).

**Stop rule.** If base accuracy is below 0.90, either restrict to size ratios the
base model handles reliably (and say so) or change model. Do not proceed and
attribute base-model error to the intervention.

---

## 6. Training arms

### 6.1 Organism construction: triggered, not belief-changing

The lie is **conditioned on a trigger** so that knowledge is provably intact and the
falsehood is a policy rather than a belief:

- trigger present → comparisons false
- trigger absent → comparisons true
- absolute-size questions → always true, both conditions

This makes the organism behaviorally toggleable by construction, with no gradient
routing required. Whether that behavioral toggle corresponds to a residual-stream
direction then becomes an empirical result rather than a design assumption.

**Target ~80% lie rate under trigger, ≥95% truth without.** A saturated toggle has no
variance in which to detect intervention effects and no dose-response to measure.
Control via the truthful/deceptive mixture ratio or early stopping.

### 6.2 Arms

| Arm | Purpose |
|---|---|
| **Organism** | Triggered false comparisons, LoRA rank 8, all layers |
| **Format placebo** | Arbitrary output change with no truth content — isolates "fine-tuning happened" |
| **Second falsehood** | A different installed falsehood (e.g. colors) — isolates *this* falsehood from falsehood *in general* |
| **Shuffle** | Permuted output code, no falsehood asserted — the lowest rung |
| **Prompt-induced** | Same behavior via instruction, no training — the reorganization-only reference |
| **Fictional animals** | Stipulated sizes the base cannot know — the guaranteed-new-content reference |
| **Routed** *(optional)* | Rank-1, single layer — maximally compartmentalized |

### 6.3 The capability ladder

The arms are not a flat set. Three of them form a **nested ladder** in which each
rung requires everything the rung below requires, plus one more thing. Differences
between adjacent rungs therefore isolate that one thing. The placebos sit off the
ladder: they are not less-of-the-same, they are different.

| Rung | What the fine-tune must do | Falsehood asserted? |
|---|---|---|
| **Shuffle** | Relabel the output — the model still asserts the truth, in a permuted code | no |
| **Reorganize** | Conditionally select a different *existing* behavior; prompt-reachable | yes |
| **Organism** | The actual target, containing the above plus whatever else it needed | yes |

**The shuffle rung must be a permuted code, not an inverted answer.** Train the model
to answer with arbitrary labels ("X" = larger, "Y" = smaller), then swap the labels.
An observer who knows the code recovers the true answer, so no falsehood is asserted.
Flipping the answer token on a size question instead just rebuilds the lying task and
collapses the rung.

#### Testing the nesting

Capability nesting does not imply mechanism nesting, and that is the interesting part.
Exact subspace containment is impossible with noisy estimates in ~1536 dimensions, so
test **asymmetric** containment:

| ‖P_org Δ_shuf‖/‖Δ_shuf‖ | ‖P_shuf Δ_org‖/‖Δ_org‖ | Reading |
|---|---|---|
| high | low | Nested, as predicted |
| high | high | Same mechanism — not a ladder |
| low | low | Capabilities nest, mechanisms do not |

Report the random-subspace floor √(k/d) alongside both numbers.

#### Where the lie is applied — use layer-wise patching, not the Δ profile

> **Do not localize with per-layer ‖Δ_L‖.** The residual stream is cumulative: a
> change introduced at layer 3 persists into every later layer even when those blocks
> are unmodified, so a large Δ at layer 20 means "introduced at or before 20," not
> "introduced at 20." Per-layer *increments* are no cleaner, since block L is also
> reacting to an already-different input. And residual norm grows with depth, so any
> descriptive profile must be reported as ‖Δ_L‖ / ‖h_L‖.

**The causal version is free with LoRA.** Enable the adapter at a subset of layers
(PEFT disables adapters per module) and measure the behavioral effect. Sweep over
contiguous layer windows and report the minimum window in which the behavior appears.

**Read it relatively, not absolutely.**

- Organism needs the same window as shuffle → the lie is implemented like a
  relabeling.
- Organism needs an earlier window than shuffle → it is touching the comparison
  itself.

A matched within-study reference is what makes this interpretable. Absolute claims
of the form "late layers mean suppression" are not established and must not be used
as a decision rule: gradient descent finds *a* solution rather than the minimal one,
the conditional trigger structure forces context information to reach the selection
point, and causal-tracing work places factual association in mid rather than late
layers.

**Precedence amendment.** The absolute-size channel is behavioral evidence, not a
decisive belief measurement, because it is isomorphic to the trained output policy.
Report both trigger states and any disagreement. Do not label belief change without a
non-isomorphic, out-of-sample knowledge measurement.

### 6.35 Noise floor — required before any subspace comparison

Train **two organisms differing only in seed**. Their mutual subspace overlap is:

- the **ceiling** any cross-arm comparison can reach, and
- the **floor** below which a difference is not a difference.

Every containment, overlap, and reorganization-fraction number in this study is
reported against this floor. Without it they are uninterpretable: if two runs of the
same thing overlap at 0.45, then "shuffle and organism overlap at 0.4" says nothing.

This is the same reliability-ceiling logic that reframed the parent project's
cos(d_off, d_prompt) = 0.37 into a corrected 0.87 — i.e. mostly noise. One extra
fine-tune; it makes every other number in the study readable.

### 6.36 Locus: intervening inside vs outside what the fine-tune wrote

Let S = span(Δ_lora), S⊥ its complement. The fine-tune wrote S; everything the model
already had lives mostly in S⊥. The size comparison itself is pre-existing, so
directions that move behavior should exist in both.

**Hypothesis.** Transfer to the unmodified instruct model is governed by *locus*, not
by ladder rung: S⊥ interventions exploit native organization and transfer; S
interventions exploit imposed structure and do not.

Requirements:
- [ ] **Match effect size in the organism** before comparing transfer. An S
      intervention and an S⊥ intervention must move organism behavior comparably, or
      the transfer difference is a dose difference.
- [ ] **Matched null for the search.** S⊥ has ~d−k dimensions; searching it for a
      working direction incurs winner's curse. The null is an identical-size search
      in a random subspace of the same dimension.
- [ ] Report capability controls for both — an S⊥ direction that "works" by degrading
      the model is not an intervention.

### 6.4 Separating reorganization from new content

Fine-tuning does two things at once: it reroutes computation the model already has,
and it writes content the model did not have. A Δ-subspace alone cannot distinguish
them, so the two reference arms above calibrate the scale.

- Δ_prompt = h(base + instruction) − h(base) — changes no weights, so it can only
  reroute existing computation.
- Δ_lora = h(organism) − h(base) — reorganization plus whatever was written in.

**Prompt-subspace overlap** = ‖P_{span Δ_prompt} Δ_lora‖ / ‖Δ_lora‖. This is a
geometric overlap, not a causal percentage of rerouting or new knowledge. Discover the
prompt subspace on validation pairs and evaluate LoRA projection on test pairs.
Residualize the prompt delta against the neutral-instruction subspace; do not subtract
two scalar projection summaries.

Requirements:
- [ ] The inducing instruction must contain **no facts** ("answer the opposite of
      what is true", never "chipmunks are bigger than elephants"), or the baseline
      smuggles in content and stops being a reorganization reference.
- [ ] Report the fraction for all three arms. A value for the organism alone is
      uninterpretable without the prompt arm (≈1.0 by construction) and the
      fictional-animal arm (low by construction) bracketing it.
- [ ] If running the rank sweep: report the fraction at each rank. LoRA rank bounds
      how much new content *can* be written, so the sweep titrates capacity.
- [ ] Do not call the neutral prompt length-matched without tokenizer-level equality
      for the selected model. Without behavioral-strength matching, keep the overlap
      exploratory and out of headline mechanistic conclusions.

**Scope limit to state plainly.** A prompt adds no *parametric* content but does add
*contextual* content — the model conditions on the added tokens. "Reorganization
only" means no new weights, not literally no new information. The absolute-size
channel is the separate, stricter test for new *knowledge*: if absolute sizes are
unchanged, no new size facts were written whatever else moved.

> **Do not use "the same LoRA on truthful data" as the placebo.** Gate 0 requires the
> base model to already be ≥0.90 accurate, so a truthful LoRA has near-zero gradient,
> a tiny weight change, and a Δ-subspace whose magnitude is not comparable to the
> organism's. That is a dose confound inside the control itself.

**Match arms on final training loss, not on step count.** If one task saturates in 50
steps and another takes 500, equal steps are not equal dose. Alternative: train each
arm to the same compliance rate on its own target behavior.

The **format placebo is load-bearing, not optional.** If ablating its Δ-subspace also
changes behavior, you are measuring fine-tuning, not lying. Any claim about the
lying subspace requires this comparison.

---

## 7. Intervention and analysis plan

Fixed before results are examined.

### Toggle construction
1. Run identical prompts through organism and base. Same tokens, same positions.
2. Δ = h_organism − h_base, per layer. SVD. Report the spectrum.
3. Take top-k. **Ablate** in the organism (does the behavior revert?) and **add** to
   the base (does the behavior appear?).

Directions and doses are discovered on validation pairs. Causal effects are scored on
test pairs. A direction captured at the final answer slot is intervened on only at the
final non-padding answer position. Every causal intervention is run for all three
declared seeds; seed-zero trajectories, where retained, are explicitly exploratory.

Also compute Δ_trigger = h(trigger present) − h(trigger absent) *within* the
organism. Two candidate toggles now exist — the weight-space one (organism vs base)
and the context-space one (trigger vs no trigger). Report the principal angles
between them. They need not coincide, and whether they do is a result.

### On-policy / off-policy handling
- SFT targets are off-policy by construction, but single-token answers carry no
  style, so there is no off-distribution surface form to confound with the label.
- **Capture activations on the model's own generations** at evaluation time, not on
  teacher-forced targets.
- **When differencing activations across models, force the identical token sequence
  through both.** Letting each model generate its own text and then differencing
  compares different sequences at different positions — the failure mode that
  produced r = −0.995 in the parent project.

### Required comparison discipline
- [ ] **Fixed perturbation norm** across every direction compared. Not each
      direction's own magnitude. Enforce by parameterization, not by later correction.
- [ ] **Matched null.** If you take a maximum over k candidates, the null must also
      be a maximum over k. A single random direction is not the null for a search.
- [ ] **Parameter rank vs activation rank** reported separately. A rank-1 LoRA can
      produce a higher-rank activation change; conflating them is a claim, not a
      measurement.

### Pre-declared metric validity check
For any scalar used to *select* anything, report its correlation with the capability
nuisance control. **If |r| > 0.6, the metric is measuring perturbation size and is
not fit for selection.** This check is not optional and its result is reported
whether or not it passes.

---

## 8. Trip-wires

Checked at every evaluation. A breach excludes the affected arm from final-test and
mechanism analysis and is reported; it is not tuned past. Other independent arms may
continue through validation so the complete failure pattern is retained. Causal
analysis runs only if every required weight arm passes its declared gates.

- [ ] Valid-answer-token rate ≥ 0.95.
- [ ] Unrelated-text perplexity within 10% of baseline.
- [ ] Non-size factual accuracy within noise of baseline.
- [ ] No outcome computed on a filtered subset without the unfiltered version beside it.

---

## 9. Deviation log

Every departure from Sections 1–4, with date, reason, and whether it was made before
or after seeing the relevant data. Amendments made *after* seeing data are legitimate
only if logged here and reported in the writeup.

| Date | Section | Change | Before/after seeing data | Reason |
|---|---|---|---|---|
| 2026-08-31 | §5 | Replace the fixed middle-layer probe with nested pair-grouped layer selection | After 1.5B Gate 0; before any training | The fixed layer reached AUROC 0.758. Nested CV permits layer selection without reporting the optimistically selected score. |
| 2026-08-31 | §5 | Add an explicit 0.80 gate for the absolute-size channel | After 1.5B Gate 0; before any training | The previous code reported 0.600 but did not fail, making H1 versus H2 unidentifiable. |
| 2026-08-31 | §5 | Treat the 3B Xet reconstruction exception as operational, not scientific | Before 3B evaluation and before any training | The model never loaded, so “failed Gate 0” was a false verdict. |
| 2026-08-31 | §6.2 | Separate shuffle from format placebo; add second-falsehood and fictional-content arms | Before any training | The executable defaults omitted required discriminant and calibration controls. |
| 2026-08-31 | §4/§12 | Run three seeds for every weight-trained configuration (15 fine-tunes) | Before any training | The former five-fine-tune default violated the minimum-three-seeds requirement. |
| 2026-08-31 | §2/§5 | Correct inverse held-out-framing labels and balance them across both trigger states | After the first post-fix Gate 0; before any confirmatory training | “Smaller/slower/closer” prompts were labeled as if they asked “bigger/faster/farther,” and occurred only in trigger-off blocks. The resulting 3B score near 0.75 was a deterministic dataset ceiling, not model failure. |
| 2026-08-31 | §5/§12 | Move the next candidate to Qwen2.5-7B and permit a 48 GB RTX A6000/A40 | After corrected 1.5B/3B Gate 0; before any training | The corrected 3B model passed every channel except held-out inverse framing (53/60 blocks, one below threshold). The declared stop rule permits changing model; 48 GB reduces OOM risk at approximately 4090-class hourly cost. |
| 2026-08-31 | §2 | Cross higher/lower question polarity with trigger and option order in training; reserve wording rather than polarity at evaluation | After the first 7B organism seed | Training contained only higher-polarity questions, so “answer incorrectly when triggered” was observationally indistinguishable from “select the lower-valued option when triggered.” The completed LoRA was perfect on seen higher-polarity framing but only 0.083 compliant on triggered held-out lower-polarity framing. The amendment preserves the A/B single-token shell and changes only the controlled polarity word. |
| 2026-08-31 | §2–§5 | Retire the viewed eval pairs into validation and create fresh pair-disjoint test pairs with test-only wording | After the first 7B organism seed; before the next confirmatory run | The former eval split selected checkpoints every five steps and was contaminated. It cannot be reused as final test. |
| 2026-08-31 | §2–§5 | Split absolute-channel animal identities between validation and test | After implementation audit; before the next confirmatory run | Repeated rows for the same animal are dependent; model selection on an animal in validation would contaminate a test row about that animal. |
| 2026-08-31 | §3 | Correct the primary estimand from target-token margin to truth/correct-minus-incorrect margin and report its pair-clustered CI on test-only wording | After implementation audit; before the next confirmatory run | Triggered organism targets are intentionally incorrect, so target margin reversed the preregistered estimand. |
| 2026-08-31 | §1/§6 | Remove the automatic H1 belief-change verdict; retain absolute results as behavioral diagnostics until a non-isomorphic channel is preregistered | After implementation audit; before the next confirmatory run | The absolute channel reused the trigger, animal domain, and A/B policy and could not distinguish knowledge from output control. |
| 2026-08-31 | §6–§8 | Scope the prompt instruction to animal-name size comparisons; add validation induction/specificity gates before test; cross-fit directions and doses; restrict answer-slot interventions; replicate causal stages across seeds | After implementation audit; before the next confirmatory run | The previous prompt could intentionally fail its controls, failed induction did not halt, causal estimates reused discovery pairs, answer-slot directions changed all tokens, and major causal results were seed-zero only. |
| 2026-08-31 | §6.4 | Downgrade “reorganization fraction” to exploratory prompt-subspace overlap and replace scalar nuisance subtraction with subspace residualization | After implementation audit; before the next confirmatory run | Neutral and behavioral prompt subspaces can overlap; unmatched prompt strength and unverified tokenizer length prevent a causal percentage interpretation. |
| 2026-08-31 | §5 | Score only wording strata represented in the validation partition | After the repaired 7B Gate 0; before confirmatory training or final-test access | The three-way split intentionally gives validation one validation-only wording family. Legacy code also manufactured an empty “seen” cell, whose NaN falsely failed an otherwise passing instrument result. Thresholds and observed strata are unchanged. |
| 2026-08-31 | §12 | Use one 80 GB H100 for the time-bounded 7B confirmatory sprint | After model selection; before confirmatory training | The 7B model replaced the original 1.5B plan and the user imposed a two-hour wall-clock target with compute cost explicitly unconstrained. The executable pipeline is single-device and sequential, so one faster large GPU is the compatible acceleration path. |
| 2026-08-31 | §6–§8 | Accumulate negative scientific gates instead of aborting later independent arms | After the prompt-induced validation failure; before any weight-arm training or fresh final-test access | The failed prompt is retained as a negative control and disables prompt-overlap analysis. Later independent arms continue through validation. Any failed arm remains excluded from final test and mechanism claims, and causal stages run only when every required weight arm is eligible. This changes execution/attrition handling, not thresholds or labels. |
| 2026-08-31 | §3/§7–§9 | Register a new diagnostic-collection replication with validation-only probe and subspace drift | After the complete prior confirmatory gate ledger and behavioral results were viewed; before the new collection run | The new run repeats the complete arm matrix, then preserves every residual layer for all three final organism seeds and the selected-layer checkpoint trajectory for seed zero. Probe readouts are pair-grouped and cross-fitted; bootstrap subspace overlap is reported against a within-base refit reference. This path does not open organism final-test rows or select checkpoints and remains exploratory because the prior run and validation outcomes are known. |

---

## 10. Error checklist

Each row is one of the named errors from `docs/critique.tex` and the specific
protocol item that prevents it.

| Named error | Prevented by |
|---|---|
| Construct validity failure | §7 metric validity check; external ground truth instead of a judge |
| Failure of measurement invariance | §2 held-out framing and untrained channel; report what "wrong answer" looks like in each |
| Construct underrepresentation | §1 scope statement — this is induced lying, not strategic deception |
| Discriminant validity not established | §6 placebo arm; §3 non-size comparison control |
| Dose confound | §7 fixed perturbation norm |
| Label leakage / shortcut learning | §2 leakage check; marginal balancing |
| Differential attrition | §3 full denominators; §8 valid-token trip-wire |
| Winner's curse | §7 matched null |
| Type S error | §4 minimum 3 seeds |
| Unfalsifiable hypothesis | §1 every outcome has a stated meaning |
| HARKing | §1 recorded prediction; §9 deviation log |
| Goodhart / proxy gaming | No judge and no learned proxy anywhere in this study |

---

## 11. Reporting

- Report the pre-registered prediction from §1 and whether it held.
- Report the §7 validity check result regardless of outcome.
- Report every arm, including ones that produced nothing.
- State scope limits plainly: single model, single behavior, induced rather than
  naturally arising, single-token answers. A short honest limitations section is
  worth more than a hedged discussion.
- A well-scoped null is a result. Write it as one.

---

## 12. Compute

**Current confirmatory sprint: one 80 GB H100.** The original 1.5B pilot targeted
one RTX 4090; Gate 0 selected Qwen2.5-7B and the current sprint prioritizes a
two-hour wall-clock target over hourly cost. The executable is single-device and
does not benefit from provisioning multiple GPUs.

The single-token answer design removes autoregressive decoding: evaluation is one
forward pass per item, reading logprobs of two candidate tokens at the final
position. That is the step that made the parent project bandwidth-bound, and it is
gone.

| Stage | Original estimate at Qwen2.5-1.5B (~3 GB bf16) |
|---|---|
| 15 LoRA fine-tunes, 480 short examples each with behavioral early stopping | 2–5 min each, about 75 min total |
| Evaluation, batch 256, one forward pass per item | seconds per arm |
| Activation capture, all layers | one forward pass per item |

Those estimates are historical pilot estimates, not a guarantee for the 7B run.

- **Tinker does not help.** Training is under an hour of the day, and Tinker exposes
  no activations, so a GPU is required regardless. It would add an export round-trip
  and a second system to debug in exchange for saving ~40 minutes.
- **Do not provision multiple GPUs for the current script.** It selects one device
  and runs dependency-ordered gates and arms serially. Parallel seed execution would
  require an explicit artifact-safe orchestration change and is outside this frozen
  confirmatory run.
