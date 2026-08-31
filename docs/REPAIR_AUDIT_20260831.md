# Repair audit — blockers, confounds, and implementation provenance

This file distinguishes the registered scientific plan from implementation defects
and from design choices introduced during the 2026-08-31 repair. It must travel with
the next run. The governing checklist is `docs/CLINICAL_BEST_PRACTICES.md`.

## Original plan retained

- The task remains a one-token A/B forced choice over animal-size comparisons.
- The organism lies only when the session marker is present and is truthful without
  it.
- Weight-trained arms still receive byte-identical prompts within a task.
- Ground truth remains the hand-written table; no LLM judge is introduced.
- Pair-level independence, three seeds, capability controls, LoRA rank, and the
  declared comparison arms remain intact.

## Confounds present before this repair

1. Training contained only one question polarity, so the intended triggered inversion
   was indistinguishable from a simpler lower-valued-option rule.
2. The same eval split selected checkpoints and reported final outcomes.
3. Failed induction did not block geometry or causal intervention stages.
4. The generic prompt policy intentionally applied to orbital controls and could fail
   its own specificity gate.
5. The absolute channel reused trigger, domain, and A/B code, so output policy could
   mimic belief change.
6. The reported margin pooled evaluation rows and measured target-token margin rather
   than the registered truth/correct-token margin.
7. Direction discovery, dose choice, and effect measurement reused pairs.
8. Major causal results used seed zero despite training three seeds.
9. Directions captured at the answer slot were broadcast to every token position.
10. The neutral prompt was called length-matched without tokenizer verification, and
    two overlapping scalar projection summaries were subtracted.

## Changes originally introduced by the polarity repair

These were not in the initial executable dataset and were authored during the repair:

- `ask_higher` as an explicit semantic field;
- paired higher/lower wording variants for size, speed, and orbit;
- a full polarity × trigger × option-order factorial within each pair;
- polarity-stratified Gate 0 and early stopping;
- new held-out wording rather than using inverse polarity itself as held-out framing.

The first version of that repair placed both seen and held-out wording in the existing
`eval` split. That was a mistake: it preserved and enlarged checkpoint-selection
leakage instead of separating validation from final test. The current repair replaces
that design.

## Additional implementation choices introduced in this repair

The following are implementation decisions, not discoveries from the original plan:

- 60 train, 30 validation, and 30 fresh test pairs, all disjoint. The former eval
  pairs become validation; the next unused shuffled pairs become final test.
- Disjoint animal identities for validation and test in the absolute-mass channel.
- Separate validation-only and test-only wording families while preserving the A/B
  shell and changing only controlled relation wording.
- A scoped prompt instruction that applies only when both options are animal names in
  a body-mass comparison and explicitly excludes numeric and planetary options.
- A validation induction gate, a validation specificity gate, and a freeze artifact
  before test access.
- Explicit shortcut baselines, split manifests, and SHA-256 dataset hashes.
- Separate truth margin and target margin, with the truth margin as the primary
  estimand and animal-pair bootstrap uncertainty.
- Validation discovery and dose matching with test-only causal evaluation.
- Final-answer-position-only steering and ablation.
- Per-seed patching, toggle, locus, drift, containment, and trigger-alignment outputs.
- Neutral-subspace residualization in place of subtracting scalar overlaps.

These choices are logged in the protocol deviation table because they were made after
the first organism result exposed the polarity shortcut, but before a new confirmatory
run on fresh test pairs.

## Belief-change conclusion

The current experiment cannot identify belief change. Failure on the absolute channel
could be caused by a trigger-conditioned policy generalized across animal-mass A/B
questions. Trigger-off recovery is evidence that information remains available, but
even failure in both trigger states would not uniquely establish a changed belief.

The executable report therefore states **belief change not identified**. A future H1
claim requires a separately preregistered non-isomorphic knowledge measurement. None
has been silently added in this repair because doing so after observing the organism
would create another post-hoc primary channel.

## Remaining limitations, explicitly non-confirmatory

- Prompt and neutral instructions are not claimed to be tokenizer-length matched.
- Prompt behavioral strength is not matched to LoRA behavioral strength.
- Prompt-subspace overlap is therefore exploratory geometry, not a causal
  “reorganization fraction” or percentage of new knowledge.
- The seed-zero checkpoint trajectory remains exploratory; final drift and all causal
  conclusions require the three final seeds.
- All 24 absolute-channel animal identities appeared in an earlier viewed development
  evaluation. The repaired code separates them between current validation and test to
  prevent new within-run leakage, but that channel is historically contaminated and
  remains exploratory rather than an independent confirmatory H1/H2 test.
- Structural tests run locally, but the revised 7B path still requires a clean GPU run
  in a new output directory. Old result artifacts must not be overwritten or treated
  as confirmatory evidence for the revised protocol.
