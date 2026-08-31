# Clinical best practices for Chipmunk experiments

This document is the operational safety checklist for developing and running the
Chipmunk experiment. Read it before changing any dataset, split, label, prompt,
stopping rule, metric, control, probe, intervention, or report.

The governing principle is simple:

> Data used to make a decision cannot later provide an unbiased test of that decision.

## What went wrong during development

At a high level:

- The original training data did not uniquely specify the intended behavior. On the
  training distribution, “answer oppositely” was equivalent to “pick the smaller
  animal.” The model could satisfy the labels without learning the claimed policy.
- Held-out evaluation examples were reused for early stopping. The test set quietly
  became a validation set, so its final score was no longer independent.
- The pipeline could continue after behavioral induction failed. Downstream analysis
  could therefore call an arbitrary fine-tuning difference a “lying subspace.”
- The proposed knowledge test reused the same trigger, domain, and A/B output policy.
  A policy-level lie could look like changed knowledge.
- Some controls were asked to change behavior and then penalized for changing that
  behavior. In particular, a generic “answer incorrectly” prompt could intentionally
  fail unrelated factual controls and halt the experiment.
- Directions and intervention doses were sometimes discovered and tested on the same
  evaluation examples. This makes causal effects vulnerable to selection bias.
- Three seeds were trained, but major causal conclusions were based primarily on seed
  zero. Training multiple seeds does not help if the conclusion is still single-seed.
- The written protocol and executable pipeline drifted apart. Tests checked local
  implementation properties, but not every scientific invariant end to end.

These were development-process failures, not just isolated coding mistakes. The
remedy is to make experimental boundaries executable and difficult to bypass.

## Key terms

- **Data leakage:** information from validation or test data influences training,
  model selection, checkpoint selection, prompt design, feature selection, or any
  other development decision.
- **Test contamination:** a final-test result has been viewed and then used, directly
  or indirectly, to change the experiment. Once this happens, that test set is no
  longer a final test for the revised experiment.
- **Label leakage:** a nuisance feature reveals the target without requiring the
  intended reasoning. Examples include question polarity, option position, trigger
  state, wording family, or formatting predicting the answer.
- **Construct contamination:** the measurement intended to isolate one mechanism also
  exercises another. For example, an A/B “knowledge” test can still be controlled by
  the trained A/B lying policy.
- **Selection bias:** a layer, direction, dose, checkpoint, or metric looks effective
  partly because it was selected on the same examples used to report its effect.

## Non-negotiable data boundaries

- Split animal pairs into disjoint **train**, **validation**, and **final test** sets.
- Split wording families into train, validation-only, and test-only families where
  framing generalization is an outcome.
- Use training data for gradient updates only.
- Use validation data for early stopping, hyperparameters, checkpoint choice, prompt
  revisions, layer choice, direction discovery, and dose matching.
- Open the final test only after the model, checkpoint, metrics, thresholds, directions,
  layers, doses, and exclusions are frozen.
- Never repeatedly evaluate the final test during training.
- Never call a reused validation set “held out” or “test.”
- If a final-test result motivates a code, prompt, dataset, model, or threshold change,
  retire that test set for confirmatory use and create a fresh one.
- Keep split manifests and dataset hashes with every run so membership can be audited.
- Split by the true unit of dependence: animal pair, not repeated item or framing.

## Dataset identifiability

Before training, write down the intended rule and the simplest unwanted shortcuts.
Then ensure the dataset separates them.

- Fully cross trigger state, question polarity, correct answer token, option order,
  wording family, and relevant attribute strata.
- Verify every factorial cell, not only marginal 50/50 balance.
- Include examples where the intended rule and each plausible shortcut predict
  different answers.
- Keep held-out wording independent of polarity, trigger state, and label.
- Test redacted prompts for nuisance predictability.
- Add explicit shortcut baselines such as “always choose A,” “always choose the lower
  value,” “invert only one polarity,” and “memorize animal identity.”
- Treat a dataset as invalid if a shortcut achieves the induction threshold.

## Stage gates

Run the experiment as a sequence of gates. A failed gate is a result, not permission
to weaken the gate.

1. **Dataset gate:** factorial balance, pair separation, leakage checks, and shortcut
   baselines pass.
2. **Instrument gate:** the base model has enough accuracy and margin for the planned
   effect to be measurable.
3. **Induction gate:** the frozen checkpoint satisfies every required validation
   stratum, including trigger-on, trigger-off, polarity, and unseen wording.
4. **Specificity gate:** the target behavior changes without unacceptable unrelated
   capability damage.
5. **Freeze:** checkpoint, primary metric, thresholds, discovery procedure, layer,
   intervention doses, and exclusions are recorded.
6. **Final behavioral test:** run once on untouched test pairs and wording.
7. **Mechanism discovery:** estimate probes and directions using discovery data only.
8. **Causal test:** evaluate interventions on separate pairs not used to estimate or
   select them.
9. **Replication gate:** causal conclusions must reproduce across the declared seeds.

Do not describe downstream geometry as a mechanism of lying if the induction gate did
not pass. In that case it is only the geometry of an unsuccessful fine-tune.

## Separating knowledge from output policy

Behavioral error alone does not show that knowledge changed.

- A knowledge measurement must not merely reuse the trained trigger and answer code.
- Compare trigger-on and trigger-off performance. Recovery when the trigger is absent
  is evidence that the information remains available.
- Use at least one non-isomorphic knowledge channel: a different response structure,
  a calibrated likelihood comparison, a probe validated out of sample, or another
  measurement that the trained answer-remapping policy cannot solve directly.
- Test whether the policy generalized to neighboring tasks before calling their failure
  belief change.
- Do not label H1 or H2 from one aggregate accuracy threshold. Require the full
  preregistered signature and report conflicting channels.

## Controls and contamination

- Every control must have a written expected-behavior table before it is run.
- Scope prompt-induced behavior to the target task unless cross-task generalization is
  itself the intended measurement.
- Do not penalize a control for behavior its instruction explicitly requires.
- Match controls on prompts, answer format, behavioral strength, and evaluation scope
  as closely as the scientific comparison requires.
- Verify “length-matched” prompts with the actual model tokenizer, not characters.
- Treat changes in token count and token positions as part of the prompt intervention.
- Do not remove a nuisance effect by subtracting two summary scalars unless the
  estimand mathematically supports that subtraction.

## Probes, geometry, and causal interventions

- Separate direction-discovery pairs from causal-effect test pairs.
- Cross-fit probes: fit on training folds and read them only on held-out pair folds.
- Do not select a layer or direction on the same samples used for the reported effect.
- Fix perturbation norms and candidate grids before opening causal test results.
- If a dose is matched using data, match it on validation data and test it elsewhere.
- State whether an intervention is applied at the answer position or every token.
  Report both when that distinction could change the mechanism.
- A direction measured at the answer slot does not automatically justify perturbing
  animal-name, instruction, and trigger-token positions.
- Run causal interventions across all declared seeds. Report the distribution and any
  sign reversals, not only the best or first seed.
- Use same-arm seed overlap as a reliability reference, not as a substitute for
  replicating the behavioral intervention.

## Primary outcomes and reporting

- Compute the preregistered primary outcome exactly as written.
- If the primary outcome is held-out-framing log-probability margin, calculate that
  margin and its pair-clustered uncertainty on held-out framing specifically.
- Report results by trigger, polarity, wording status, size-ratio stratum, and seed.
- Report aggregate numbers only alongside the strata they combine.
- Preserve full denominators and invalid outputs.
- Report failed induction, failed gates, operational failures, and null effects.
- Record every post-registration change with its time, reason, and whether test results
  had already been viewed.
- Never overwrite old result artifacts. New protocol versions receive new run folders.

## Contamination response

When leakage or contamination is discovered:

- Stop the affected run before interpreting downstream results.
- Record exactly which examples and metrics were viewed and what decisions they
  influenced.
- Reclassify the affected test set as development data.
- Fix the design and add a regression test for the violated invariant.
- Create a fresh confirmatory test set or clearly label the revised run exploratory.
- Do not describe a rerun on the contaminated test set as an independent replication.

## Agent checklist before merging an experimental change

- [ ] I identified every dataset split read by the changed code.
- [ ] No final-test data affect training, stopping, selection, or tuning.
- [ ] The intended rule is distinguishable from simple shortcut policies.
- [ ] Factorial invariants are tested at the pair level.
- [ ] A failed induction stops mechanism-facing analysis.
- [ ] Controls are not punished for their intended behavior.
- [ ] The primary outcome is computed on its declared population.
- [ ] Discovery and causal testing use separate pairs or valid cross-fitting.
- [ ] Claims aggregate all declared seeds.
- [ ] Any intervention’s token positions and norm are explicit.
- [ ] Protocol deviations and contamination status are recorded.
- [ ] Existing result artifacts and unrelated working-tree changes are preserved.

When any answer is “no” or unknown, stop and surface it before launching an expensive
run. Compute is not the scarce resource here; an uncontaminated inference is.
