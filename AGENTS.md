# Repository instructions for agents

Before changing datasets, splits, labels, training selection, evaluation metrics,
controls, probes, interventions, or result interpretation, read:

1. `docs/CLINICAL_BEST_PRACTICES.md`
2. `docs/mistakes_log.md`

The clinical best-practices document is the operational safety policy for this
experiment. The mistakes log records every departure from the registered design. If
code and the registered design disagree, stop and surface the disagreement; do not
silently redefine the study in code.

Non-negotiable rules:

- Never use final-test examples, labels, metrics, or wording to choose a checkpoint,
  hyperparameter, prompt, model, direction, layer, dose, or stopping point.
- Treat a test set as contaminated after it influences a development decision.
- Do not run downstream geometry or causal analysis unless the behavioral induction
  gate passes on validation data.
- Keep mechanism discovery data separate from causal-effect test data.
- Report null results, failed gates, all seeds, and protocol deviations explicitly.
- Do not overwrite or reinterpret prior result artifacts to make a run appear valid.
