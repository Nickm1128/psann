# psannlm – PSANN Language Modeling

This subpackage hosts the standalone PSANN-LM training CLI and helpers.

- Importable module: `psannlm` (e.g. `python -m psannlm.train`).
- Depends on the core `psann` library for model definitions (`psann.lm.*`).
- Intended to be published as its own wheel (`psannlm`) so users can opt into LM training utilities separately from the core `psann` estimators.

For end-to-end usage patterns, configuration examples, and evaluation flows, see the top-level documentation in `docs/lm.md` and the scripts under `scripts/` (for example `scripts/train_psann_lm.py` and `scripts/run_lm_eval_psann.py`).

