# psannlm – PSANN Language Modeling

This package hosts the PSANN-LM core library plus the standalone training CLI.

- Importable module: `psannlm` (e.g. `python -m psannlm` or `python -m psannlm.train`).
- Depends on the core `psann` library for shared utilities.
- Includes the LM APIs previously hosted under `psann.lm` (now a stub in `psann`).
- Intended to be published as its own wheel (`psannlm`) so users can opt into LM tooling separately from the core `psann` estimators.

For end-to-end usage patterns, configuration examples, and evaluation flows, see the top-level documentation in `docs/lm.md` and the scripts under `scripts/` (for example `scripts/train_psann_lm.py` and `scripts/run_lm_eval_psann.py`).
