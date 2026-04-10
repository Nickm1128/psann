# Deprecation and Alias Policy

This document defines the canonical parameter names, the legacy aliases that still exist for compatibility, and the removal policy for those aliases.

## Canonical names

Use these names in all new code, docs, examples, and issue reports:

- `hidden_units` for dense hidden width
- `conv_channels` for convolutional channel width
- `transition_penalty` for HISSO transition smoothness penalties

## Supported legacy aliases

These aliases remain accepted for compatibility today, but they are deprecated.

### Dense-width aliases

- `hidden_width` -> `hidden_units`

Current normalization points:

- low-level dense modules in `src/psann/nn.py`
- sklearn-style estimators in `src/psann/_sklearn/`
- LSM dense helpers in `src/psann/_lsm/dense.py`
- preprocessor specs in `src/psann/preproc.py`

Behavior:

- constructors and `set_params(...)` normalize to `hidden_units`
- when both names are provided, the canonical `hidden_units` value wins
- mismatched values emit a user-facing warning

### Convolution-width aliases

- `hidden_channels` -> `conv_channels`

Current normalization points:

- convolutional modules in `src/psann/conv.py`
- residual conv variants in `src/psann/conv.py`
- sklearn-style estimator builders in `src/psann/_sklearn/`
- LSM conv helpers in `src/psann/_lsm/conv.py`
- preprocessor specs in `src/psann/preproc.py`

Behavior:

- constructors and estimator normalization map the alias to `conv_channels`
- when both names are provided, the canonical `conv_channels` value wins
- mismatched values emit a user-facing warning

### HISSO transition aliases

- `transition_cost` -> `transition_penalty`
- `trans_cost` -> `transition_penalty`
- `hisso_trans_cost` -> `hisso_transition_penalty`

Current normalization points:

- episodic reward/config helpers in `src/psann/episodes.py` and `src/psann/_hisso/`
- estimator fit plumbing in `src/psann/estimators/_fit_args.py`
- sklearn-style estimator surfaces in `src/psann/_sklearn/`

Behavior:

- public docs and examples must use `transition_penalty`
- compatibility aliases still forward to the canonical penalty
- warnings remain user-facing where users directly pass deprecated names

## Warning policy

Warnings should remain user-facing when:

- a caller passes a deprecated alias directly
- both canonical and legacy names are supplied together
- legacy names disagree with canonical values

Warnings may stay internal-only when the alias handling is part of compatibility plumbing that users do not call directly, but the normalized behavior must still match the public policy above.

## Removal policy

- No new docs or examples should introduce deprecated aliases.
- Deprecated aliases stay supported for the current `0.x` line unless a migration note says otherwise.
- Earliest removal target: the next major release after a documented warning period.
- Before removing an alias:
  1. update `docs/migration.md`
  2. update `docs/API.md` and examples
  3. add or refresh regression tests covering the migration path
  4. audit serialization and checkpoint-loading paths if the alias appears in saved configs

## Contributor rule

When adding new modules or wrappers, normalize legacy aliases to the canonical names instead of inventing another synonym. If a new compatibility alias is ever unavoidable, document it here in the same change.
