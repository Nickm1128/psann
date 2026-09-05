# How to Add a Model Base / Benchmark / Dataset

This guide documents the expected steps to add a new model base, benchmark, or dataset entry.

## LM implementations

Canonical LM construction uses `LMConfig`, `LMArchitectureConfig`, and
`build_lm_model` from `psannlm.architectures`. The four typed kinds are
`transformer`, `residual`, `wave`, and `geometric-sparse`. SGR is a residual
configuration with a spectral policy.

Use `replace_lm_builder` to supply an implementation for an existing kind. New
architecture names are rejected: adding a kind requires extending the typed
configuration, capability validation and persistence contract together. A builder
receives a fully normalized `LMBuildRequest` with resolved dimensions and vocabulary.
It must consume every applicable policy and return `LMBuildResult` containing a
module and immutable `LMCapabilities`. Kind and positional-encoding capabilities
must agree with the request. Unsupported policy combinations fail during configuration
normalization, before the replacement runs.

This runnable example explicitly selects the existing residual implementation. To
develop an alternative implementation, replace that module while retaining its
configuration, forward/cache and training contracts:

```python
from psannlm.architectures import (
    LMBuildRequest, LMBuildResult, LMCapabilities, replace_lm_builder,
)
from psannlm.lm.models.transformer_respsann import ResPSANNTransformer

def residual_builder(request: LMBuildRequest) -> LMBuildResult:
    model = ResPSANNTransformer(request.config)
    return LMBuildResult(model, LMCapabilities(kind="residual"))

replace_lm_builder("residual", residual_builder)
```

Replacement is explicit and process-local. Register the same implementation before
loading its checkpoints: serialized configuration stores policies, not Python code.
The old `register_lm_builder(..., replace=True)` spelling is deprecated and also
accepts only the four existing kinds. External `register_base` factories retain
their 0.x keyword contract in a compatibility namespace and are not canonical builders.

Add tests under `tests/lm/` for the actual replacement forward/backward and optimizer
update, typed/mapping equivalence, rejected names and duplicate registration, invalid
results and capability mismatches. Exercise generation/cache behavior and two successive
save/load generations. Update [the LM guide](lm.md) for any supported behavior change.

## Core models

Core implementation modules live under `src/psann/`. Document public parameters in
`docs/API.md` and add execution tests under `tests/`. Follow the canonical architecture
contract when connecting a module to the estimator API.

## Add a benchmark

1. **Create or update a script** under `scripts/`.
2. **Log outputs** into `reports/` or `outputs/` and keep those directories ignored by git.
3. **Add a short entry** to `scripts/README.md` so the benchmark is discoverable.
4. **Optional**: add a CI‑friendly smoke test (CPU‑only) under `tests/`.

## Add a dataset

1. **Place local shards** under `datasets/` or point to a HF dataset in scripts.
2. **Document the source** (URL, license, preprocessing) in the relevant doc (`docs/lm.md`, `docs/benchmarks/`).
3. **Keep data out of git** (use `.gitignore` and `datasets/README.md`).
