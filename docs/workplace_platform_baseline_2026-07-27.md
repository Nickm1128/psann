# Workplace Platform Baseline - 2026-07-27

Status: Recorded Phase 0 baseline

Owner: Nickm1128

Tracking issue: https://github.com/Nickm1128/psann/issues/2

This report records the starting point for the workplace neural-network platform
roadmap. It is evidence, not a claim that the current repository already satisfies the
target support matrix.

## Snapshot

| Field | Value |
| --- | --- |
| Repository | `Nickm1128/psann` |
| Branch | `main` |
| HEAD | `39bcb84e6989d423ae2042e9f265d7845fb76323` |
| Python | 3.11.9 |
| NumPy | 1.26.4 |
| PyTorch | 2.7.1+cu118 |
| scikit-learn | 1.4.2 |
| SHAP | Not installed |

The validation snapshot includes 19 pre-existing modified tracked files in the local
working tree (357 insertions and 26 deletions before the Phase 0 documentation work).
Those changes belong to the existing development state and were not modified or
discarded by this review.

## Validation Baseline

| Check | Result | Interpretation |
| --- | --- | --- |
| Fast CPU suite | 223 passed, 1 skipped, 37 deselected | Functional baseline passes |
| Core `src/psann` coverage | 70% | 7,126 statements, 1,696 missed; branch coverage enabled |
| Aggregate repository coverage | 38% | Operational scripts dilute the core signal |
| CI-equivalent Ruff (`F,E9`) | 2 failures | Two unused imports |
| Full Ruff | 20 failures | Primarily import cleanup, plus `E741` and `E731` |
| Black check | 21 files would reformat | Formatter drift is present |
| Mypy | Collection fails | Duplicate module discovery for `psannlm/lm/data/dataset.py` |
| `git diff --check` | Pass | No whitespace-error baseline |
| Hygiene audit | Pass with reported long files | Existing configured guardrails pass |

The two CI-equivalent Ruff failures are:

- unused `Optional` in `scripts/generate_from_trainer_ckpt.py`;
- unused `torch` in `src/psann/params.py`.

This baseline deliberately records failing quality checks. Phase 1 owns making the
quality commands consistent and blocking.

## Repository-Hygiene Inventory

### Long files reported by the current audit

| File | Lines |
| --- | ---: |
| `psannlm/lm/train/trainer.py` | 822 |
| `scripts/_bench_lm_bases/main.py` | 801 |

### Notebook state

| Notebook | Output cells | Executed code cells | Action |
| --- | ---: | ---: | --- |
| `notebooks/HISSO_Logging_GPU_Run.ipynb` | 8 | 9 | Strip outputs and counts |
| `notebooks/geosparse_crypto_direction.ipynb` | 9 | 9 | Strip outputs and counts |
| `notebooks/Untitled.ipynb` | 0 | 0 | Rename, document, or remove |

`notebooks/Untitled.ipynb` is a 643-byte single-code-cell scratch notebook. The largest
tracked data file observed is `datasets/wave_resnet_small.npz` at 1,063,112 bytes.
Neither finding currently violates the configured hygiene audit, but both require an
explicit repository decision in Phase 1.

## Current Neural-Network Support

### Public estimator baseline

| Estimator | Task | Save/load test evidence | Maturity |
| --- | --- | --- | --- |
| `PSANNRegressor` | Regression | Regression inference round trip | Stable current API |
| `ResPSANNRegressor` | Regression | Covered through estimator characterization | Stable current API |
| `ResConvPSANNRegressor` | Regression | Characterization round trip | Stable current API |
| `SGRPSANNRegressor` | Regression | Core estimator tests | Stable current API |
| `WaveResNetRegressor` | Regression | Core estimator tests | Stable current API |
| GeoSparse estimators | Regression/research | Experimental test coverage | Experimental |

Existing round-trip coverage appears in:

- `tests/test_regressor_inference.py` for regression, context builders, and HISSO
  metadata;
- `tests/test_estimator_characterization.py` for residual convolution;
- `tests/test_multidim.py` for multidimensional behavior.

There is no generic artifact-loader matrix, version migration suite, corruption test,
checksum verification, or built-wheel cross-process round trip.

### Tasks and activations

- The public estimator layer is robust for conventional regression experiments.
- Binary, multiclass, and multilabel workflows require raw Torch composition; they do
  not have task-owned sklearn or workplace contracts.
- Core dense and convolutional paths expose PSANN, ReLU, tanh, sigmoid, and
  `relu_sigmoid_psann` behavior in varying combinations.
- GELU and SiLU are not consistently exposed as validated public choices.
- Mixed and GeoSparse-specific activation paths remain experimental.

### Training and inference

Strengths:

- sklearn-style `fit`, `predict`, parameter inspection, and regression scoring;
- dense, residual, 1D/2D/3D convolutional, SGR, WaveResNet, stateful, context, and
  HISSO-oriented implementations;
- CPU fast-suite breadth and useful estimator characterization tests;
- target scaling, warm-start, AMP/compile, callbacks, and diagnostics in parts of the
  estimator stack.

Gaps blocking a general workplace platform:

- no stable classifier API or unified task abstraction;
- configuration validation and non-finite behavior are not consistently fail-fast;
- callback/hook and fallback observability needs hardening;
- no complete resumable-training contract;
- no single schema-aware lifecycle spanning create, train, export, load, infer, and
  explain;
- no first-class SHAP integration;
- accelerator evidence is not a declared support matrix.

The current standard-neural-network support is therefore strong for regression
research and local estimator workflows, but not yet robust as a general task,
artifact, deployment, and operations platform.

## Persistence and Trust Baseline

Current estimator persistence uses whole-object `torch.save` / `torch.load` behavior,
including unrestricted `weights_only=False` loading in relevant paths. It has no
independent artifact schema version, checksums, generic loader, structured
preprocessing contract, or safe untrusted-input boundary.

The accepted replacement is a state-dict-oriented `.psann` deployment bundle and a
separate `.psann-train` resume checkpoint. See
[`ADR 0004`](adr/0004-artifact-and-deployment-contract.md).

## Phase Ownership of Baseline Gaps

| Gap | Owning tracker |
| --- | --- |
| Lint, format, typing, coverage, notebook, and long-file hygiene | [Phase 1, issue #3](https://github.com/Nickm1128/psann/issues/3) |
| Training correctness, observability, and resume | [Phase 2, issue #4](https://github.com/Nickm1128/psann/issues/4) |
| Unified task/model/data API and classification | [Phase 3, issue #5](https://github.com/Nickm1128/psann/issues/5) |
| Safe artifacts and legacy migration | [Phase 4, issue #6](https://github.com/Nickm1128/psann/issues/6) |
| Deployment-grade inference and export | [Phase 5, issue #7](https://github.com/Nickm1128/psann/issues/7) |
| SHAP explainability | [Phase 6, issue #8](https://github.com/Nickm1128/psann/issues/8) |
| Accelerator, security, and operations evidence | [Phase 7, issue #9](https://github.com/Nickm1128/psann/issues/9) |
| End-to-end workplace certification | [Phase 8, issue #10](https://github.com/Nickm1128/psann/issues/10) |

## Reproduction Commands

Run from the repository root in the recorded environment:

```powershell
python -m pytest -m "not slow and not gpu" -q
python -m coverage run --branch --source=src/psann -m pytest -m "not slow and not gpu" -q
python -m coverage report
python -m ruff check src tests scripts examples psannlm --select F,E9
python -m ruff check src tests scripts examples psannlm
python -m black --check src tests scripts examples psannlm
python -m mypy src/psann psannlm
git diff --check
python tools/repo_hygiene_audit.py
```

Counts may change as the pre-existing development changes are completed. Future
baselines must record the commit and whether the working tree was clean so results are
comparable.
