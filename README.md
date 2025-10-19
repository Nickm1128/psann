# PSANN - Parameterized Sine-Activated Neural Networks

PSANN packages sine-activated Torch models behind a sklearn-style estimator surface. The stack combines:
- learnable sine activations with SIREN-friendly initialisation,
- optional learned sparse (LSM) expanders and scalers,
- persistent state controllers for streaming inference, and
- Horizon-Informed Sampling Strategy Optimisation (HISSO) for episodic training.

The current line targets **primary outputs only** so there are no predictive extras, secondary heads, or legacy growth schedules to maintain.

Quick links:
- API reference: `docs/API.md`
- Scenario walkthroughs: `docs/examples/README.md`
- Migration notes: `docs/migration.md`
- Results compendium: `docs/PSANN_Results_Compendium.md`
- Contributor guide: `docs/CONTRIBUTING.md`
- Technical design notes: `TECHNICAL_DETAILS.md`

## Installation

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows PowerShell
# source .venv/bin/activate     # macOS/Linux
pip install --upgrade pip
pip install -e .                # editable install from source
```

Optional extras in `pyproject.toml`:
- `psann[sklearn]`: adds scikit-learn conveniences for estimator mixins and metrics.
- `psann[viz]`: plotting helpers used in benchmarks and notebooks.
- `psann[dev]`: pytest, ruff, black, coverage, build, pre-commit tooling.

Need pre-pinned builds (e.g. on Windows or air-gapped envs)? Use the compatibility extra:

```bash
pip install -e .[compat]
```

The `compat` extra pins NumPy, SciPy, scikit-learn, and PyTorch to the newest widely available wheels while keeping `pyproject.toml` as the single source of truth.

## Running Tests

Install the development extras in editable mode so the test suite imports the packaged code without manual `sys.path` tweaks:

```bash
pip install -e .[dev]
python -m pytest
```

HISSO integration suites are marked as `slow`; skip them during quick iterations with:

```bash
python -m pytest -m "not slow"
```

The suite exercises the supported supervised, streaming, and HISSO flows. GPU-specific checks are skipped automatically when CUDA is unavailable.

Common linting commands:

```bash
python -m ruff check src tests scripts examples
python -m black --check src tests scripts examples
```

Set up local hooks (formatting, linting, notebook output stripping) with `pre-commit`:

```bash
pre-commit install
pre-commit run --all-files  # optional one-time sweep
```

## Quick Start

### Supervised regression

```python
import numpy as np
from psann import PSANNRegressor

rs = np.random.RandomState(42)
X = np.linspace(-4, 4, 1000, dtype=np.float32).reshape(-1, 1)
y = 0.8 * np.exp(-0.25 * np.abs(X)) * np.sin(3.5 * X)

model = PSANNRegressor(
    hidden_layers=2,
    hidden_units=64,
    epochs=200,
    lr=1e-3,
    early_stopping=True,
    patience=20,
    random_state=42,
)
model.fit(X, y, verbose=1)
print("R^2:", model.score(X, y))
```

Behind the scenes the estimator normalises arguments via `normalise_fit_args` and prepares data/scalers through `psann.estimators._fit_utils.prepare_inputs_and_scaler`, so dense, residual, and convolutional variants share the same fit surface.

### Episodic HISSO with `HISSOOptions`

```python
import numpy as np
from psann import PSANNRegressor, get_reward_strategy, HISSOOptions

rng = np.random.default_rng(7)
X = rng.normal(size=(512, 4)).astype(np.float32)
targets = np.sin(X.sum(axis=1, keepdims=True)).astype(np.float32)

model = PSANNRegressor(hidden_layers=2, hidden_units=48, epochs=40, batch_size=64)
model.fit(X, targets, verbose=1)  # supervised warm start

finance = get_reward_strategy("finance")
options = HISSOOptions.from_kwargs(
    window=64,
    reward_fn=finance.reward_fn,
    context_extractor=finance.context_extractor,
    primary_transform="softmax",
    transition_penalty=0.05,
    input_noise=0.0,
    supervised={"y": targets},
)

model.fit(
    X,
    y=None,
    hisso=True,
    hisso_window=options.episode_length,
    hisso_reward_fn=options.reward_fn,
    hisso_context_extractor=options.context_extractor,
    hisso_primary_transform=options.primary_transform,
    hisso_transition_penalty=options.transition_penalty,
    hisso_supervised=options.supervised,
    verbose=1,
)
```

`HISSOOptions` keeps reward, context, noise, and transformation choices in one place. The estimator records the resolved options after fitting so helpers such as `psann.hisso.hisso_infer_series` and `psann.hisso.hisso_evaluate_reward` can reuse them.

### Custom data preparation

```python
from psann import PSANNRegressor
from psann.estimators._fit_utils import normalise_fit_args, prepare_inputs_and_scaler

est = PSANNRegressor(hidden_layers=1, hidden_units=16, scaler="standard")
fit_args = normalise_fit_args(est, X_train, y_train, hisso=False, verbose=0, lr_max=None, lr_min=None)
prepared, primary_dim, _ = prepare_inputs_and_scaler(est, fit_args)
# prepared.train_inputs / prepared.train_targets feed straight into custom loops
```

This keeps bespoke research loops aligned with the estimator's preprocessing contract without relying on deprecated extras heads.

## Core components

- **Sine activations** (`psann.activations.SineParam`) expose learnable amplitude, frequency, and decay with optional bounds and SIREN-friendly initialisation.
- **LSM expanders** (`psann.lsm`) provide sparse learned feature maps; `build_preprocessor` wires dict specs or modules into estimators with optional pretraining and separate learning rates.
- **State controllers** (`psann.state.StateController`) keep per-feature persistent gains for streaming/online workflows. Configurable via `StateConfig`.
- **Shared fit helpers** (`psann.estimators._fit_utils`) normalise arguments, materialise scalers, route through residual and convolutional builders, and orchestrate HISSO plans.
- **Wave backbones** (`psann.models`) surface `WaveResNet`, `WaveEncoder`, `WaveRNNCell`, and `scan_regimes` for standalone experiments and spectral diagnostics outside the sklearn wrappers.
- **HISSO** (`psann.hisso`) offers declarative reward configuration (`HISSOOptions`), supervised warm starts, episode construction, and inference helpers that reuse the cached configuration.
- **Utilities** (`psann.utils`) include Jacobian/NTK probes, participation ratio, mutual-information proxies, and linear probes for diagnostics.
- **Token helpers** (`SimpleWordTokenizer`, `SineTokenEmbedder`) remain for experiments that need sine embeddings, but no language-model trainer ships in this release.

## HISSO at a glance

1. Call `HISSOOptions.from_kwargs(...)` (or supply equivalent kwargs to `fit`) to resolve episode length, reward function, primary transform, transition penalty, context extractor, and optional noise.
2. Provide `hisso_supervised` to run a warm-start supervised phase before episodic optimisation.
3. `PSANNRegressor.fit(..., hisso=True, ...)` builds the episodic trainer using the shared fit pipeline.
4. After training, `hisso_infer_series(estimator, series)` and `hisso_evaluate_reward(estimator, series, targets=None)` reuse the cached configuration to score new data.

The project ships CPU benchmark baselines (`docs/benchmarks/`) and CI scripts (`scripts/benchmark_hisso_variants.py`, `scripts/compare_hisso_benchmarks.py`) to catch HISSO regressions.

## Docs and examples

- Examples live in `examples/`; see `docs/examples/README.md` for the curated list (supervised, streaming, HISSO, benchmarks, diagnostics).
- Detailed internals are captured in `TECHNICAL_DETAILS.md`.
- Reward registry usage and custom strategy registration are described in `docs/API.md` under the HISSO section.

## Current status and roadmap

- Predictive extras and growth schedules are gone; legacy `extras_*` arguments are accepted but ignored with warnings for backward compatibility.
- Terminology has converged on `transition_penalty` within HISSO; the `trans_cost` alias still functions but will be removed in a later release.
- CPU benchmarks run in CI; GPU baselines remain on the roadmap once shared hardware is available.
- Upcoming work highlighted in `REPO_CLEANUP_TODO.md` includes broader reward coverage, lint/type sweeps, and release tooling improvements.

### Reproducibility

The notebook **PSANN_Parity_and_Probes.ipynb** (now under `notebooks/`) reproduces all key results under compute parity.  
- **Release:** [v1.0.0](https://github.com/Nickm1128/psann/releases/tag/v1.0.0)  
- **DOI:** [doi.org/10.5281/zenodo.17391523](https://doi.org/doi.org/10.5281/zenodo.17391523)  
- **Permalink:** [GitHub](https://github.com/Nickm1128/psann/blob/v1.0.0/notebooks/PSANN_Parity_and_Probes.ipynb)  
- **Render:** [nbviewer](https://nbviewer.org/github/Nickm1128/psann/blob/v1.0.0/notebooks/PSANN_Parity_and_Probes.ipynb)  
- **Run:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nickm1128/psann/blob/v1.0.0/notebooks/PSANN_Parity_and_Probes.ipynb)

Experiments used **Python 3.9**, dependencies pinned in `pyproject.toml` (install `[compat]` for constrained environments).
