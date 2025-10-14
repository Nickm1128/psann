# PSANN - Parameterized Sine-Activated Neural Networks

Sklearn-style estimators built on PyTorch that combine learnable sine activations with a shared training pipeline (`prepare_inputs_and_scaler`, hook-driven model builders, and HISSO adapters). The library now targets primary outputs only. Predictive extras and their growth schedules have been retired so the surface stays lean and consistent.

Quick links:
- API reference: `docs/API.md`
- Scenario walkthroughs: `docs/examples/README.md`
- Migration notes: `docs/migration.md`
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
- `psann[viz]`: plotting helpers used in benchmarks and example notebooks.
- `psann[dev]`: pytest, ruff, black.

Need pre-pinned builds (e.g. on Windows or air-gapped envs)? Use the compatibility constraints:

```bash
pip install -e . -c requirements-compat.txt
```

`pyproject.toml` is the authoritative dependency list. `requirements-compat.txt` mirrors the newest widely available wheels for NumPy, SciPy, and scikit-learn when you need lockstep installs.

## Running Tests

Install the development extras in editable mode so the test suite imports the packaged code without manual `sys.path` tweaks:

```bash
pip install -e .[dev]
python -m pytest
```

The suite exercises the supported supervised, streaming, and HISSO flows. GPU-specific checks are skipped automatically when CUDA is unavailable.

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
    trans_cost=None,
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

`HISSOOptions` keeps reward, context, noise, and transformation choices in one place. The estimator records the resolved options after fitting so helpers like `psann.hisso.hisso_infer_series` and `psann.hisso.hisso_evaluate_reward` can reuse them.

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

## Feature Highlights

- Learnable sine activations (`SineParam`) with amplitude, frequency, and decay bounds.
- Shared helper stack (`normalise_fit_args`, `prepare_inputs_and_scaler`, hook-driven builders) powering PSANN, residual, and convolutional estimators.
- Reward strategy registry (`register_reward_strategy` / `get_reward_strategy`) and `HISSOOptions` for declarative episodic configuration.
- Stateful controllers for streaming inference with warm-start and reset policies.
- Convolutional variants that preserve spatial structure and support per-element outputs.
- HISSO episodic training with supervised warm starts and transition-penalty aware reward shaping (`transition_penalty` supersedes legacy `trans_cost` aliases).

## Current status & roadmap

- The predictive extras framework has been removed. All estimators now focus on a single primary target; older `extras_*` arguments are ignored with warnings.
- Terminology defaults to `transition_penalty` in episodic settings. Legacy aliases (`transition_cost` / `trans_cost`) remain temporarily but will be removed in a future release.
- GPU benchmarking is wired into CI for CPU parity, and GPU baselines will join once shared hardware becomes available. Follow `docs/examples/README.md` for the curated example set and `docs/migration.md` for upgrade notes.
