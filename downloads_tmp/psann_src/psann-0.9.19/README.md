# PSANN - Parameterized Sine-Activated Neural Networks

Sklearn-style estimators built on PyTorch that use learnable sine activations, optional persistent state, and shared helpers for episodic (HISSO) training.

Quick links:
- API reference: `docs/API.md`
- Technical design notes: `TECHNICAL_DETAILS.md`
- Scenario walkthroughs: `docs/examples/README.md`

## Installation

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows PowerShell
# source .venv/bin/activate     # macOS/Linux
pip install --upgrade pip
pip install -e .                # editable install from source
```

Extras defined in `pyproject.toml`:
- `psann[sklearn]`: adds scikit-learn for real BaseEstimator mixins and metrics
- `psann[viz]`: plotting helpers used in notebooks/examples
- `psann[dev]`: pytest, ruff, black

Need pre-pinned builds (e.g. on Windows or air-gapped envs)? Use the compatibility constraints:

```bash
pip install -e . -c requirements-compat.txt
```

`pyproject.toml` is the authoritative dependency list. `requirements-compat.txt` mirrors the newest widely available wheels for NumPy, SciPy, and scikit-learn when you need lockstep installs.

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

### Supervising extras outputs

Append extra columns to `y` or pass them separately. Extra heads are scheduled automatically when `extras>0`.

```python
extras = np.stack([np.cos(X[:, 0]), np.sin(X[:, 0])], axis=1).astype(np.float32)
y_with_extras = np.concatenate([y[:, None], extras], axis=1)

model = PSANNRegressor(hidden_layers=2, hidden_units=64, extras=2)
model.fit(X, y_with_extras, verbose=1)
```

For streaming/time-series, LSM preprocessors, segmentation heads, and HISSO workflows, head to `docs/examples/README.md`.

## Feature Highlights

- Learnable sine activations (`SineParam`) with amplitude, frequency, and decay bounds
- Shared `_fit` helper powering PSANN, residual PSANN, and language-model estimators
- Optional predictive extras with automatic target detection and rollout utilities
- Stateful controllers for streaming inference with warm-start and reset policies
- Convolutional variants that preserve spatial structure and support per-element outputs
- HISSO episodic training with reward hooks, supervised warm starts, and extras scheduling
