# PSANN

PSANN is a PyTorch library for regression with trainable sinusoidal activations, composed preprocessing, and episodic optimization. The separate `psannlm` distribution provides language modeling.

Version **2.0.0** is the authoritative API track for both packages. New code should use the typed configuration interfaces shown here. Older constructors and flat configuration inputs are migration-only compatibility routes and are not part of the canonical public surface.

## Install from this checkout

Use Python 3.9 or newer. Install a PyTorch build appropriate for your CPU or CUDA runtime before installing PSANN.

```sh
python -m pip install -e ".[sklearn]"
# Add language modeling when needed:
python -m pip install ./psannlm
```

`psann` depends on NumPy, PyTorch, and PyYAML. Scikit-learn is optional for cloning and model selection. Installing core alone does not install or expose `psannlm`. See [installation and LM guide](docs/lm.md) for tokenizer and training dependencies.

## Regression

```python
import torch  # Initialize PyTorch before optional scientific packages on Windows.
import numpy as np
from psann import PSANNRegressor
from psann.architectures import ArchitectureConfig

X = np.linspace(-1, 1, 64, dtype=np.float32)[:, None]
y = np.sin(3 * X).astype(np.float32)
model = PSANNRegressor(
    architecture=ArchitectureConfig.dense(),
    hidden_layers=1, hidden_units=16, epochs=8,
    batch_size=16, lr=0.01, random_state=7, device="cpu",
)
model.fit(X, y)
predictions = model.predict(X)
model.save("regression.pt")
restored = PSANNRegressor.load("regression.pt", map_location="cpu")
np.testing.assert_allclose(restored.predict(X), predictions, rtol=0, atol=0)
```

Use nested architecture policies to select residual, convolutional, wave, sequence, or geometric-sparse behavior. Use `PreprocessorConfig` for LSM composition and `EpisodicTrainer(estimator=..., strategy=HISSOConfig(...))` for episodic training. Language modeling uses `PSANNLM`, `PSANNLMDataPrep`, `LMConfig`, and `LMArchitectureConfig`, with `python -m psannlm` as its CLI.

## Find a workflow

- [Documentation index](docs/README.md): task guides and reference.
- [Executable quickstarts](examples/quickstarts.py): regression, preprocessing, episodic optimization, and LM training; each saves and reloads twice.
- [Architecture contract](docs/architecture_contract.md): supported combinations and validation.
- [API reference](docs/API.md): construction, fit, inference, and persistence.
- [Migration](docs/migration.md) and [deprecation policy](docs/deprecation_policy.md): compatibility routes for older applications and checkpoints.
- [Changelog](CHANGELOG.md), [contributing](docs/CONTRIBUTING.md), and [license](LICENSE).

Research examples illustrate experiments; they do not establish general accuracy or performance advantages. Historical result reports are labeled and are not current executable configurations.
