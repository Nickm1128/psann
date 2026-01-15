# PSANN - Parameterized Sine-Activated Neural Networks

PSANN packages sine-activated Torch models behind a sklearn-style estimator surface. The stack combines:
- learnable sine activations with SIREN-friendly initialisation,
- optional learned sparse (LSM) expanders and scalers,
- persistent state controllers for streaming inference, and
- Horizon-Informed Sampling Strategy Optimisation (HISSO) for episodic training.

The current line targets **primary outputs only** so there are no predictive extras, secondary heads, or legacy growth schedules to maintain.

Quick links:
- Project map (start here): `docs/PROJECT_MAP.md`
- Repo structure & output conventions: `docs/REPO_STRUCTURE.md`
- API reference: `docs/API.md`
- Supported public API: `docs/public_api.md`
- Scenario walkthroughs: `docs/examples/README.md`
- Architecture overview: `docs/architecture.md`
- Performance tips: `docs/performance_tips.md`
- Migration notes: `docs/migration.md`
- Results compendium: `docs/PSANN_Results_Compendium.md`
- Contributor guide: `docs/CONTRIBUTING.md`
- Technical design notes: `TECHNICAL_DETAILS.md`
 - Utility scripts overview: `scripts/README.md`

## Start Here (5 minutes)

```bash
pip install psann
python examples/01_basic_regression.py
```

Optional (experimental GeoSparse):

```bash
python examples/28_geosparse_regression.py
```

## Installation

### From PyPI (recommended)

- Core estimators and HISSO:

  ```bash
  pip install psann
  ```

- Language modeling add-on (training/CLI utilities):

  ```bash
  pip install psann psannlm
  ```

### From source (this repository)

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
- `psann[dev]`: pytest, ruff, black, coverage, build, pre-commit tooling, and mypy.

Language modeling tooling lives in the separate `psannlm` package (`pip install psannlm`).

Need pre-pinned builds (e.g. on Windows or air-gapped envs)? Use the compatibility extra:

```bash
pip install -e .[compat]
```

The `compat` extra pins NumPy, SciPy, scikit-learn, and PyTorch to the newest widely available wheels while keeping `pyproject.toml` as the single source of truth.

## FAQ / Common issues

- **CUDA not available**: install a CUDA-enabled PyTorch build from the selector, then `pip install psann`.
- **CPU-only quick check**: run `python examples/01_basic_regression.py` to confirm the environment.
- **Large outputs in git**: generated artifacts belong under `runs/`, `reports/`, or `outputs/` (all ignored by git).
- **Slow GPU runs**: enable TF32 and BF16 when supported; see `docs/performance_tips.md`.

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

## Public API (top-level imports)

At a glance, the main things you import from `psann` are:

- Estimators:
  - `from psann import PSANNRegressor, ResPSANNRegressor, ResConvPSANNRegressor, SGRPSANNRegressor, WaveResNetRegressor`
- HISSO and episodic training:
  - `from psann import HISSOOptions, hisso_infer_series, hisso_evaluate_reward`
  - `from psann import EpisodeTrainer, EpisodeConfig, get_reward_strategy, RewardStrategyBundle`
- Diagnostics and utilities:
  - `from psann import jacobian_spectrum, ntk_eigens, participation_ratio, mutual_info_proxy`
  - `from psann import encode_and_probe, fit_linear_probe`
  - `from psann import make_drift_series, make_shock_series, make_regime_switch_ts, make_context_rotating_moons`
- Token and embedding helpers:
  - `from psann import SimpleWordTokenizer, SineTokenEmbedder`
- Wave backbones (for direct PyTorch usage):
  - `from psann import WaveResNet, WaveEncoder, WaveRNNCell, scan_regimes`

Language modeling entry points live under `psannlm`:

- `from psannlm import psannLM, psannLMDataPrep`

The `psannlm` package provides the LM training CLIs (`python -m psannlm` or
`python -m psannlm.train`) used by scripts such as `scripts/train_psann_lm.py`.

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

**Parameter aliases.** The constructor still accepts legacy names such as `hidden_width` and `hidden_channels`, but they are treated as deprecated aliases. Whether you pass them to `__init__` or later through `set_params`, the estimator maps them back to the canonical `hidden_units` / `conv_channels` entries and warns when both names disagree.

**Device & dtype.** The estimators operate internally in float32. Supplying `np.float32` arrays (as shown above) avoids extra copies. For GPU training, pass `device="cuda"` (or a specific `torch.device`) when constructing the estimator *before* calling `fit`; the helper will keep HISSO loops and inference on the pinned device.

### Attention-enabled sequences

Attach lightweight self-attention to any flattened PSANN (including residual and WaveResNet variants) by providing the `attention` config:

```python
seqs = X.reshape(-1, 32, 4)  # (batch, timesteps, features)
model = PSANNRegressor(
    hidden_layers=2,
    hidden_units=64,
    epochs=80,
    batch_size=32,
    attention={"kind": "mha", "num_heads": 4},  # optional MultiheadAttention
)
model.fit(seqs, y, verbose=0)
```

The estimator infers the `(timesteps, features)` layout automatically from the training tensor shape and inserts `nn.MultiheadAttention` after the PSANN per-token backbone. The same knob works when `preserve_shape=True` (including `per_element=True` conv paths); spatial positions are treated as tokens before passing through the existing convolutional heads. Set `attention=None` (default) to keep the legacy architecture.

### Residual regression with `ResPSANNRegressor`

```python
import numpy as np
from psann import ResPSANNRegressor

rng = np.random.default_rng(1234)
X = rng.uniform(-2.0, 2.0, size=(512, 4)).astype(np.float32)
y = (np.sin(X[:, :1]) + 0.25 * X[:, 1:2]).astype(np.float32)

est = ResPSANNRegressor(
    hidden_layers=4,
    hidden_units=48,
    lr=1e-3,
    epochs=120,
    early_stopping=True,
    patience=15,
    random_state=1234,
)
est.fit(X, y, verbose=0)
print("Residual R^2:", est.score(X, y))
```

`ResPSANNRegressor` keeps the same `.fit`/`.predict` interface but routes training through the residual backbone with DropPath, RMSNorm, and optional HISSO hooks enabled.

### Spectral-gated regression with `SGRPSANNRegressor`

```python
import numpy as np
from psann import SGRPSANNRegressor

rng = np.random.default_rng(7)
X = rng.standard_normal((256, 32, 3)).astype(np.float32)  # (batch, timesteps, features)
y = (X[:, -1:, :1].sum(axis=1) + 0.1 * X.mean(axis=(1, 2), keepdims=True)).astype(np.float32)

est = SGRPSANNRegressor(
    hidden_layers=3,
    hidden_units=32,
    epochs=40,
    k_fft=64,
    gate_type="rfft",
    pool="last",
)
est.fit(X, y, verbose=0)
print("SGR R^2:", est.score(X, y))
```

`SGRPSANNRegressor` adds per-channel phase shifts plus a lightweight spectral gate over the sequence axis. It expects `(N, T, F)` inputs and keeps the rest of the sklearn API identical.

### Language modeling (PSANN-LM)

Install the core estimators plus the LM add-on from PyPI:

```bash
pip install psann psannlm
```

Use the `psannlm` package for in-code training/generation and one-command training/CLI workflows:
- High-level APIs: `from psannlm import psannLM, psannLMDataPrep`
- CLI entrypoint: `python scripts/train_psann_lm.py` (thin wrapper around `python -m psannlm`)

Then spin up a minimal CPU demo:

```python
from psannlm import psannLM, psannLMDataPrep

texts = ["hello world", "goodnight moon", "the quick brown fox jumps over the lazy dog"]
dp = psannLMDataPrep(
    texts,
    tokenizer="auto",      # sentencepiece -> tokenizers -> char fallback
    max_length=64,
    pack_sequences=True,
)

model = psannLM(
    base="waveresnet",
    d_model=256,
    n_layers=4,
    n_heads=4,
    vocab_size=dp.vocab_size,
)
model.fit(dp, epochs=2, batch_tokens=8_192, lr=2e-4, amp="fp32", ddp="off")
print(model.generate("Once upon a time", max_new_tokens=48, top_p=0.9))
```

For a fully scripted example (dataset prep → train → generation logging), run:

```bash
python examples/lm/minimal_train.py --epochs 12 --repeat 64 --out reports/examples/<run_dir>
```

See `docs/lm.md` for the complete PSANN-LM reference plus GPU benchmark notes, and browse the ready-made snippets in `examples/lm/`.

### Convolutional regression with `ResConvPSANNRegressor`

```python
import numpy as np
from psann import ResConvPSANNRegressor

rng = np.random.default_rng(321)
X = rng.normal(size=(64, 12, 3)).astype(np.float32)  # (N, length, channels)
y = X.mean(axis=(1, 2), keepdims=True).astype(np.float32)

conv_est = ResConvPSANNRegressor(
    hidden_layers=3,
    hidden_units=32,
    conv_channels=24,
    conv_kernel_size=3,
    epochs=60,
    batch_size=16,
    data_format="channels_last",
    random_state=321,
)
conv_est.fit(X, y, verbose=0)
print("Conv R^2:", conv_est.score(X, y))
```

Passing `data_format="channels_last"` lets you keep inputs as `(N, length, channels)` arrays; the estimator handles the channel-first conversion internally while respecting `float32` inputs and the shared alias policy.
Tip: For GPU training set `device="cuda"` on construction, and keep NumPy arrays as `np.float32` to avoid extra copies.

### Wave-based regression with `WaveResNetRegressor`

```python
import numpy as np
from psann import WaveResNetRegressor

X = np.linspace(0, 2 * np.pi, 400, dtype=np.float32).reshape(-1, 1)
context = np.stack(
    [np.sin(X[:, 0]), np.cos(X[:, 0])],
    axis=1,
).astype(np.float32)
y = (np.sin(3 * X) + 0.1 * np.cos(5 * X)).astype(np.float32)

wave = WaveResNetRegressor(
    hidden_layers=4,
    hidden_units=64,
    epochs=150,
    lr=3e-4,
    w0=30.0,
    w0_warmup_epochs=20,
    progressive_depth_initial=2,
    progressive_depth_interval=25,
    random_state=7,
)
wave.fit(X, y, context=context, verbose=0)
print("WaveResNet R^2:", wave.score(X, y, context=context))
```

`WaveResNetRegressor` applies SIREN-style initialisation with optional `w0` warmup and progressive depth expansion. Providing explicit `float32` context arrays keeps inference aligned with the estimator's cached `context_dim`.
Tip: Specify `device="cuda"` for GPU runs; use `np.float32` inputs (including `context`) to stay on the fast path without extra dtype casts.

For sequence-shaped inputs `(N, T, F)`, `WaveResNetRegressor` can optionally apply a lightweight spectral gate over the inferred sequence axis before the WaveResNet readout. Enable it with `use_spectral_gate=True` and configure `k_fft`, `gate_type`, `gate_groups`, `gate_init`, and `gate_strength`.

### Convolutional WaveResNet with attention

```python
import numpy as np
from psann import WaveResNetRegressor

rng = np.random.default_rng(5)
frames = rng.normal(size=(48, 12, 3)).astype(np.float32)  # (batch, positions, channels)
targets = frames.mean(axis=(1, 2), keepdims=False).astype(np.float32).reshape(48, 1)

conv_wave = WaveResNetRegressor.with_conv_stem(
    conv_channels=24,
    conv_kernel_size=3,
    data_format="channels_last",
    hidden_layers=3,
    hidden_units=48,
    epochs=40,
    batch_size=16,
    attention={"kind": "mha", "num_heads": 4},  # token attention over the conv stem
)
conv_wave.fit(frames, targets, verbose=0)
print("Conv WaveResNet R^2:", conv_wave.score(frames, targets))
```

`with_conv_stem(...)` wires the estimator into the convolutional training path so tensors stay channel-first internally. The helper keeps the WaveResNet backbone (including optional context injection) while inserting a learnable convolutional stem and the shared attention builder before the dense WaveResNet readout.

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

`HISSOOptions` keeps reward, context, noise, and transformation choices in one place. The estimator records the resolved options after fitting so helpers such as `psann.hisso_infer_series` and `psann.hisso_evaluate_reward` can reuse them.

### Context builders

Setting `context_builder="cosine"` (or supplying a callable) instructs the estimator to synthesise auxiliary context features during `fit`, `predict`, and sequence roll-outs. Builder parameter dictionaries are deep-copied, so mutating your original config after `set_params` will not affect the estimator. Calling `set_params(context_builder=None)` clears the cached builder and resets the inferred `context_dim`, letting you switch back to explicit context arrays cleanly.

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

- **Sine activations** (`psann.SineParam`) expose learnable amplitude, frequency, and decay with optional bounds and SIREN-friendly initialisation.
- **LSM expanders** (`psann.LSM`, `psann.LSMExpander`, `psann.LSMConv2d`, `psann.LSMConv2dExpander`) provide sparse learned feature maps; `build_preprocessor` wires dict specs or modules into estimators with optional pretraining and separate learning rates.
- **State controllers** (`psann.StateController`) keep per-feature persistent gains for streaming/online workflows. Configurable via `StateConfig`.
- **Shared fit helpers** (`psann.estimators._fit_utils`) normalise arguments, materialise scalers, route through residual and convolutional builders, and orchestrate HISSO plans.
- **Wave backbones** (`psann.WaveResNet`, `psann.WaveEncoder`, `psann.WaveRNNCell`, `psann.scan_regimes`) surface the standalone components for experiments and spectral diagnostics outside the sklearn wrappers.
- **HISSO** (`psann.HISSOOptions`, `psann.hisso_infer_series`, `psann.hisso_evaluate_reward`) offers declarative reward configuration, supervised warm starts, episode construction, and inference helpers that reuse the cached configuration.
- **Utilities** (`psann.jacobian_spectrum`, `psann.ntk_eigens`, `psann.participation_ratio`, `psann.mutual_info_proxy`, `psann.encode_and_probe`, `psann.fit_linear_probe`, `psann.make_context_rotating_moons`, `psann.make_drift_series`, `psann.make_shock_series`, `psann.make_regime_switch_ts`) cover diagnostics and synthetic regimes.
- **Token helpers** (`SimpleWordTokenizer`, `SineTokenEmbedder`) remain for experiments that need sine embeddings, but no language-model trainer ships in this release.

## Package layout and tooling

- `src/psann/` – core PSANN library (estimators, HISSO, wave backbones, diagnostics, token helpers).
- `src/psann/lm/` – stub module that forwards users to `psannlm`.
- `psannlm/` – standalone PSANN-LM package published as `psannlm` on PyPI (LM APIs + training/CLI).
- `scripts/` – utility scripts for benchmarks, GPU validation, language modeling, and releases. See `scripts/README.md` for a categorized overview.
- `benchmarks/` – benchmark plans and result summaries for PSANN and PSANN-LM.
- `notebooks/` – Colab-friendly analysis notebooks (HISSO logging, parity/probes, sine comparisons, context demos); these are example/analysis artifacts and are not imported as modules.

## HISSO at a glance

1. Call `HISSOOptions.from_kwargs(...)` (or supply equivalent kwargs to `fit`) to resolve episode length, reward function, primary transform, transition penalty, context extractor, and optional noise.
2. Provide `hisso_supervised` to run a warm-start supervised phase before episodic optimisation.
3. `PSANNRegressor.fit(..., hisso=True, ...)` builds the episodic trainer using the shared fit pipeline.
4. After training, `hisso_infer_series(estimator, series)` and `hisso_evaluate_reward(estimator, series, targets=None)` reuse the cached configuration to score new data.

The project ships CPU benchmark baselines (`docs/benchmarks/`) and CI scripts (`scripts/benchmark_hisso_variants.py`, `scripts/compare_hisso_benchmarks.py`) to catch HISSO regressions.

### HISSO logging CLI

Use `python -m psann.scripts.hisso_log_run` to run HISSO sessions on remote nodes and collect reproducible artefacts. The command accepts JSON/YAML configs (see `configs/hisso/` templates) and emits:
- `metrics.json` with loss/reward/throughput summaries and optional portfolio metrics.
- `events.csv` containing append-only epoch logs and runtime notes (device, shuffle policy, AMP state).
- `checkpoints/` with the best estimator snapshot (and `latest.pt` when `--keep-checkpoints` is passed).
- `config_resolved.yaml` mirroring the resolved estimator/device settings for traceability.

Example:
```bash
python -m psann.scripts.hisso_log_run \
  --config configs/hisso/dense_cpu_smoke.yaml \
  --output-dir runs/hisso \
  --run-name dense_cpu_debug \
  --device cpu \
  --seed 7
```
Logging directories:
- Local shells: prefer `runs/hisso/` under the repo for easy diffing and archival.
- Colab/Runpod: prefer `/content/hisso_logs/` so artifacts persist in the notebook workspace or can be zipped for download.
The CLI intentionally requires an explicit `--output-dir`; no implicit defaults are applied.
When `device` points to a CUDA target and the config enables `mixed_precision`, the trainer switches to AMP + GradScaler automatically.

Looking for a guided walkthrough? Use the paired notebooks (both ship with Colab badges):
- `notebooks/HISSO_Logging_CLI_Walkthrough.ipynb` – CPU-first dry run with TODO placeholders.
- `notebooks/HISSO_Logging_GPU_Run.ipynb` – installs a published PyPI release (or a Git commit when specified via PSANN_PACKAGE_SPEC) and executes the logging CLI on CUDA to collect final metrics and checkpoints.

### Convolutional stems

`PSANNRegressor.with_conv_stem(...)`, `ResPSANNRegressor.with_conv_stem(...)`, and `WaveResNetRegressor.with_conv_stem(...)` return estimators wired into the convolutional training path without instantiating the legacy `*ConvPSANNRegressor` wrappers. The helpers enable `preserve_shape`, switch training to channel-first tensors, and honour `conv_channels`, `conv_kernel_size`, and `per_element` flags (WaveResNet keeps `per_element=False`). Example:
```python
est = PSANNRegressor.with_conv_stem(
    hidden_layers=2,
    hidden_units=32,
    conv_channels=16,
    conv_kernel_size=3,
    epochs=20,
    batch_size=32,
    random_state=42,
)
est.fit(images, targets)
```
Residual variants reuse the same call while producing `ResidualPSANNConv2dNet` cores when 2D inputs are supplied.

### Stateful dataloaders

When `stateful=True`, the training dataloader preserves sequence order. PSANN disables shuffling whenever `state_reset` is `"epoch"` or `"none"` so stateful models consume contiguous batches; keep the default `state_reset="batch"` to retain randomised mini-batches.

## Docs and examples

- Examples live in `examples/`; see `docs/examples/README.md` for the curated list (supervised, streaming, HISSO, benchmarks, diagnostics).
- Detailed internals are captured in `TECHNICAL_DETAILS.md`.
- Reward registry usage and custom strategy registration are described in `docs/API.md` under the HISSO section.

## Current status and roadmap

- Predictive extras and growth schedules are gone; legacy `extras_*` arguments are accepted but ignored with warnings for backward compatibility.
- Terminology has converged on `transition_penalty` within HISSO; the `trans_cost` alias still functions but will be removed in a later release.
- CPU benchmarks run in CI; CUDA smoke runs now live under `runs/hisso/dense/` and `runs/hisso/wave_resnet/`.
  - Colab (2025-11-01): dense (seed 7) duration 2.68 s, throughput 203 eps/s, train/val/test loss 0.245/0.304/0.231, reward_mean -0.111; WaveResNet (seed 11) duration 3.34 s, throughput 161 eps/s, train/val/test loss 1.435/1.402/1.569, reward_mean -0.182 (std 0.068).
  - Runpod L4 (2025-11-02; AMP float16): WaveResNet small `configs/hisso/wave_resnet_small.yaml` → `runs/hisso/wave_resnet_cuda_runpod_20251102_153117/` — 19.41 s over 1920 episodes (~107.3 eps/s), best_epoch 17, train/val/test 0.621/0.755/0.670, reward_mean -0.114 (std 0.010), turnover 2.69.
  - Note: Colab did not expose CUDA memory metrics; instrument `torch.cuda.max_memory_allocated()` on runpod if co-locating jobs.
- Upcoming work highlighted in `REPO_CLEANUP_TODO.md` includes broader reward coverage, lint/type sweeps, and release tooling improvements.

### Reproducibility

The notebook **PSANN_Parity_and_Probes.ipynb** (now under `notebooks/`) reproduces all key results under compute parity.  
- **Release:** [v1.0.0](https://github.com/Nickm1128/psann/releases/tag/v1.0.0)  
- **DOI:** [doi.org/10.5281/zenodo.17391523](https://doi.org/doi.org/10.5281/zenodo.17391523)  
- **Permalink:** [GitHub](https://github.com/Nickm1128/psann/blob/v1.0.0/notebooks/PSANN_Parity_and_Probes.ipynb)  
- **Render:** [nbviewer](https://nbviewer.org/github/Nickm1128/psann/blob/v1.0.0/notebooks/PSANN_Parity_and_Probes.ipynb)  
- **Run:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nickm1128/psann/blob/v1.0.0/notebooks/PSANN_Parity_and_Probes.ipynb)

Experiments used **Python 3.9**, dependencies pinned in `pyproject.toml` (install `[compat]` for constrained environments).
- When using the Colab notebook, set PSANN_PACKAGE_SPEC to the desired release (e.g., 'psann==0.10.10') or a git URL so the wheel includes 'psann.scripts'.
