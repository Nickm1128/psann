# PSANN API Reference

Install the core library with `pip install psann`. When you need scikit-learn conveniences, use `pip install psann[sklearn]`; the base wheel only depends on NumPy and PyTorch. Language-modeling utilities live under `psann.lm` with additional dependencies pulled in via the `psann[lm]` extra or the separate `psannlm` package. For pinned environments use the compatibility extra (`pip install -e .[compat]`) as documented in the README. This document summarises the public, user-facing API of `psann` with parameter names, expected shapes, and behavioural notes.

## psann.PSANNRegressor

Sklearn-style estimator that wraps PSANN networks (MLP and convolutional variants). Constructor parameters are grouped by concern. Unless otherwise stated, arguments accept plain Python scalars.

### Constructor parameters

> **Alias policy.** `hidden_width` and `hidden_channels` remain as deprecated aliases so existing pipelines keep working. They always normalise to the canonical `hidden_units` / `conv_channels` values via `PSANNRegressor.set_params`; when both names are supplied the canonical keyword wins and a `UserWarning` is emitted.

**Architecture**
- `hidden_layers: int = 2` - number of PSANN blocks.
- `hidden_units: int = 64` - width/features per hidden block (preferred name).
- `hidden_width: int | None` - deprecated alias for `hidden_units`; conflicts emit a warning and the canonical `hidden_units` value wins (automatically normalised by `set_params`).
- `w0: float = 30.0` - SIREN-style initialisation scale.
- `activation: ActivationConfig | None` - forwarded to `SineParam`.
- `activation_type: str = "psann" | "relu" | "tanh"` - nonlinearity per block.
- `attention: dict | AttentionConfig | None` - optional token attention module (e.g. `{"kind": "mha", "num_heads": 4}`) that activates when inputs are sequences shaped `(batch, timesteps, features)` or preserved spatial tensors. Flattened inputs currently require `lsm=None`, while preserve-shape paths (including `per_element=True`) treat each spatial location as a token. Defaults to `"none"` which preserves historical behaviour.

**Training**
- `epochs: int = 200`, `batch_size: int = 128`, `lr: float = 1e-3`.
- `optimizer: str = "adam" | "adamw" | "sgd"`.
- `weight_decay: float = 0.0`.
- `loss: str | callable = "mse" | "l1" | "smooth_l1" | "huber" | callable`.
- `loss_params: dict | None` - extra kwargs for built-in losses.
- `loss_reduction: str = "mean" | "sum" | "none"`.
- `early_stopping: bool = False`, `patience: int = 20`.

**Runtime**
- `device: "auto" | "cpu" | "cuda" | torch.device`.
- `random_state: int | None` - seeds NumPy, Torch, and Python.
- `num_workers: int = 0` - DataLoader workers for supervised fits.

**Input handling**
- `preserve_shape: bool = False` - use convolutional body instead of flattening.
- `data_format: "channels_first" | "channels_last"` - layout when preserving shape.
- `conv_kernel_size: int = 1` - kernel size for convolutional blocks.
- `conv_channels: int | None` - channel count inside conv blocks (defaults to `hidden_units`; the legacy `hidden_channels` alias is still accepted but must match and is normalised via `set_params`).
- `per_element: bool = False` - return outputs at every spatial position (1x1 convolutional head) instead of pooled targets.
- `output_shape: tuple[int, ...] | None` - target shape for pooled heads; defaults to `(target_dim,)` inferred from `y`.

**Stateful and streaming options**
- `stateful: bool = False` - enable persistent amplitude-like state.
- `state: StateConfig | Mapping[str, Any] | None` - prefer `StateConfig(...)` to configure `rho`, `beta`, `max_abs`, `init`, and `detach`; mappings are still accepted for compatibility.
- `state_reset: str = "batch" | "epoch" | "none"` - reset cadence during training.
- `stream_lr: float | None` - learning rate for `step(..., update=True)` or teacher-forced streaming updates.

**Preprocessors**
- `lsm: dict | LSMExpander | LSMConv2dExpander | LSM | LSMConv2d | nn.Module | None` - attach a learned sparse expander or custom module.
- `lsm_train: bool = False` - jointly train the attached expander (dense or convolutional) inside the estimator.
- `lsm_pretrain_epochs: int = 0` - optional pretraining epochs for expanders when `allow_train=True`.
- `lsm_lr: float | None` - separate learning rate for expander parameters.
- `scaler: str | object | None` - string alias (`"standard"`/`"minmax"`) or any transformer exposing `fit`/`transform`.
- `scaler_params: dict | None` - keyword arguments forwarded to the built-in scalers.

**HISSO configuration**
- `hisso_window: int | None` - episode length when training with `hisso=True` (defaults to 64).
- `hisso_reward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None` - reward callback that consumes transformed primary outputs and context.
- **Device tip:** HISSO runs entirely on the estimator’s current device. Set `device="cuda"` (or the desired `torch.device`) before calling `fit` to keep episodes on GPU, and supply float32 inputs/contexts to avoid host copies. On CPU-only machines that still install CUDA wheels the training loop is wrapped in a guard that suppresses `torch.cuda.is_current_stream_capturing()` failures when the runtime is missing, so HISSO can run without a GPU driver.
- `hisso_context_extractor: Callable[[torch.Tensor], torch.Tensor] | None` - optional callable that derives context tensors from inputs; outputs are coerced onto the estimator’s device/dtype and aligned to the primary action width (singleton channels broadcast, wider contexts are trimmed or repeated). Temporal/episode mismatches raise a `ValueError` that reports both shapes.
- `hisso_primary_transform: str | None` - transform applied to primary outputs before reward evaluation (`"identity"` | `"softmax"` | `"tanh"`).
- `hisso_transition_penalty: float | None` - smoothness penalty applied between HISSO steps (alias `hisso_trans_cost` is tolerated for compatibility).
- `hisso_supervised: Mapping[str, Any] | bool | None` - opt into a supervised warm start before HISSO (provide `{"y": targets}` to reuse labels).

Predictive extras and their growth schedules have been removed; any legacy `extras_*` arguments are ignored with warnings.

### `fit`

```python
def fit(
    self,
    X: np.ndarray,
    y: np.ndarray | None,
    *,
    validation_data: Optional[tuple[np.ndarray, np.ndarray]] = None,
    verbose: int = 0,
    noisy: Optional[NoiseSpec] = None,
    hisso: bool = False,
    hisso_window: Optional[int] = None,
    hisso_reward_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    hisso_context_extractor: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    hisso_primary_transform: Optional[str] = None,
    hisso_transition_penalty: Optional[float] = None,
    hisso_trans_cost: Optional[float] = None,
    hisso_supervised: Optional[Mapping[str, Any] | bool] = None,
    lr_max: Optional[float] = None,
    lr_min: Optional[float] = None,
) -> "PSANNRegressor":
    ...
```

- `X`: `(N, F1, ..., Fk)` for flattened inputs, `(N, C, ...)` or `(N, ..., C)` when `preserve_shape=True`.
- `y`: required when `hisso=False`; accepts `(N,)` or `(N, target_dim)` for pooled heads, or spatial layouts matching `X` when `per_element=True`.
- `validation_data`: `(X_val, y_val)` tuple used by early stopping; both arrays are coerced to `float32` internally.
- `noisy`: optional Gaussian noise standard deviation applied to inputs during training (scalar or array-like).
- `hisso`: switch to episodic Horizon-Informed Sampling Strategy Optimisation. When true the helper normalises reward/context/transform settings via `HISSOOptions.from_kwargs` before launching the episodic loop.
- `lr_max` / `lr_min`: optional bounds for one-cycle style schedulers.

When HISSO is enabled and no targets are provided the primary dimension defaults to 1. If you provide `hisso_supervised={"y": targets}` the estimator runs a supervised warm start before episodic training.

### Other methods

- `predict(X) -> np.ndarray` - returns pooled targets `(N, T)` or per-element outputs matching the configured spatial layout.
- `score(X, y) -> float` - coefficient of determination (R^2) using scikit-learn when available, with a lightweight fallback otherwise.
- `hisso_infer_series(X_obs, *, trainer_cfg=None) -> np.ndarray` - run the trained HISSO policy over a full series using the stored primary transform.
- `hisso_evaluate_reward(X_obs, *, trainer_cfg=None) -> float` - evaluate the configured reward function across observed inputs.
- `predict_sequence(X_seq, *, reset_state=True, return_sequence=False, update_state=True)` - deterministic rollout for stateful models; set `return_sequence=True` to capture the full trace.
- `predict_sequence_online(X_seq, y_seq, *, reset_state=True, return_sequence=True, update_state=True)` - teacher-forced rollout that applies per-step streaming updates when `stream_lr` is configured.
- `step(x_t, *, target=None, update_params=False, update_state=True)` - single-step inference; pass a target with `update_params=True` to apply an immediate streaming update.
- `reset_state()` / `commit_state_updates()` - manage the internal state controller when `stateful=True`.

### Stateful and streaming workflow

1. Configure the estimator with `stateful=True`, provide a `StateConfig(...)`, and set `stream_lr` if online updates are required.
2. Fit on supervised data as usual; optionally stage a HISSO warm start via `hisso_supervised` before reinforcement fine-tuning.
3. Use `predict_sequence(...)` for open-loop rollouts, or `predict_sequence_online(...)` when teacher forcing and online adaptation are required.
4. Utilities such as `psann.make_drift_series`, `psann.make_shock_series`, and `psann.make_regime_switch_ts` provide quick regression regimes for exercising the streaming APIs.

## psann.SineParam

Learnable sine activation with per-feature amplitude, frequency, and decay.

Constructor:
- `out_features: int`
- `amplitude_init=1.0`, `frequency_init=1.0`, `decay_init=0.1`
- `learnable=('amplitude', 'frequency', 'decay') | str`
- `decay_mode='abs' | 'relu' | 'none'`
- `bounds={'amplitude': (low, high), ...}`
- `feature_dim=-1` - axis that holds feature channels

Forward applies `A * exp(-d * g(z)) * sin(f * z)` with broadcast parameters.

## LSM expanders and preprocessors

- `LSM(...)` - Torch module that expands inputs with sparse random weights; callable from PyTorch graphs.
- `LSMExpander(...)` - learns an OLS readout; exposes `fit/transform/fit_transform/score_reconstruction`, behaves like a standard `nn.Module` (`forward`, `to`, `train`, `eval`), and accepts either NumPy arrays or torch tensors (returning the same type).
- `LSMConv2d(...)` / `LSMConv2dExpander(...)` - channel-preserving 2D equivalents for spatial data; the expander mirrors the dense API (tensor-aware transforms, module wrappers) and offers `score_reconstruction` for per-pixel diagnostics.
- `build_preprocessor(value, *, allow_train=False, pretrain_epochs=0, data=None)` - normalises user input (dict/spec/module) into `(preprocessor_module, base_model)` tuples. Provides optional pretraining when `allow_train=True` and data is supplied.

## Token and embedding helpers

- `SimpleWordTokenizer` - whitespace tokenizer with `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>` tokens plus `fit/encode/decode` helpers for prototyping text pipelines.
- `SineTokenEmbedder(embedding_dim, trainable=False, base=10000.0, scale=1.0, ...)` - sine-based token embeddings with optional learnable amplitude/phase/offset parameters and lazy table materialisation via `set_vocab_size`.

These utilities are exposed for experiments that need sine embeddings or lightweight tokenisation; they do not ship a full language-model trainer in this release.

