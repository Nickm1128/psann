# PSANN API Reference

Install the core library with `pip install psann`. When you need scikit-learn conveniences, use `pip install psann[sklearn]`; the base wheel only depends on NumPy and PyTorch. Language-modeling utilities now live in the separate `psannlm` package (`pip install psannlm`). For pinned environments use the compatibility extra (`pip install -e .[compat]`) as documented in the README. This document summarises the public, user-facing API of `psann` with parameter names, expected shapes, and behavioural notes.

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
- `activation_type: str = "psann" | "relu" | "tanh" | "relu_sigmoid_psann"` - nonlinearity per block.
- `attention: dict | AttentionConfig | None` - optional token attention module (e.g. `{"kind": "mha", "num_heads": 4}`). Canonical preprocessing for attention must be a typed custom `tokens→tokens` module; dense LSM preprocessing is rejected rather than ignored.

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
- `preprocessor: PreprocessorConfig | Mapping | None` - the canonical preprocessing boundary. Use `PreprocessorConfig(LSMConfig.dense(...))` for flat LSM input or `LSMConfig.convolutional(...)` for 2D spatial input. The training policy is explicit in `PreprocessorTrainingConfig(trainable=..., lr=...)`.
- `lsm`, `lsm_train`, `lsm_pretrain_epochs`, and `lsm_lr` remain deprecated 0.x compatibility arguments. They emit one `DeprecationWarning`; do not combine them with `preprocessor`.
- `scaler: str | object | None` - string alias (`"standard"`/`"minmax"`) or any transformer exposing `fit`/`transform`.
- `scaler_params: dict | None` - keyword arguments forwarded to the built-in scalers.

**HISSO configuration**
- `hisso_window: int | None` - episode length when training with `hisso=True` (defaults to 64).
- `hisso_batch_episodes: int | None` - number of episodes sampled per HISSO optimizer update (defaults to 32 when omitted).
- `hisso_updates_per_epoch: int | None` - number of HISSO optimizer updates per epoch (defaults to compatibility behavior when omitted).
- `hisso_reward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None` - reward callback that consumes transformed primary outputs and context.
- **Device tip:** HISSO runs entirely on the estimator’s current device. Set `device="cuda"` (or the desired `torch.device`) before calling `fit` to keep episodes on GPU, and supply float32 inputs/contexts to avoid host copies. On CPU-only machines that still install CUDA wheels the training loop is wrapped in a guard that suppresses `torch.cuda.is_current_stream_capturing()` failures when the runtime is missing, so HISSO can run without a GPU driver.
- `hisso_context_extractor: Callable[[torch.Tensor], torch.Tensor] | None` - optional callable that derives context tensors from inputs; outputs are coerced onto the estimator’s device/dtype and aligned to the primary action width (singleton channels broadcast, wider contexts are trimmed or repeated). Temporal/episode mismatches raise a `ValueError` that reports both shapes. If the extractor rejects tensor inputs and HISSO falls back to NumPy, a one-time runtime warning explains the potential host/device transfer cost and how to fix it.
- `hisso_primary_transform: str | None` - transform applied to primary outputs before reward evaluation (`"identity"` | `"softmax"` | `"tanh"`).
- `hisso_transition_penalty: float | None` - smoothness penalty applied between HISSO steps (alias `hisso_trans_cost` is tolerated for compatibility). When the reward callback accepts `transition_penalty` (or legacy `trans_cost`), HISSO forwards this value automatically during training and `hisso_evaluate_reward`.
- `stateful` / `state_reset` interaction: when `stateful=True`, HISSO mirrors supervised loop semantics (`state_reset="batch"` resets before each sampled episode, `"epoch"` resets once per epoch, `"none"` leaves state untouched). Staged state updates are committed after each HISSO optimizer step.
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
    hisso_batch_episodes: Optional[int] = None,
    hisso_updates_per_epoch: Optional[int] = None,
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
- `hisso_batch_episodes` / `hisso_updates_per_epoch`: tune HISSO schedule (`episodes_per_batch` and update count) without changing model code.
- Recommended starting presets: CPU `hisso_batch_episodes=8`, `hisso_updates_per_epoch=4`; CUDA `hisso_batch_episodes=16`, `hisso_updates_per_epoch=4` (increase batch size until memory limits).
- `lr_max` / `lr_min`: optional bounds for one-cycle style schedulers.

When HISSO is enabled and no targets are provided the primary dimension defaults to 1. If you provide `hisso_supervised={"y": targets}` the estimator runs a supervised warm start before episodic training.

### Other methods

- `predict(X) -> np.ndarray` - returns pooled targets `(N, T)` or per-element outputs matching the configured spatial layout.
- `score(X, y) -> float` - coefficient of determination (R^2) using scikit-learn when available, with a lightweight fallback otherwise.
- `score_reconstruction(X) -> float` - score a fitted canonical LSM preprocessor's reconstruction. It is available only after fitting an LSM-backed `preprocessor=` configuration; it uses the fitted scaler and input layout (including channels-last conversion) and remains available after schema-v2 checkpoint reloads. It raises a clear error for custom/no-preprocessor estimators.
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

## Sequence architecture (canonical)

Use `PSANNRegressor(architecture=ArchitectureConfig.for_sequence(...))` for the
spectral-gated sequence architecture. `SGRPSANNRegressor` is retained only as a
deprecated compatibility wrapper.

**Key parameters**
- `phase_init: float = 0.0` - initial phase offset for each hidden channel.
- `phase_trainable: bool = True` - toggles phase learning.
- `use_spectral_gate: bool = True` - enable the FFT/Fourier feature gate.
- `k_fft: int = 64` - window length for spectral gating.
- `gate_type: "rfft" | "fourier_features"` - FFT-based or fixed Fourier features.
- `gate_groups: "depthwise" | "full"` - depthwise (per-channel) or full 1x1 mixing.
- `gate_init: float = 0.0` - initial gate logits (sigmoid ~0.5).
- `gate_strength: float = 1.0` - residual scale for the gated branch.
- `pool: "last" | "mean"` - reduce token outputs to a fixed vector for the head.

**Notes**
- Expects `(N, T, F)` inputs (sequence length `T`, feature width `F`).
- Does not support `preserve_shape=True` or `per_element=True`.
- Canonical sequence preprocessing accepts only typed custom `tokens→tokens` modules. Legacy SGR LSM compatibility remains warned/ignored through 0.x.
- Attention configs are ignored; the spectral gate operates on the inferred sequence axis.

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

Use the canonical boundary for estimators:

```python
from psann import PSANNRegressor
from psann.preprocessing import LSMConfig, LSMPretrainingConfig, PreprocessorConfig

estimator = PSANNRegressor(
    preprocessor=PreprocessorConfig(
        LSMConfig.dense(output_dim=64, pretraining=LSMPretrainingConfig(epochs=5))
    )
)
```

`ModulePreprocessorConfig(module, input_topology, output_topology, output_dim)` is the
typed route for custom modules. `LSM`, `LSMExpander`, `LSMConv2d`, and
`LSMConv2dExpander` remain low-level research tools; import them from
`psann.preprocessing` or top-level `psann`. `psann.lsm`, `psann.preproc`,
`PreprocessorSpec`, and `build_preprocessor` are 0.x compatibility paths.

## Token and embedding helpers

- `SimpleWordTokenizer` - whitespace tokenizer with `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>` tokens plus `fit/encode/decode` helpers for prototyping text pipelines.
- `SineTokenEmbedder(embedding_dim, trainable=False, base=10000.0, scale=1.0, ...)` - sine-based token embeddings with optional learnable amplitude/phase/offset parameters and lazy table materialisation via `set_vocab_size`.

These utilities are exposed for experiments that need sine embeddings or lightweight tokenisation; they do not ship a full language-model trainer in this release.
