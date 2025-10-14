# PSANN API Reference
Install with `pip install psann[sklearn]` when you need scikit-learn conveniences; the base wheel only depends on NumPy and PyTorch. For pinned environments use `pip install -e . -c requirements-compat.txt` as documented in the README, or plan to install scikit-learn>=1.5 if you want the extra on NumPy 2.x.
This document summarises the public, user-facing API of `psann` with parameter names, expected shapes, and behavioural notes.
## psann.PSANNRegressor
Sklearn-style estimator that wraps PSANN networks (MLP and convolutional variants). Constructor parameters are grouped by concern. Unless otherwise stated, arguments accept plain Python scalars.
### Constructor parameters
**Architecture**
- `hidden_layers: int = 2` - number of PSANN blocks.
- `hidden_units: int = 64` - width/features per hidden block (preferred name).
- `hidden_width: int | None` - deprecated alias for `hidden_units` (kept for backward compatibility).
- `w0: float = 30.0` - SIREN-style initialisation scale.
- `activation: ActivationConfig | None` - forwarded to `SineParam`.
- `activation_type: str = "psann" | "relu" | "tanh"` - nonlinearity selection per block.
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
- `random_state: int | None` - seeds numpy, torch, and Python.
- `num_workers: int = 0` - DataLoader workers.
**Input handling**
- `preserve_shape: bool = False` - use convolutional body instead of flattening.
- `data_format: "channels_first" | "channels_last"` - layout when preserving shape.
- `conv_kernel_size: int = 1` - kernel size for conv blocks.
- `conv_channels: int | None` - channel count inside conv blocks (defaults to `hidden_units`).
- `per_element: bool = False` - return outputs at every spatial position (1x1 convolutional head) instead of pooled targets.
- `output_shape: tuple[int, ...] | None` - target shape for pooled head; defaults to `(target_dim,)` inferred from `y`.
**Stateful and streaming options**
- `stateful: bool = False` - enable persistent amplitude-like state.
- `state: dict | None` - keys `rho`, `beta`, `max_abs`, `init`, `detach` (see `StateController`).
- `state_reset: str = "batch" | "epoch" | "none"` - reset cadence during training.
- `stream_lr: float | None` - learning rate for `step(..., update=True)`.
**Preprocessors**
- `lsm: dict | LSMExpander | LSMConv2dExpander | LSM | LSMConv2d | nn.Module | None` - attach a learned sparse expander or custom module.
- `lsm_train: bool = False` - jointly train the attached expander (dense or convolutional) inside the estimator.
- `lsm_pretrain_epochs: int = 0` - optional pretraining epochs for expanders when `allow_train=True`.
- `lsm_lr: float | None` - separate learning rate for expander parameters.
- `scaler: str | object | None` - built-in `"standard"` / `"minmax"` scalers or custom estimator with `fit/transform`.
- `scaler_params: dict | None` - keyword arguments forwarded to the built-in scalers.
**Extras and warm starts**
- `extras: int = 0` - number of predictive extras outputs appended after the primary target.
- `extras_growth: int | dict | ExtrasGrowthConfig | None` - supply staged growth metadata (dimension, warm-start epochs, freeze flags, loss weighting). Legacy dict keys such as `extras`, `warm_start`, `freeze_until`, and `extras_loss_*` continue to hydrate the new config.
- `extras_warm_start_epochs: int | None = None` - convenience alias that hydrates `ExtrasGrowthConfig.warm_start_epochs` without replacing the entire config.
- `extras_freeze_until_plateau: bool | None = None` - when truthy, keep the extras head frozen after the warm-start window until validation (or training) loss stops improving for `patience` epochs.
- `warm_start: bool = False` - reuse fitted weights across consecutive `fit` calls when shapes align.
### `fit`
```python
def fit(
    X,
    y=None,
    *,
    validation_data=None,
    verbose=0,
    noisy=None,
    extras_targets=None,
    extras_loss_weight=None,
    extras_loss_mode=None,
    extras_loss_cycle=None,
    hisso=False,
    hisso_window=None,
    hisso_reward_fn=None,
    hisso_context_extractor=None,
    hisso_trans_cost=None,
    hisso_supervised=None,
    lr_max=None,
    lr_min=None,
    hisso_extras_weight=None,
    hisso_extras_mode=None,
    hisso_extras_cycle=None,
): ...
```
Shapes:
- `X`: `(N, F1, ..., Fk)` for flattened inputs, `(N, C, ...)` or `(N, ..., C)` when `preserve_shape=True`.
- `y`: `(N,)` or `(N, T)` for pooled targets. With `per_element=True`, match input spatial dims `(N, C_out, ...)` or `(N, ..., C_out)`.
- `extras_targets`: `(N, extras)` when supervising extras separately. If `extras > 0` and `y` has `primary_dim + extras` columns, extras are auto-detected and `extras_targets` is optional.
- `validation_data`: `(X_val, y_val)` or `(X_val, y_val, extras_val)`.
Behaviour notes:
- `noisy` adds Gaussian input noise; accepts scalars, flattened vectors, or tensors matching the internal input shape.
- `lr_max`/`lr_min` enable a linear learning-rate schedule over epochs (both must be provided).
- HISSO arguments control episodic reinforcement-style training. `hisso_supervised` can be `True` or a config dict to warm start with supervised epochs.
- Extras losses default to alternating updates when a weight is present. Override cadence via `extras_loss_mode` (`"joint"` or `"alternate"`) and `extras_loss_cycle` (e.g. `2` for alternating epochs).
- Extras warm-start staging freezes the extras head until `extras_warm_start_epochs` elapse (and optionally until losses plateau). Use `set_extras_warm_start_epochs()` to adjust the schedule after construction; the method rewires the current `ExtrasGrowthConfig` without discarding other fields.
- Whenever extras dimensions change (via `set_extras_growth`, `expand_extras_head`, or checkpoint auto-expansion) the estimator drops cached optimisers, schedulers, AMP scalers, and HISSO trainers. Expect a warning noting the reset; rebuild `extras_cache` only after observing a full episode length.
Returns the fitted estimator.
### Other methods
- `predict(X) -> ndarray` - returns pooled targets `(N, T)` or per-element outputs matching input shape.
- `score(X, y) -> float` - R^2 using scikit-learn when available, simple fallback otherwise.
- `hisso_infer_series(X_obs, E0=None) -> (allocations, extras)` - run trained HISSO policy over a full series.
- `hisso_evaluate_reward(X_obs, n_batches=8) -> float` - average episodic reward across random windows.
- `predict_sequence(X_seq, reset_state=True, return_sequence=False)` - deterministic rollout for stateful models.
- `predict_sequence_online(X_seq, y_seq, reset_state=True)` - teacher-forced rollout with per-step updates.
- `step(x_t, y_t=None, update=False)` - single-step inference; `update=True` applies an immediate gradient step using `y_t`.
- `set_extras_warm_start_epochs(epochs, *, freeze_until_plateau=None)` - update the staged extras schedule in-place; use `None` to clear the warm-start window or toggle the plateau gate without rebuilding the full config.
- `reset_state()` and `commit_state_updates()` - manage the internal state controller when `stateful=True`.

#### Extras staging and compatibility
- load() only auto-expands extras heads when the saved HISSO metadata reports a matching episode length and the persisted extras cache spans at least one full window. Otherwise the estimator keeps the checkpoint width and emits a warning pointing to this section.
- Legacy extras, extras_growth, and warm-start dict shorthands continue to hydrate ExtrasGrowthConfig. Combine them with set_extras_warm_start_epochs() for readability in new code.
- When migrating existing checkpoints expect warnings about optimizer/scheduler resets; these are informational and indicate that cached moments were dropped on purpose after the head width changed.

#### Extras staging and compatibility
- `load()` only auto-expands extras heads when the saved HISSO metadata reports a matching episode length and the persisted extras cache spans at least one full window. Otherwise the estimator keeps the checkpoint width and emits a warning pointing to this section.
- Legacy `extras`, `extras_growth`, and warm-start dict shorthands continue to hydrate `ExtrasGrowthConfig`. Combine them with `set_extras_warm_start_epochs()` for readability in new code.
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
## LSM Expanders and Preprocessors
- `LSM(...)` - Torch module that expands inputs with sparse random weights; callable from PyTorch graphs.
- `LSMExpander(...)` - Learns an OLS readout; exposes `fit/transform/fit_transform/score_reconstruction`, behaves like a standard `nn.Module` (`forward`, `to`, `train`, `eval`), and accepts either NumPy arrays or torch tensors (returning the same type).
- `LSMConv2d(...)` / `LSMConv2dExpander(...)` - channel-preserving 2D equivalents for spatial data; the expander mirrors the dense API (tensor-aware transforms, module wrappers) and now offers `score_reconstruction` for per-pixel diagnostics.
- `build_preprocessor(value, *, allow_train=False, pretrain_epochs=0, data=None)` - normalises user input (dict/spec/module) into `(preprocessor_module, base_model)` tuples. Provides optional pretraining when `allow_train=True` and data is supplied.
## psann.PSANNLanguageModel and utilities
- `SimpleWordTokenizer` - whitespace tokenizer with special tokens and `fit/encode/decode` helpers.
- `SineTokenEmbedder(embedding_dim, learnable=True, ...)` - produces sine-based token embeddings, supports resizing via `set_vocab_size`.
- `LMConfig` - dataclass of language-model hyperparameters (`embedding_dim`, `extras_dim`, `episode_length`, etc.).
- `PSANNLanguageModel(tokenizer, embedder, lm_cfg, hidden_layers=2, hidden_units=64, ...)`:
  - `fit(corpus, epochs=50, lr=1e-3, noisy=None, verbose=1, ppx_every=None, ppx_temperature=None, curriculum_type=None, ...)`
  - `predict(text, return_embedding=False)` - next-token string or embedding.
  - `gen(prompt, max_tokens=20)` - autoregressive generation.
  - `save(path)` / `load(path)` - persist pipeline components and learned weights.
Perplexity is estimated with a cosine-similarity softmax against the embedding table (temperature via `ppx_temperature`). `curriculum_type="progressive_span"` limits training windows to a growing fraction of each document for warm starts.
