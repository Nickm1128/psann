# PSANN Technical Details

This document explains the behaviour of the classes exported by PSANN. The README focuses on how to install and run examples; here we cover the mathematics and control flow behind the implementation. References below use the fully qualified module path (for example `psann.activations.SineParam`).

## 1. SineParam activation (`psann.activations.SineParam`)

`SineParam` is the core non-linearity used in every PSANN block. For an input tensor `z` with features along dimension `d`, the activation for feature index `i` is

```
A_i = softplus(a_i)
f_i = softplus(b_i) + eps_f
d_i = softplus(c_i)

h_i(z) = A_i * exp(-d_i * g(z_i)) * sin(f_i * z_i)
```

where:

- `a_i`, `b_i`, `c_i` are the unconstrained learnable parameters.
- `softplus(x) = log(1 + exp(x))` keeps the magnitudes positive, avoiding zero derivatives.
- `eps_f` is a small constant (`1e-6`) inserted so that frequencies never collapse to zero.
- `g(z)` is a selectable damping function:
  - `"abs"`: `g(z) = |z|`
  - `"relu"`: `g(z) = max(z, 0)`
  - `"none"`: `g(z) = 0`

Optional bounds clamp the post-softplus values. The output tensor is shaped exactly like the input, so the activation can be applied to dense, recurrent, or convolutional tensors as long as the feature dimension is known. Weight initialisation follows the SIREN prescription: for layers after the input, weights are sampled from `U(-sqrt(6 / fan_in) / w0, sqrt(6 / fan_in) / w0)` where `w0` is configurable (defaults to `30.0`).

## 2. Estimator architecture (`psann.sklearn`)

### 2.1 PSANN blocks

`psann.nn.PSANNBlock` contains (in order):

1. affine transformation `y = W x + b`
2. optional dropout or normalisation (depending on activation config)
3. `SineParam`
4. optional residual connection

For the residual variant the block outputs

```
out = x + alpha * block(x)
```

where `alpha` is the learnable residual scale initialised by `residual_alpha_init`.

### 2.2 Dense networks

`psann.nn.PSANNNet` builds a stack of `n` blocks followed by a linear readout. Let `x_0 = input`. For `k = 1..n`:

```
x_k = block_k(x_{k-1})
```

The final prediction is `y = W_out x_n + b_out`. The network can optionally wrap a preprocessor (see section 3) so that `x_0 = preproc(input)` without touching the estimator code.

### 2.3 Convolutional networks

`psann.conv.PSANNConv{1,2,3}dNet` extends the same idea to tensors `(batch, channels, spatial...)`. Convolutional blocks hold:

```
z = conv_k(x)
y = SineParam(z, feature_dim=channel_dim)
```

If `per_element = True`, the head is a 1x1 convolution that emits `y` with the same spatial layout as the input. Otherwise the tensor is pooled (global average) and a dense head computes the output.

### 2.4 Residual wrappers

`psann.sklearn.ResPSANNRegressor` and `psann.sklearn.ResConvPSANNRegressor` are thin wrappers that select the residual variant of the dense or convolutional bodies while reusing the estimator plumbing (argument normalisation, HISSO hooks, streaming API).

Both `PSANNRegressor` and `ResPSANNRegressor` expose `with_conv_stem(...)` / `enable_conv_stem(...)` helpers that wire the estimator into the convolutional training path without switching classes. The helpers flip `preserve_shape=True`, force channel-first training batches, and honour `conv_channels`, `conv_kernel_size`, and `per_element` so higher-dimensional inputs can be handled with the same API as dense fits.

### 2.5 Context builders

Estimators can automatically synthesise auxiliary context features via `context_builder`:

- `None` leaves inputs unchanged. You can still pass explicit context arrays to `fit`/`predict`.
- `"cosine"` instantiates the bundled cosine basis builder. It honours `context_builder_params` such as `frequencies`, `include_sin`, `include_cos`, and `normalise_input`, then caches the callable for reuse.
- Custom callables receive a flattened `np.float32` array with shape `(n_samples, input_dim)` and must return a `float32` array with the same sample dimension.

When `_auto_context` is invoked during `fit`, the builder output is validated, coerced to `float32`, and its feature dimension is stored in `_context_dim_`. Calling `set_params` with new `context_builder` or `context_builder_params` clears the cached callable so the next invocation rebuilds it with the updated configuration.


## 3. Preprocessors and LSMs (`psann.lsm`, `psann.preproc`)

PSANN supports optional Learned Sparse Map (LSM) expanders that turn low-dimensional inputs into high-dimensional, structured features:

```
z = R x + u
h = sin(z)
```

where:

- `R` is a sparse matrix sampled once using parameters such as `sparsity`, `hidden_units`, and `hidden_layers`.
- `u` is an optional trainable bias.
- The expander may include an internal linear readout trained by ordinary least squares (LSMExpander) or operate as a pure module (LSM).

`psann.preproc.build_preprocessor` accepts:

- dictionaries describing `LSM` or `LSMConv2d` specs,
- instances of modules with `forward`,
- or objects exposing `fit/transform`.

When `lsm_train=True`, `_fit_utils._build_optimizer` makes two parameter groups:

```
params = [
    {"params": model.core.parameters(), "lr": lr_main},
    {"params": model.preproc.parameters(), "lr": lr_preproc},
]
```

`lsm_pretrain_epochs > 0` triggers a pre-fit call to the expander before main training begins.

## 4. Scaling, losses, and optimisation (`psann.estimators._fit_utils`)

### 4.1 Scaling

Two built-in scalers handle numerical stability:

- `"standard"`: Welford running statistics maintain `(n, mean, M2)` to compute `(X - mean) / std`.
- `"minmax"`: track per-feature `min` and `max` and transform `(X - min) / (max - min)`.

Custom scalers only need `fit` and `transform`, optionally `partial_fit`.

### 4.2 Loss functions

The estimator accepts string aliases or callables:

- `"mse"`: `(1/N) * sum((y_hat - y)^2)`
- `"l1"`: `(1/N) * sum(|y_hat - y|)`
- `"smooth_l1"` (Huber with beta=1)
- `"huber"` (configurable with `loss_params`)

### 4.3 Optimisation

`optimizer` can be `"adam"`, `"adamw"`, or `"sgd"` (with momentum 0.9). Weight decay applies to the chosen optimiser. Mixed precision is not enabled by default, but the training loop is compatible with `torch.cuda.amp` when extended.

### 4.4 Fit pipeline

1. `normalise_fit_args` coerces features and targets to contiguous `np.float32`, wires through any user-provided context arrays, and merges HISSO kwargs. It also seeds RNGs for reproducibility.
2. `prepare_inputs_and_scaler` applies scalers, executes `_auto_context` when `context_builder` is set, caches the resulting context dimension, and returns a `PreparedInputState` with float32 tensors plus HISSO metadata.
3. `_make_fit_context` builds the `ModelBuildRequest`, attaches preprocessors, and moves the model to the target device via `_ensure_model_device` before training hooks run.
4. `run_supervised_training` (or `maybe_run_hisso` for episodic runs) executes the loop using the cached device/dtype policy.

Early stopping tracks validation loss, stores the best `state_dict`, and restores those weights when the patience window expires.

**Shuffle vs. stateful sequences.** The training dataloader chooses `shuffle=False` whenever `stateful=True` and `state_reset` is `"epoch"` or `"none"` so recurrent/stateful models see contiguous batches. Use the default `state_reset="batch"` (or `stateful=False`) to retain shuffled mini-batches.
### 4.5 Device & dtype policy

- Inputs, targets, and context features are coerced to `float32` up front. Supplying `np.float32` arrays avoids extra copies before tensors are created.
- Setting `device="cuda"` (or a concrete `torch.device`) prior to `fit` keeps the backbone, preprocessors, and HISSO state on that device. `_ensure_model_device` caches the last move to skip redundant transfers.
- Validation and inference reuse the cached device; CPU predictions return `float32` NumPy arrays while GPU paths stay on-device unless `predict(..., return_numpy=True)` is requested.
- Context builders and preprocessors must return `float32` arrays so downstream tensor materialisation never needs mid-pipeline dtype casts.

## 5. Persistent state controllers (`psann.state`)

`StateConfig` defines the coefficients for `StateController`:

```
state_t = rho * state_{t-1} + (1 - rho) * beta * mean(|y_t|)
state_t = clamp( max_abs * tanh(state_t / max_abs), -max_abs, max_abs )
```

- `rho in [0, 1)` controls how much memory is retained.
- `beta >= 0` scales the instantaneous response.
- `max_abs > 0` is the saturation limit.
- `detach` toggles gradient flow; if `True`, the state is detached before multiplication so the optimiser does not receive gradients through recurrent use.

Forward pass:

```
y_scaled = y * state.view(...)
```

Updates are buffered in `_pending_state` and only applied after the backward pass via `commit()`, preventing `in-place modification of a leaf` errors when autograd is tracking the tensor. Reset modes:

- `reset("batch")`: called between mini-batches.
- `reset("epoch")`: called once per epoch.
- `reset("none")`: leaves state untouched for long sequences or streaming inference.

The estimator exposes streaming helpers:

- `predict_sequence` free-runs without updating the model.
- `predict_sequence_online` applies teacher forcing; each step optionally updates parameters using `stream_lr`.
- `step` processes a single window, optionally performing a gradient step when `update_params=True`.

## 6. HISSO episodic training (`psann.hisso`)

HISSO (Horizon-Informed Sampling Strategy Optimisation) turns supervised estimators into episodic optimisers. The workflow:

1. Users supply `hisso=True` along with optional keyword overrides. `HISSOOptions.from_kwargs` normalises them into a dataclass:

   ```
   episode_length      T
   primary_transform   tau (identity, softmax, or tanh)
   transition_penalty  lambda
   reward_fn           R(outputs, context)
   context_extractor   C(inputs)
   input_noise_std     sigma (scalar)
   supervised          warm-start configuration
   ```

2. If `supervised` is truthy, `coerce_warmstart_config` constructs `HISSOWarmStartConfig` specifying targets, batch size, learning rates, and number of epochs. `run_hisso_supervised_warmstart` performs a standard regression pass before switching to episodic updates.

3. `build_hisso_training_plan` samples windows from the training series. For each episode `e`:

   ```
   sequence X_e in R^{T x F}
   optional targets Y_e in R^{T x D}
   context C_e = context_extractor(X_e) if provided
   ```

   The estimator rolls forward using the same `prepare_inputs_and_scaler` outputs as supervised training.

4. The reward is computed on the transformed primary output `tau(y_t)` and the context. The built-in finance strategy uses log returns:

   ```
   r_t = log(alloc_t^T * price_t) - lambda * ||alloc_t - alloc_{t-1}||_1
   reward = sum_t r_t
   ```

   Other rewards can be registered through `psann.rewards.register_reward_strategy`.

5. Gradients are accumulated across the episode and optimised with the same loss infrastructure. After training, the estimator retains:

   - `_hisso_options_` (resolved configuration),
   - `_hisso_trainer_` (history, profiling data),
   - `_hisso_cfg_` (trainer runtime settings),
   - `_hisso_reward_fn_` and `_hisso_context_extractor_`.

6. Inference helpers reuse these caches:

   ```
   hisso_infer_series(estimator, X)  -> rollout of allocations
   hisso_evaluate_reward(estimator, X, targets=None) -> scalar reward
   ```

7. Remote logging CLI: `psann.scripts.hisso_log_run` loads JSON/YAML configs (see `configs/hisso/`) and drives episodic runs on remote nodes. It captures metrics (`metrics.json`), event logs (`events.csv`), checkpoints, and a resolved-config snapshot so GPU jobs can be synced back into this repository. Mixed precision toggles automatically when the config sets `mixed_precision=true` and the target device is CUDA.

## 7. Wave-based backbones (`psann.models`, `psann.utils`)

This section details the WaveResNet family and the rationale behind its design. These models leverage sinusoidal activations with SIREN-style initialisation to capture oscillatory and high-frequency structure while preserving stable gradients.

### 7.1 WaveResNet (residual sine MLP)

`psann.models.wave_resnet.WaveResNet` is a deep residual MLP built from sine residual blocks with optional context modulation. It consumes vectors `x ? R^{input_dim}` (optionally a context `c ? R^{context_dim}`) and predicts `y ? R^{output_dim}`.

Structure:

- Stem: `h = W0 x + b0`, activation `SineParam(w0_first · h)`. The scalar `w0_first` scales the input frequency as in SIREN.
- Residual stack: for each block `k = 1..depth`:

  ```
  u = W1 h + b1
  u = w0_hidden · u
  if context: u ? u + F(c)              # additive phase shift
  v = SineParam(u)                       # per-feature amplitude/frequency/decay
  if context: v ? FiLM(v, c)             # multiplicative/additive gain from context
  v = W2 v + b2
  if norm: v ? RMSNorm(v) or identity
  out = h + a · Dropout(v)               # learnable residual scale a
  h = out
  ```

  - `SineParam` implements `A · exp(-d · g(z)) · sin(f · z)` with per-feature, optionally trainable amplitude `A`, frequency `f`, and damping `d` (see §1). In WaveResNet these parameters are per hidden feature and can be frozen or learned via the `activation_config.learnable` setting. The model constructor also exposes `trainable_params` for convenience.
  - `F(c)` is a learned linear projection from the context that acts as an additive phase shift before the sine; `FiLM(v, c)` applies feature-wise affine modulation after the nonlinearity. Both can be toggled independently.
  - Normalisation: choose between none, weight normalization (applied to linears), or RMSNorm on the residual branch output.
  - The residual scale `a` is a learned scalar per block initialised from `residual_alpha_init` (commonly `0.0` for residual-in-residual training stability).

- Head: a linear readout `y = W_out h + b_out`.

Initialisation and frequency control:

- All linears receive SIREN initialisation (`apply_siren_init`), using `w0_first` for the first linear encountered and `w0_hidden` for all subsequent linears. This sets the effective frequency support of the network at initialisation time.
- The sklearn wrapper `WaveResNetRegressor` optionally warms up `w0` values over the first `w0_warmup_epochs`, interpolating from `first_layer_w0_initial / hidden_w0_initial` to the target `first_layer_w0 / hidden_w0`. This helps optimiser stability on noisy data and aligns with the idea of gradually increasing frequency bandwidth during training.
- Progressive depth: `WaveResNetRegressor` can start with a shallow stack (`progressive_depth_initial`) and periodically append new residual blocks, adding their parameters to the optimizer while keeping running `w0` values consistent with the warmup schedule.

Why sine + residuals?

- A sine network with SIREN init can represent high-frequency signals with far fewer layers than ReLU/Tanh MLPs and avoids vanishing gradients via careful scaling. The residual wrapper keeps the effective Jacobian close to identity early on, improving depth-wise trainability.
- The `SineParam` generalises a pure `sin` by learning per-feature amplitude, frequency, and an exponential envelope (damping). This can be viewed as a learned, adaptive Fourier-like basis where each channel chooses its own frequency and envelope during training.
- Context modulation (phase shift + FiLM) offers a simple conditional mechanism that is interpretable in the time/frequency domain: the context can steer phases and per-feature gains.

Shapes and usage:

- Inputs: `(batch, input_dim)`; optional context: `(batch, context_dim)`.
- Internals are feature-wise (`feature_dim=-1`). `SineParam` parameters broadcast across the batch.
- The factory `build_wave_resnet(**kwargs)` builds a backbone directly; `WaveResNetRegressor` provides an sklearn interface with training schedules and progressive depth.

Mapping to code:

- Residual block definition: `src/psann/layers/sine_residual.py:51`.
- WaveResNet assembly and activation injection: `src/psann/models/wave_resnet.py:21`.
- SIREN init: `src/psann/initializers.py:39`.
- Estimator schedules (w0 warmup, progressive depth): `src/psann/sklearn.py:1616`.

### 7.2 Other wave components

- `WaveEncoder`: a lightweight encoder that applies stacks of 1D convolutions (with sine activations) and optional pooling to compress sequences into fixed-length embeddings before decoding.
- `WaveRNNCell`: a recurrent cell with sine activations. Given state `s_{t-1}` and input `x_t`, a pair of sine-activated affine maps produce a new hidden state; stacking forms RNNs or seq2seq models.

### 7.3 Diagnostics

The diagnostics suite (`psann.utils`) works on both estimator networks and standalone wave models:

- `jacobian_spectrum(model, inputs)` computes singular values of the Jacobian for sensitivity analysis.
- `ntk_eigens(model, inputs)` estimates Neural Tangent Kernel spectra.
- `participation_ratio(matrix)` measures effective dimensionality.
- `mutual_info_proxy(model, inputs, targets)` provides an information-theoretic proxy for representation quality.
- `fit_linear_probe(embeddings, targets)` trains linear readouts for probing frozen representations.
## 8. Benchmarks and scripts

Two scripts help manage performance baselines:

- `scripts/benchmark_hisso_variants.py` fits dense and convolutional HISSO configurations across datasets (`synthetic`, `portfolio`). It records wall-clock times, rewards, and episode metadata in JSON.
- `scripts/compare_hisso_benchmarks.py` compares new benchmark outputs with the stored baseline (`docs/benchmarks/`), applying tolerance thresholds to wall-clock and reward statistics.

These scripts run in CI to catch regressions introduced by code changes. They rely on the estimator interfaces described above, so improvements to `PSANNRegressor` automatically propagate to the benchmarks.

## 9. Summary

- The README explains installation and provides gentle usage examples.
- This document explains how the components work: mathematical definitions for the sine activation, state controllers, preprocessors, optimisers, HISSO, and wave backbones.
- For exhaustive argument reference, consult `docs/API.md`. For runnable scenarios, see the scripts listed in `docs/examples/README.md`.


