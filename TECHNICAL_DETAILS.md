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

1. `normalise_fit_args` coerces inputs (`X`, `y`, optional HISSO kwargs) into `NormalisedFitArgs`. It validates dtypes, handles `validation_data`, and enforces reproducibility by calling `seed_all`.
2. `prepare_inputs_and_scaler` applies scaling, handles shape conversions, and returns `PreparedInputState` containing:
   - flattened and channels-first tensors,
   - target shapes,
   - scaler state,
   - metadata for HISSO.
3. Hooks are assembled into a `ModelBuildRequest`. `build_model_from_hooks` constructs the torch module and attaches preprocessors.
4. `run_supervised_training` or `maybe_run_hisso` executes the appropriate training routine.

Early stopping is implemented by tracking validation loss and storing the best `state_dict`. When `early_stopping=True` and a plateau is detected, the best weights are restored.

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

## 7. Wave-based backbones (`psann.models`, `psann.utils`)

The models module exports standalone building blocks for sinusoidal architectures outside the sklearn wrappers:

- `WaveResNet`: a residual network composed of `SineParam` convolutional layers. Each residual unit implements

  ```
  h = SineParam(Conv(h, kernel_size, padding))
  h = SineParam(Conv(h, kernel_size, padding))
  out = h_in + alpha * h
  ```

  `alpha` follows a dropout-style schedule (`drop_path_max`) to control stochastic depth.

- `WaveEncoder`: a stack of 1D convolutions with sine activations and optional pooling. Useful for compressing sequences before decoding.

- `WaveRNNCell`: recurrent cell using sine activations. Given state `s_{t-1}` and input `x_t`,

  ```
  h_t = SineParam(W_h [x_t, s_{t-1}] + b_h)
  s_t = SineParam(W_s [x_t, s_{t-1}] + b_s)
  ```

  The cell can be wrapped to form RNNs or sequence-to-sequence models.

- `build_wave_resnet(**kwargs)` and `scan_regimes(config)` provide convenience factories for constructing and analysing these backbones.

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
