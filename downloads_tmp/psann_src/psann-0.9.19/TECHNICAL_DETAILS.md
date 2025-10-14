# PSANN Technical Details

This document explains the core components of PSANN: the parameterised sine activation, network architecture, stateful extensions, predictive extras, HISSO training, and the lightweight PSANN-based language model.

## 1. Parameterised Sine Activation

Given pre-activation vector `z` (e.g. `z = xW + b`), each unit outputs:

```
h = A * exp(-d * g(z)) * sin(f * z)
```

where amplitude `A`, frequency `f`, and decay `d` are learnable scalars per output feature. The decay function `g(z)` controls amplitude attenuation:
- `abs`: `g(z) = |z|`
- `relu`: `g(z) = max(0, z)`
- `none`: `g(z) = 0` (no decay term)

Parameterisation:
- Parameters are stored in an unconstrained space and mapped with softplus to keep them positive and numerically stable.
- Optional lower/upper bounds clamp the mapped values.
- Weight initialisation follows SIREN heuristics so that gradients stay in range for deep stacks.

Intuition: frequency controls oscillation speed, amplitude scales the output, and decay provides a stabilising envelope that prevents runaway activations.

## 2. PSANN Blocks and Networks

- `PSANNBlock`: Linear layer followed by `SineParam`. Optionally multiplies activations by a `StateController` output.
- `PSANNNet`: MLP stack of PSANN blocks with a final linear readout.
- `PSANNConv{1,2,3}dNet`: convolutional equivalents that keep spatial layout intact. `per_element=True` swaps the global pooling head for a 1x1 convolution so predictions are emitted at every spatial position.

## 3. Persistent State for Time Series

Each block may include a `StateController` with a learnable scalar per feature. The state is updated from the magnitude of current activations and clipped with a smooth envelope.

Update rule (per feature):
```
s_t = rho * s_{t-1} + (1 - rho) * beta * mean(|y_t|)
s_t = max_abs * tanh(s_t / max_abs)
```

- `rho` controls persistence (`0 < rho < 1`).
- `beta` scales the contribution from the new activation magnitude.
- `max_abs` defines the asymptotic bound via `tanh`.
- `detach` determines whether the state participates in autograd (`True` detaches).

Implementation notes:
- During forward, a detached copy of the state multiplies activations when `detach=True` to avoid stale references.
- Proposed state updates are stored and committed after the optimiser step via `commit_state_updates()` to avoid in-place modification during backprop.
- Reset policy: `state_reset` accepts `"batch"`, `"epoch"`, or `"none"`; batching is kept ordered when state spans across samples.
- `predict_sequence` and `predict_sequence_online` reuse the same state machinery for inference.

## 4. Multi-Dimensional Inputs

Two processing modes support generic tensor inputs:
- Flattened MLP: inputs `(N, F1, ..., Fk)` are reshaped to `(N, prod(F*))`.
- Preserve shape: convolutional nets expect `(N, C, ...)` or `(N, ..., C)` depending on `data_format`. Optional per-element head produces outputs with the same spatial layout.

Gaussian input noise (`noisy`) accepts a scalar, a flattened vector, or a tensor that matches the internal layout and is broadcast across the batch.

## 5. Loss Functions and Scheduling

Built-in losses: mean-squared error, L1/MAE, SmoothL1, and Huber. Custom callables are supported when they take `(pred, target)` and return a scalar tensor per batch. Extras supervision can run jointly or in alternating cycles by configuring `extras_loss_mode` (`"joint"` or `"alternate"`) and `extras_loss_cycle`.

## 6. Initialisation and Stability

- Linear and convolutional layers use SIREN-style uniform initialisers derived from fan-in.
- Sine parameters use softplus mappings; decay ensures bounded outputs over time.
- State updates are clamped by `tanh` to avoid divergence.

## 7. Research Directions

1. Frequency/amplitude priors: spectral regularisation, parameter tying, or gating over `(A, f, d)`.
2. Physics-informed hybrids: constrain `f` and `d` for damped harmonic regimes or couple with analytical filters.
3. State dynamics: learn update coefficients, add gates, or explore truncated BPTT batching.
4. Spatial models: deeper conv PSANNs with multi-scale features or attention over spatial tokens.
5. Representation learning: self-supervised pretraining and spectral pretext tasks.
6. Robustness: OOD detection and uncertainty calibration for sine-activated models.

## 8. Predictive Extras and HISSO

Predictive extras append `K` additional outputs to the primary target. During supervised training you can either append extras columns to `y` or provide `extras_targets`. Extras are rolled forward internally so they can act as latent context for the next timestep.

Horizon-Informed Sampling Strategy Optimisation (HISSO):
- `fit(..., hisso=True, hisso_window=T)` samples windows `(B, T, F)` and runs sequential rollouts with extras fed back into the model.
- Reward defaults to a portfolio-style objective; pass `hisso_reward_fn` to override.
- `hisso_supervised` performs a supervised warm start before switching to reward-driven updates.
- `hisso_extras_weight/mode/cycle` mirrors the supervised extras scheduling but inside the episodic loop.

## 9. PSANN Language Model

`PSANNLanguageModel` predicts next-token embeddings while optionally rolling extras forward.

Components:
- `SimpleWordTokenizer`: whitespace tokeniser with `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>` tokens.
- `SineTokenEmbedder`: sine-based embeddings where amplitude, phase, and offset can be learnable.
- `PSANNNet` core: maps `[embedding_t, extras_t]` to `[embedding_{t+1}, extras_{t+1}]`.

Training:
- Samples windows of length `T` from the corpus.
- Minimises MSE between predicted and target embeddings; extras follow the same scheduling controls.
- Optional perplexity estimates derive from cosine-similarity softmax over the embedding matrix (`ppx_temperature` controls sharpness).
- `curriculum_type="progressive_span"` limits episode start positions during warm-up to provide shorter contexts first.

Persistence: `save(path)` stores model weights, tokenizer vocab, and config. `load(path)` reconstructs the components and restores extras caches when present.

## 10. API pointers

Refer to `docs/API.md` for the full argument reference and method signatures used by the public API.
