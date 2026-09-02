# Architecture Contract

This document records the current estimator behavior and the approved Phase 3 target.
It is a contract for compatibility work; it does not mean the target architecture
normalizer, registry, or unified estimator has already been implemented.

## Canonical estimator target

Phase 3 makes `PSANNRegressor(architecture: ArchitectureLike = "dense",
hidden_layers: int = 2, hidden_units: int = 64, ...)` the only canonical estimator.
`ArchitectureLike` is an `ArchitectureConfig`, tagged mapping, or string. Existing
optimization, loss, scaling, output, device, AMP, compile, and warm-start arguments
remain flat. The exact flat groups are structure (`hidden_layers`, `hidden_units`,
`output_shape`); optimization (`epochs`, `batch_size`, `lr`, `optimizer`,
`weight_decay`, `early_stopping`, `patience`, `num_workers`, `warm_start`); objective
(`loss`, `loss_params`, `loss_reduction`); scaling (`scaler`, `scaler_params`,
`target_scaler`, `target_scaler_params`); and execution (`device`, `random_state`,
`amp`, `amp_dtype`, `compile`, `compile_backend`, `compile_mode`,
`compile_fullgraph`, `compile_dynamic`).

Old architecture-specific flat arguments remain a deprecated 0.x compatibility
adapter. Supplying an explicit `architecture` and an old architecture option is an
error that names both paths. `hidden_width` remains the deprecated alias of
`hidden_units`. Phase 4 moves the temporary LSM knobs to `preprocessor`; Phase 5 moves
HISSO fit flags to episodic composition.

## Approved typed policy target

Phase 3 adds frozen policy dataclasses under `psann.architectures`; mapping-valued
fields are defensively copied/frozen during normalization. The target objects are:

| Policy | Target fields and defaults |
| --- | --- |
| `ActivationConfig` | `kind="psann"`, amplitude/frequency/decay initializers `1.0/1.0/0.1`, `learnable=("amplitude", "frequency", "decay")`, `decay_mode="abs"`, optional bounds/types/ratios, `slope_init=1.0`, `slope_trainable=True`, `clip_max=1.0` |
| `ResidualConfig` | `norm="rms"`, `alpha_init=0.0`, `drop_path=0.0`, `first_w0=12.0`, `hidden_w0=1.0` |
| `ConvolutionConfig` | `channels=None`, `kernel_size=1`, `data_format="channels_first"`, `per_element=False` |
| `AttentionConfig` | `kind="mha"`, `num_heads=4`, `dropout=0.0`, `bias=True`, `batch_first=True`, `add_bias_kv=False`, `add_zero_attn=False` |
| `StateConfig` | `rho=0.95`, `beta=1.0`, `init=1.0`, `max_abs=5.0`, `detach=True`, `reset="batch"`, `stream_lr=None` |
| `ContextConfig` | `dim=None`, `builder=None`, `builder_params=None`, `film=True`, `phase_shift=True` |
| `WaveConfig` | `first_w0=30.0`, `hidden_w0=1.0`, `norm="none"`, `dropout=0.0`, `grad_clip_norm=5.0`, optional warmup/progressive depth |
| `W0WarmupConfig` | `first_initial=10.0`, `hidden_initial=0.5`, `epochs=10` |
| `ProgressiveDepthConfig` | `initial_layers`, `interval=15`, `growth=1` |
| `SpectralConfig` | `k_fft=64`, `gate_type="rfft"`, `groups="depthwise"`, `init=0.0`, `strength=1.0` |
| `SequenceConfig` | `phase_init=0.0`, `phase_trainable=True`, `pool="last"` |
| `GeometryConfig` | `shape=None`, `k=8`, `pattern="local"`, `radius=1`, `offsets=None`, `wrap_mode="clamp"`, `bias=True`, `compute_mode="gather"`, `seed=None` |

`ArchitectureConfig(kind, activation=ActivationConfig(), residual=None,
convolution=None, attention=None, state=None, context=None, wave=None,
spectral=None, sequence=None, geometry=None)` has `dense`, `convolutional`, `wave`,
`sequence`, and `geometric_sparse` constructors. Activation kinds initially are
`psann`, `relu`, `tanh`, and `relu-sigmoid-psann`; mixed/phase variants remain gated.
Presence enables residuals, convolution, attention, state, context, wave, spectral,
sequence, and geometry as applicable. Compatibility-only `attention.kind="none"`
canonicalizes to absent attention.

Constructor validation preserves positive widths/kernel/W0 values, probability bounds,
recognized norm/gate/pool/data-format values, positive context dimensions, progressive
depth no greater than `hidden_layers`, non-negative spectral strength, and valid
geometry shape/connectivity. Cross-field depth validation runs at estimator validation.

## Normalization and capability target

Normalization strips outer whitespace, lowercases, and treats hyphens and underscores
as equivalent separators; arbitrary punctuation and internal spaces are not removed.
Canonical strings do not warn. The legacy aliases below issue `DeprecationWarning` at
the caller.

| Input | Canonical target |
| --- | --- |
| `dense`, `residual`, `convolutional`, `residual-convolutional`, `wave`, `sequence`, `geometric-sparse` | Corresponding canonical preset; residual/convolution/wave/sequence/geometry required policies are inserted as applicable |
| `psann` | deprecated `dense` |
| `respsann`, `res-psann`, `res_psann` | deprecated `residual` |
| `resconvpsann`, `res-conv-psann`, `res_conv_psann` | deprecated `residual-convolutional` |
| `waveresnet`, `wave-resnet`, `wave_resnet` | deprecated `wave` |
| `sgrpsann`, `sgr-psann`, `sgr_psann` | deprecated `sequence` |
| `geosparse`, `geo-sparse`, `geo_sparse` | deprecated `geometric-sparse` |

The normalizer's concrete input matrix is intentionally narrow: outer whitespace,
case, and hyphen/underscore separators are equivalent; no other punctuation or
internal whitespace is normalized.

| Input representation | Result |
| --- | --- |
| `"dense"`, `" DENSE "`, `"geometric-sparse"`, `"geometric_sparse"` | Accepted canonical presets with no warning. |
| `"psann"`, `"res-psann"`, `"res_psann"`, `"wave-resnet"`, `"wave_resnet"` | Accepted deprecated aliases; each warns and maps to its documented canonical preset. |
| Typed `ArchitectureConfig(kind="dense")` and `{kind: "dense"}` | Accepted and equal after normalization. |
| `{kind: "dense", residual: {norm: "rms"}}` | Accepted tagged mapping with a valid nested policy. |
| `"dense!"`, `"dense residual"` | Rejected: punctuation and internal spaces are not aliases. |
| `{kind: "unknown"}` or `{kind: "dense", extra: 1}` | Rejected with a path-specific `ValueError`. |
| `{kind: "dense", residual: "rms"}` | Rejected with a path-specific `TypeError`. |
| Explicit architecture plus a legacy architecture-specific flat argument | Rejected and names both conflicting paths. |
| One invalid policy combination for each kind (for example sequence plus attention) | Rejected before fit. |

A tagged mapping uses `ArchitectureConfig` field names, for example
`{kind: dense, residual: {norm: rms, alpha_init: 0.0, drop_path: 0.1}}`. Typed,
mapping, and string inputs share one normalizer. Unknown top-level/nested keys raise a
path-specific `ValueError`; wrong nested types raise a path-specific `TypeError`; no
extra mapping key reaches a builder; equal representations compare equal.

| Kind | Residual | Convolution | Attention | State | Context | Spectral | Required policy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `dense` | optional | invalid | optional | optional | invalid | invalid | none |
| `convolutional` | optional | required | optional | invalid | invalid | invalid | `ConvolutionConfig` |
| `wave` | required | optional | optional only without spectral | invalid | optional | optional only without attention | `WaveConfig` |
| `sequence` | invalid | invalid | invalid | invalid | invalid | optional | `SequenceConfig` |
| `geometric-sparse` | required | invalid | invalid | invalid | invalid | invalid | `GeometryConfig` |

Every kind accepts `ActivationConfig` subject to builder validation; sequence initially
requires PSANN activation. Preprocessor and episodic compatibility are Phase 4/5
capability work, not class checks.

The Phase 3 executable matrix must accept canonical case/separator variants and every
legacy alias family, plus one valid combination per kind. It must reject an unknown
preset, unknown nested key, wrong nested type, explicit architecture/legacy conflict,
and one invalid policy combination per kind.

## sklearn, builder, and checkpoint target

`get_params(deep=False)` returns immutable `architecture` plus flat parameters;
`deep=True` adds paths such as `architecture__residual__norm`. Nested `set_params`
reconstructs frozen parents, resets fitted/build caches, and fails helpfully for an
absent optional policy or unknown path. Clone, grid search, joblib, pickle, and legacy
wrapper configuration preserve normalized architecture equality.

Registry entries consume `ArchitectureBuildRequest`, return an
`ArchitectureBuildResult` with `ArchitectureCapabilities` and
`ArchitectureLifecycle`, and expose `on_model_built`, `on_fit_start`,
`before_optimizer_step`, `on_epoch_end`, and `on_fit_end`. Wave warmup/progressive
depth moves to lifecycle hooks.

New saves target schema `psann.regressor`, version `1`, package version, canonical
parameters/tagged architecture, fitted metadata, model state, and progressive-depth
structure. Phase 2 retains the current unversioned whole-model payload and does not
rewrite checkpoints. Legacy class migration is PSANN→dense/convolutional,
ResPSANN→dense+residual, ResConv→convolutional+residual, Wave→wave,
SGR→sequence, and GeoSparse→geometric-sparse. Incompatible explicit loader/class
requests remain errors.

## Current Phase 2 characterization

The six current public classes remain direct imports: `PSANNRegressor`,
`ResPSANNRegressor`, `ResConvPSANNRegressor`, `WaveResNetRegressor`,
`SGRPSANNRegressor`, and `GeoSparseRegressor`. Current shared fit/scaling/inference
and serialization remain in `psann._sklearn`; the separate `psannlm` distribution
retains its own registry names (`respsann`, `sgrpsann`, `waveresnet`, `geosparse`) and
is characterization-only in this phase.

Phase 2 fixes only legacy parameter/checkpoint drift: fallback `get_params` follows
constructor signatures, GeoSparse uses public constructor names, default unsupported
inherited payload keys are filtered only for the documented affected classes, and
non-default or unknown values fail rather than being discarded. CPU map-location loads
remain on CPU; CUDA map-location coverage verifies the converse. This is not the Phase
3 checkpoint schema or architecture implementation.
