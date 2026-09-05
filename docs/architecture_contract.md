# Architecture contract

A single `PSANNRegressor` normalizes a typed `ArchitectureConfig`, a tagged mapping with `kind`, or a supported preset string before construction. Typed policies are the primary interface. Configurations are immutable and nested updates are validated before replacing them.

## Core capabilities

| Core kind | Input topology | Residual | Attention | State | Context | Preprocessing |
| --- | --- | --- | --- | --- | --- | --- |
| `dense` | Flat or token grid | Optional | Yes | Nonresidual only | No | Yes |
| `convolutional` | 1D/2D/3D spatial | 2D only | Yes | No | No | Yes |
| `wave` | Flat/token grid or convolutional composition | Required policy | Yes | No | Dense wave only | Yes |
| `sequence` | Sequence | Built in | No | No | No | Yes |
| `geometric-sparse` | Flat | Required policy | No | No | No | Yes |

Use `ArchitectureConfig.dense()`, `.convolutional()`, `.for_wave()`, `.for_sequence()`, or `.geometric_sparse()`. A residual dense model is `.dense(residual=ResidualConfig(...))`. `ConvolutionConfig` controls `channels`, `kernel_size`, `data_format`, and `per_element`. Per-element prediction is available for supported spatial compositions, not a dense estimator option.

Wave attention and spectral gating cannot be combined. Context is unavailable for convolutional wave, and state cannot be combined with residual cores. Sequence architecture uses its sequence and spectral policies; it does not accept independent attention/state/context policies. Geometric-sparse shape/connectivity, residual behavior, and activation are explicit policies. Invalid topology, geometry, head divisibility, depth schedules, unknown fields, and unsupported policy combinations reject before incompatible numerical execution. Input-dependent checks occur when input shape is known.

## Language-model capabilities

`LMArchitectureConfig` has exactly four typed kinds: `transformer`, `residual`, `wave`, and `geometric-sparse`. Spectral residual is `LMArchitectureConfig.residual(spectral=SpectralConfig(...))`, not a fifth kind. All four share `LMConfig` dimensions and causal LM outputs.

- Transformer uses its supported fixed activation policy.
- Residual supports sinusoidal activation initialization, residual settings, and optional spectral gating.
- Wave supports residual settings and `LMTemporalConfig` modes `disabled`, `interleave`, `replace`, and `attention-only`.
- Geometric-sparse supports geometry, execution depth/chunking, residual settings, and fixed or mixed activations.

Unsupported policy combinations and unknown keys reject; registry replacement does not introduce additional advertised typed names. Replacement builders for the four existing kinds receive the validated `LMBuildRequest`. See [components](architecture_components.md) for shared numerical primitives.

## Persistence boundary

Core schema-v3 checkpoints reconstruct architecture, preprocessing, context descriptors, fitted shapes, weights, and supported episodic/state metadata. LM schema-v1 model and trainer checkpoints reconstruct typed configuration, tokenizer state where embedded, weights, and trainer resume state where saved. Both loaders support CPU/CUDA map location. Two consecutive save/load generations must preserve predictions or greedy tokens and configuration. Custom modules/rewards/context callables require their documented registration or importability conditions; raw weights alone cannot recover missing architecture or tokenizer identity. See [migration](migration.md).
