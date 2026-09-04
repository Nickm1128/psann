# Shared architecture components

Model packages can use the documented `psann.architectures.components` interface
without importing core implementation modules. Core remains independent of optional
model packages.

`build_activation(config, *, features)` accepts an `ActivationConfig` or equivalent
mapping for PSANN, phase PSANN, mixed activations, ReLU, tanh, and ReLU-sigmoid PSANN.
It delegates to the existing core activation builder, including parameter bounds,
learnability, phase, mixed feature allocation, seed, and layout. A fixed `"gelu"`
literal or `{"kind": "gelu"}` builds PyTorch GELU. The current `ActivationConfig`
does not include a GELU kind.

`RMSNorm` normalizes the last feature dimension with the established default epsilon
of `1e-6` and a learned `scale`. `DropPath` retains the per-sample mask and inverse
keep-probability scaling. `SpectralGate1D` retains its existing constructor, Fourier
implementations, parameters, and state keys. These three names directly export the
existing classes.

`build_geometry_connectivity(GeometryConfig(...), *, features=None)` returns an
immutable `GeometryConnectivity`. Supply a geometry shape or a feature count. When
only a feature count is given, it chooses the closest rectangular factorization.
If both are supplied, the shape must multiply to that count. Connectivity uses the
existing local, random, or hash algorithm with the configured radius, offsets,
wrapping, and seed. Bias and gather/scatter selection remain execution policies.

`GeometryConnectivity.shape` and `.indices` are nested tuples. Call
`.as_tensors(device="cpu")` to obtain `(indices, source_indices, destination_indices)`
as independent `torch.long` tensors. Mutating these tensors does not change the
connectivity object or later materializations.

Cache maintenance is available through the documented
`psann.utils.cleanup_hf_cache(max_bytes, ...)` export. It retains the existing
cache-pruning behavior and keyword arguments.
