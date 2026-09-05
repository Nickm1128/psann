# Geometric-sparse architectures

Use a geometric-sparse `ArchitectureConfig` with explicit geometry and activation policies. Connectivity can use local, random, or hash patterns; shape, neighborhood size, offsets, wrapping, seed, bias, and gather/scatter execution are validated.

```python
from psann import PSANNRegressor
from psann.architectures import ArchitectureConfig, ActivationConfig, GeometryConfig
model = PSANNRegressor(
    architecture=ArchitectureConfig.geometric_sparse(
        geometry=GeometryConfig(shape=(4, 4), k=8, seed=7),
        activation=ActivationConfig(kind="relu"),
    ),
    hidden_layers=4, epochs=20, random_state=7,
)
```

The geometry shape defines the feature lattice: its product must match the input width, or the preprocessor output width when preprocessing is composed. The example above accepts 16 features. Choose connectivity suitable for the experiment; changing it changes parameter count. Mixed activation policies specify feature ratios, layout, and seed. Supported fixed activations include GELU for this core kind.

[Example 28](../examples/28_geosparse_regression.py) provides supervised regression. [Geometry comparisons](../notebooks/geosparse_vs_relu_benchmarks.ipynb) include explicit parameter matching. LM geometry uses `LMArchitectureConfig.geometric_sparse(...)` and `LMGeometryExecutionConfig`; see [LM](lm.md). Historical reports retain their original measurements and are not current performance guarantees.
