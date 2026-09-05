# Wave architectures and context

Use `PSANNRegressor(architecture=ArchitectureConfig.for_wave(...))`. `WaveConfig` controls first/hidden frequency scale, normalization, dropout, gradient clipping, optional `W0WarmupConfig`, and `ProgressiveDepthConfig`. `ResidualConfig` supplies residual scaling. Context is a separate nested policy.

```python
from psann import PSANNRegressor
from psann.architectures import ArchitectureConfig, ContextConfig, WaveConfig
model = PSANNRegressor(
    architecture=ArchitectureConfig.for_wave(
        wave=WaveConfig(first_w0=20.0, hidden_w0=1.0),
        context=ContextConfig(dim=2),
    ),
    hidden_layers=4, hidden_units=64,
)
```

Provide `context=` to fit/predict when using explicit context. A registered builder such as `ContextConfig(builder="cosine", builder_params={"frequencies": [1.0, 2.0]})` constructs context from inputs. See the [context notebook](../notebooks/PSANN_WaveResNet_Context_Demo.ipynb) for both forms.

Attention and spectral gating are alternative wave policies and cannot be combined. Context is supported for dense wave, not convolutional wave. Progressive depth must start within the configured hidden-layer count. Checkpoints preserve the active depth and supported context descriptor. See the [capability contract](architecture_contract.md). Language-model wave policies use `LMArchitectureConfig.wave()` and [LM temporal settings](lm.md).
