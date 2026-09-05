# Extending consumers and builders

## Add a benchmark or dataset

Keep data loading, model configuration, training budget, and evaluation metrics explicit. Use the canonical task APIs and tagged YAML mappings. Add the consumer to [consumer_manifest.json](consumer_manifest.json), declare any generated/local/external inputs, and include a bounded construction test. A complete workflow test must demonstrate parameter updates and real inference/persistence, rather than only finite values or checkpoint keys.

Use identical splits, tokenizer identity, seeds, and evaluation token counts when the comparison requires them. Record exact resolved settings. If an experiment cannot be reproduced from public code/data, state its historical scope or remove the executable claim.

## Replace an LM builder

`replace_lm_builder` from `psannlm.architectures` replaces one existing typed kind. The four supported kinds are transformer, residual, wave, and geometric-sparse. The builder receives an immutable `LMBuildRequest`; its result must honor validated configuration, causal logits, model metadata, and capabilities. `available_lm_architectures()` reports supported names. Registration does not make an unknown typed kind valid.

Reuse documented numerical components from `psann.architectures.components`. Do not import private estimator implementation modules into LM code or add a core dependency on `psannlm`. Test replacement through real training, generation, and two checkpoint generations. A replacement must preserve the advertised contract, including early rejection of invalid policies.

This executable example obtains the existing residual implementation through the canonical factory and registers a typed builder for that implementation. An alternative implementation must honor the same forward, cache, configuration, and training contracts.

```python
from psannlm.architectures import (
    LMConfig, LMBuildRequest, LMBuildResult, LMCapabilities,
    build_lm_model, replace_lm_builder,
)

ResidualImplementation = type(build_lm_model(LMConfig(
    architecture="residual", vocab_size=32, d_model=16,
    n_layers=1, n_heads=2, d_mlp=32,
)).model)

def residual_builder(request: LMBuildRequest) -> LMBuildResult:
    model = ResidualImplementation(request.config)
    return LMBuildResult(model, LMCapabilities(kind="residual"))

replace_lm_builder("residual", residual_builder)
```

Replacement is explicit and process-local. Register the same implementation before loading its checkpoints: the saved configuration stores policies, not Python code.

Core integrations use `ArchitectureBuildRequest`, `build_architecture`, and documented registry/lifecycle interfaces from `psann.architectures`; keep task fit and persistence orchestration in the estimator. See [architecture](architecture.md) and [component reference](architecture_components.md).
