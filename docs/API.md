# Task API reference

## Regression

Import `PSANNRegressor` from `psann` and policies from `psann.architectures`. Construct with `architecture=ArchitectureConfig(...)` and optional `preprocessor=PreprocessorConfig(...)`. Tagged mappings and documented preset strings normalize to the same immutable configuration.

Training options remain estimator parameters: `hidden_layers`, `hidden_units`, `epochs`, `batch_size`, `lr`, `optimizer`, `loss`, `early_stopping`, `patience`, `random_state`, `device`, `amp`, and `compile`. `hidden_units` is the dense width; a convolution policy can specify channel width. Feature arrays should normally be float32. The architecture determines whether inputs are flattened, spatial, or sequential; see the [capability matrix](architecture_contract.md).

- `fit(X, y, validation_data=(X_val, y_val), context=..., verbose=...)` performs supervised training and returns the estimator. Supply context only for supported wave architectures.
- `predict(X, context=...)` returns primary predictions. `score(X, y)` computes regression R².
- `get_params(deep=True)` and `set_params(...)` support scikit-learn clone and model selection. Nested names include `architecture__activation__frequency_init` and `preprocessor__component__output_dim`. Updates validate transactionally; an invalid update leaves the previous configuration intact.
- `save(path)` writes schema-v3 core metadata and state. `PSANNRegressor.load(path, map_location="cpu")` reconstructs architecture, preprocessing, fitted shapes, and supported training/state metadata. CUDA device strings are supported when CUDA is available.

```python
from psann import PSANNRegressor
from psann.architectures import ArchitectureConfig, ActivationConfig, ResidualConfig
model = PSANNRegressor(
    architecture=ArchitectureConfig.dense(
        activation=ActivationConfig(kind="psann", frequency_init=0.5),
        residual=ResidualConfig(alpha_init=0.2),
    ),
    hidden_layers=3, hidden_units=32,
)
model.set_params(architecture__activation__frequency_init=0.75)
```

## Composed training

`PreprocessorConfig` owns the preprocessing component and its training policy; see [preprocessing](preprocessing.md). `EpisodicTrainer` owns episodic scheduling and reward configuration; see [episodic training](episodic.md). These compose with the same estimator and its persistence path.

## Language modeling

`PSANNLM(config=LMConfig(architecture=LMArchitectureConfig.wave(), ...))` owns an LM task. `PSANNLMDataPrep` prepares text and a tokenizer. Call `fit(data, train=TrainConfig(...))`, `generate(...)`, `save(path)`, and `PSANNLM.load(path, map_location=...)`. `LMTrainer` supports lower-level training and resume. [LM guide](lm.md) documents the four kinds, data preparation, CLI, and schema-v1 model/trainer checkpoints.

Configuration is strict: unknown fields and unsupported combinations fail early. See [migration](migration.md) for older input formats and checkpoint families.
