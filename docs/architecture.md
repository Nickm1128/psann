# Architecture and package boundaries

`PSANNRegressor` is the task facade. It normalizes `ArchitectureConfig` and `PreprocessorConfig`, validates capabilities, delegates numerical construction to the architecture registry, and orchestrates supervised fit, prediction, and persistence. `EpisodicTrainer` composes the estimator with `HISSOConfig` and owns episode scheduling, reward dispatch, warm start, and evaluation.

`psann.architectures.components` provides the documented activation, normalization, spectral, and connectivity primitives used across packages. Architecture builders return a model plus lifecycle/capabilities. Wave lifecycle hooks handle initialization warmup and progressive depth without moving estimator concerns into a numerical backbone.

`psannlm` is a separate package. Its immutable `LMConfig` and four-kind `LMArchitectureConfig` feed one builder path used by `PSANNLM`, training, evaluation, persistence, and CLI consumers. Core code has no dependency on LM code. The LM distribution declares its own direct runtime dependencies.

See the [capability contract](architecture_contract.md), [public import table](public_api.md), [shared components](architecture_components.md), and [repository map](PROJECT_MAP.md). Checkpoint migration behavior is specified in [migration](migration.md).
