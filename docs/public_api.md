# Public imports

Use explicit imports in applications. The canonical wildcard surfaces exclude deprecated estimator and LM base aliases. Direct compatibility imports are described only in [migration](migration.md).

| Import location | Canonical names |
| --- | --- |
| `psann` | `PSANNRegressor`, `PreprocessorConfig`, `EpisodicTrainer`, `HISSOConfig` |
| `psann.architectures` | `ArchitectureConfig`, `ActivationConfig`, `ResidualConfig`, `ConvolutionConfig`, `AttentionConfig`, `StateConfig`, `ContextConfig`, `WaveConfig`, `SpectralConfig`, `SequenceConfig`, `GeometryConfig`, `W0WarmupConfig`, `ProgressiveDepthConfig` |
| `psann.preprocessing` | `PreprocessorConfig`, `LSMConfig`, `LSMPretrainingConfig`, `PreprocessorTrainingConfig`, `ModulePreprocessorConfig` |
| `psann.episodic` | `EpisodicTrainer`, `HISSOConfig`, `EpisodeScheduleConfig`, `SupervisedWarmStartConfig` |
| `psannlm` | `PSANNLM`, `PSANNLMDataPrep`, `LMConfig`, `LMArchitectureConfig`, `TrainConfig`, `DataConfig`, `LMTrainer` |
| `psannlm.architectures` | `LMActivationInitializationConfig`, `LMTemporalConfig`, `LMGeometryExecutionConfig`, `build_lm_model`, `replace_lm_builder`, `available_lm_architectures` |

The regression estimator is also exported by `psann.estimators` and `psann.sklearn`. Import all nested architecture policies from `psann.architectures` so their immutable policy semantics are explicit.

Advanced PyTorch integrations may use `ArchitectureBuildRequest` and `build_architecture` from `psann.architectures`, or shared primitives from [architecture components](architecture_components.md). An architecture build returns a model, capabilities, and lifecycle; the estimator owns fit/inference and checkpoint orchestration. Modules beginning with an underscore are implementation details.

The core package never imports `psannlm`. LM imports shared core primitives through documented component boundaries. Both distributions carry their own package metadata and license; `psann.__version__` agrees with core distribution metadata.
