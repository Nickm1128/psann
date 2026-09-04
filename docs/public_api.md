# Public API Surface

This page lists the **supported** public API surface for PSANN. Anything not listed here should be treated as internal or experimental.

## Top-level imports (stable)

These are the identifiers exported from `psann.__init__` and are safe to import directly:

### Estimators
- `PSANNRegressor`, configured with immutable policies from `psann.architectures`.
- The historical variant estimator names remain direct deprecated imports through
  the 0.x line; use `ArchitectureConfig.dense`, `.convolutional`, `.for_wave`,
  `.for_sequence`, or `.geometric_sparse` for new code.

### Episodic training (HISSO)
- `EpisodicTrainer`, `HISSOConfig`, `EpisodeScheduleConfig`, and
  `SupervisedWarmStartConfig` from `psann.episodic`.
- `RewardStrategy`, `RewardStrategyBundle`, `get_reward_strategy`, and
  `register_reward_strategy` from `psann.episodic`.
- The 0.x `HISSOOptions`, `EpisodeTrainer`, `hisso_infer_series`, and related
  episode/HISSO helpers remain explicit deprecated compatibility imports.

### Expanders and activation config
- `PreprocessorConfig`, `LSMConfig`, `LSMPretrainingConfig`,
  `PreprocessorTrainingConfig`, `ModulePreprocessorConfig` from
  `psann.preprocessing` (also convenient top-level imports)
- `LSM`, `LSMExpander`
- `LSMConv2d`, `LSMConv2dExpander`
- `SineParam`, `ActivationConfig`
- `StateConfig`, `StateController`, `ensure_state_config`

### Token utilities
- `SimpleWordTokenizer`
- `SineTokenEmbedder`

### Core wave backbones
- `WaveResNet`, `WaveEncoder`, `WaveRNNCell`
- `build_wave_resnet`
- `scan_regimes`

### Diagnostics and synthetic data
- `jacobian_spectrum`, `ntk_eigens`, `participation_ratio`, `mutual_info_proxy`
- `encode_and_probe`, `fit_linear_probe`
- `make_context_rotating_moons`, `make_drift_series`, `make_shock_series`, `make_regime_switch_ts`

### Parameter counting helpers
- `count_params`, `dense_mlp_params`, `geo_sparse_net_params`, `match_dense_width`

## Experimental APIs

These are available but may change without notice:

- `GeoSparseRegressor` (deprecated compatibility wrapper; use
  `PSANNRegressor(architecture=ArchitectureConfig.geometric_sparse(...))`).
- `psannlm` (LM utilities; packaged separately from the core `psann` distribution).

## Internal-only modules (not stable)

The following modules are **internal** implementation details:

- `psann.estimators._fit_utils`
- `psann.layers.*`
- `psann.nn`, `psann.nn_geo_sparse`
- `psann.utils.hf_cache`

If you must rely on internal modules, pin a version and expect breaking changes.
