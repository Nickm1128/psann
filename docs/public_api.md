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
- `HISSOOptions`
- `EpisodeTrainer`, `EpisodeConfig`
- `hisso_infer_series`, `hisso_evaluate_reward`
- `get_reward_strategy`, `register_reward_strategy`
- `RewardStrategyBundle`, `FINANCE_PORTFOLIO_STRATEGY`
- `multiplicative_return_reward`, `portfolio_log_return_reward`
- `make_episode_trainer_from_estimator`

### Expanders and activation config
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
