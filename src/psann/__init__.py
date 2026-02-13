"""PSANN: Parameterized Sine-Activated Neural Networks.

Top-level package exports the primary-output sklearn estimators, expanders,
episodic trainers, and diagnostic utilities."""

from __future__ import annotations

import importlib

__all__ = [
    # Estimators
    "AttentionConfig",
    "PSANNRegressor",
    "ResPSANNRegressor",
    "ResConvPSANNRegressor",
    "SGRPSANNRegressor",
    "WaveResNetRegressor",
    "GeoSparseRegressor",
    # Expanders / activation configs
    "LSM",
    "LSMExpander",
    "LSMConv2d",
    "LSMConv2dExpander",
    "SineParam",
    "ActivationConfig",
    "StateConfig",
    "StateController",
    "ensure_state_config",
    # Episodic training & rewards
    "EpisodeTrainer",
    "EpisodeConfig",
    "multiplicative_return_reward",
    "portfolio_log_return_reward",
    "make_episode_trainer_from_estimator",
    "HISSOOptions",
    "hisso_infer_series",
    "hisso_evaluate_reward",
    "RewardStrategyBundle",
    "FINANCE_PORTFOLIO_STRATEGY",
    "get_reward_strategy",
    "register_reward_strategy",
    # Token utilities
    "SimpleWordTokenizer",
    "SineTokenEmbedder",
    # Initialisation helpers
    "apply_siren_init",
    "siren_uniform_",
    # Core models
    "WaveResNet",
    "build_wave_resnet",
    "WaveEncoder",
    "WaveRNNCell",
    "scan_regimes",
    # Analysis utilities
    "jacobian_spectrum",
    "ntk_eigens",
    "participation_ratio",
    "mutual_info_proxy",
    "fit_linear_probe",
    "encode_and_probe",
    "make_context_rotating_moons",
    "make_drift_series",
    "make_shock_series",
    "make_regime_switch_ts",
    # Param utilities
    "count_params",
    "dense_mlp_params",
    "geo_sparse_net_params",
    "match_dense_width",
]

_LAZY_ATTRS = {
    # Estimator surfaces
    "AttentionConfig": ".attention",
    "PSANNRegressor": ".sklearn",
    "ResPSANNRegressor": ".sklearn",
    "ResConvPSANNRegressor": ".sklearn",
    "SGRPSANNRegressor": ".sklearn",
    "WaveResNetRegressor": ".sklearn",
    "GeoSparseRegressor": ".sklearn",
    # Expanders / activation configs
    "LSM": ".lsm",
    "LSMExpander": ".lsm",
    "LSMConv2d": ".lsm",
    "LSMConv2dExpander": ".lsm",
    "SineParam": ".activations",
    "ActivationConfig": ".types",
    "StateConfig": ".state",
    "StateController": ".state",
    "ensure_state_config": ".state",
    # Episodic training & rewards
    "EpisodeTrainer": ".episodes",
    "EpisodeConfig": ".episodes",
    "multiplicative_return_reward": ".episodes",
    "portfolio_log_return_reward": ".episodes",
    "make_episode_trainer_from_estimator": ".episodes",
    "HISSOOptions": ".hisso",
    "hisso_infer_series": ".hisso",
    "hisso_evaluate_reward": ".hisso",
    "RewardStrategyBundle": ".rewards",
    "FINANCE_PORTFOLIO_STRATEGY": ".rewards",
    "get_reward_strategy": ".rewards",
    "register_reward_strategy": ".rewards",
    # Token utilities
    "SimpleWordTokenizer": ".tokenizer",
    "SineTokenEmbedder": ".embeddings",
    # Initialisation helpers
    "apply_siren_init": ".initializers",
    "siren_uniform_": ".initializers",
    # Core models
    "WaveResNet": ".models",
    "build_wave_resnet": ".models",
    "WaveEncoder": ".models",
    "WaveRNNCell": ".models",
    "scan_regimes": ".models",
    # Analysis utilities
    "jacobian_spectrum": ".utils",
    "ntk_eigens": ".utils",
    "participation_ratio": ".utils",
    "mutual_info_proxy": ".utils",
    "fit_linear_probe": ".utils",
    "encode_and_probe": ".utils",
    "make_context_rotating_moons": ".utils",
    "make_drift_series": ".utils",
    "make_shock_series": ".utils",
    "make_regime_switch_ts": ".utils",
    # Param utilities
    "count_params": ".params",
    "dense_mlp_params": ".params",
    "geo_sparse_net_params": ".params",
    "match_dense_width": ".params",
}


def __getattr__(name: str):
    module = _LAZY_ATTRS.get(name)
    if module is not None:
        try:
            mod = importlib.import_module(module, __name__)
        except Exception as exc:  # pragma: no cover - import-time optional deps
            raise ImportError(
                f"psann.{name} could not be imported from {module} (optional dependencies may be missing). "
                "Try installing extras like 'psann[sklearn]' or install 'psannlm' for LM tooling."
            ) from exc
        attr = getattr(mod, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY_ATTRS.keys()))


__version__ = "0.12.3"