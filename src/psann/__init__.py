"""PSANN: Parameterized Sine-Activated Neural Networks.

Top-level package exports the primary-output sklearn estimators, expanders,
episodic trainers, and diagnostic utilities."""

# Estimator surfaces
from .activations import SineParam
from .embeddings import SineTokenEmbedder

# Episodic training and reward strategies
from .episodes import (
    EpisodeConfig,
    EpisodeTrainer,
    make_episode_trainer_from_estimator,
    multiplicative_return_reward,
    portfolio_log_return_reward,
)
from .hisso import HISSOOptions, hisso_evaluate_reward, hisso_infer_series

# Initialisation helpers
from .initializers import apply_siren_init, siren_uniform_

# Feature expanders and activation configs
from .lsm import LSM, LSMConv2d, LSMConv2dExpander, LSMExpander

# Core models and analysis helpers
from .models import WaveEncoder, WaveResNet, WaveRNNCell, build_wave_resnet, scan_regimes
from .rewards import (
    FINANCE_PORTFOLIO_STRATEGY,
    RewardStrategyBundle,
    get_reward_strategy,
    register_reward_strategy,
)
from .sklearn import PSANNRegressor, ResConvPSANNRegressor, ResPSANNRegressor, WaveResNetRegressor
from .state import StateConfig, StateController, ensure_state_config

# Token utilities
from .tokenizer import SimpleWordTokenizer
from .types import ActivationConfig
from .utils import (
    encode_and_probe,
    fit_linear_probe,
    jacobian_spectrum,
    make_context_rotating_moons,
    make_drift_series,
    make_regime_switch_ts,
    make_shock_series,
    mutual_info_proxy,
    ntk_eigens,
    participation_ratio,
)

__all__ = [
    # Estimators
    "PSANNRegressor",
    "ResPSANNRegressor",
    "ResConvPSANNRegressor",
    "WaveResNetRegressor",
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
]

__version__ = "0.10.5"
