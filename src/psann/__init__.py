"""PSANN: Parameterized Sine-Activated Neural Networks.

Top-level package exports the primary-output sklearn estimators, expanders,
episodic trainers, and diagnostic utilities."""

# Estimator surfaces
from .sklearn import PSANNRegressor, ResPSANNRegressor, ResConvPSANNRegressor

# Feature expanders and activation configs
from .lsm import LSM, LSMExpander, LSMConv2d, LSMConv2dExpander
from .activations import SineParam
from .types import ActivationConfig
from .state import StateConfig

# Episodic training and reward strategies
from .episodes import (
    EpisodeTrainer,
    EpisodeConfig,
    multiplicative_return_reward,
    portfolio_log_return_reward,
    make_episode_trainer_from_estimator,
)
from .rewards import (
    RewardStrategyBundle,
    FINANCE_PORTFOLIO_STRATEGY,
    get_reward_strategy,
    register_reward_strategy,
)
from .hisso import HISSOOptions, hisso_infer_series, hisso_evaluate_reward
# Language modelling utilities
from .tokenizer import SimpleWordTokenizer
from .embeddings import SineTokenEmbedder

# Initialisation helpers
from .initializers import apply_siren_init, siren_uniform_

# Core models and analysis helpers
from .models import WaveEncoder, WaveResNet, WaveRNNCell, build_wave_resnet, scan_regimes
from .utils import (
    encode_and_probe,
    fit_linear_probe,
    jacobian_spectrum,
    make_context_rotating_moons,
    make_drift_series,
    make_shock_series,
    make_regime_switch_ts,
    mutual_info_proxy,
    ntk_eigens,
    participation_ratio,
)

__all__ = [
    # Estimators
    "PSANNRegressor",
    "ResPSANNRegressor",
    "ResConvPSANNRegressor",
    # Expanders / activation configs
    "LSM",
    "LSMExpander",
    "LSMConv2d",
    "LSMConv2dExpander",
    "SineParam",
    "ActivationConfig",
    "StateConfig",
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
    # Language modelling
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

__version__ = "0.9.19"
