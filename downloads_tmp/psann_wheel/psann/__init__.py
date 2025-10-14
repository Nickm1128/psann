"""PSANN: Parameterized Sine-Activated Neural Networks.

Sklearn-style estimators powered by PyTorch.
"""

from .sklearn import PSANNRegressor, ResPSANNRegressor, ResConvPSANNRegressor
from .lsm import LSM, LSMExpander, LSMConv2d, LSMConv2dExpander
from .activations import SineParam
from .types import ActivationConfig
from .episodes import EpisodeTrainer, EpisodeConfig, portfolio_log_return_reward, make_episode_trainer_from_estimator
from .augmented import PredictiveExtrasTrainer, PredictiveExtrasConfig, make_predictive_extras_trainer_from_estimator
from .extras import SupervisedExtrasConfig, ensure_supervised_extras_config, rollout_supervised_extras, ExtrasGrowthConfig, ensure_extras_growth_config, extras_growth_to_metadata, expand_extras_head
from .tokenizer import SimpleWordTokenizer
from .embeddings import SineTokenEmbedder
from .lm import PSANNLanguageModel, LMConfig
from .initializers import apply_siren_init, siren_uniform_
from .models import WaveEncoder, WaveResNet, WaveRNNCell, build_wave_resnet, scan_regimes
from .utils import (
    encode_and_probe,
    fit_linear_probe,
    jacobian_spectrum,
    make_context_rotating_moons,
    make_regime_switch_ts,
    mutual_info_proxy,
    ntk_eigens,
    participation_ratio,
)

__all__ = [
    "PSANNRegressor",
    "ResPSANNRegressor",
    "ResConvPSANNRegressor",
    "LSM",
    "LSMExpander",
    "LSMConv2d",
    "LSMConv2dExpander",
    "SineParam",
    "ActivationConfig",
    "EpisodeTrainer",
    "EpisodeConfig",
    "portfolio_log_return_reward",
    "make_episode_trainer_from_estimator",
    "PredictiveExtrasTrainer",
    "PredictiveExtrasConfig",
    "make_predictive_extras_trainer_from_estimator",
    "SupervisedExtrasConfig",
    "ensure_supervised_extras_config",
    "rollout_supervised_extras",
    "ExtrasGrowthConfig",
    "ensure_extras_growth_config",
    "extras_growth_to_metadata",
    "expand_extras_head",
    "SimpleWordTokenizer",
    "SineTokenEmbedder",
    "PSANNLanguageModel",
    "LMConfig",
    "apply_siren_init",
    "siren_uniform_",
    "WaveResNet",
    "build_wave_resnet",
    "WaveEncoder",
    "WaveRNNCell",
    "scan_regimes",
    "jacobian_spectrum",
    "ntk_eigens",
    "participation_ratio",
    "mutual_info_proxy",
    "fit_linear_probe",
    "encode_and_probe",
    "make_context_rotating_moons",
    "make_regime_switch_ts",
]

__version__ = "0.9.19"
