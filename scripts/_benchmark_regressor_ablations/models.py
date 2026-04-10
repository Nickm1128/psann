# ruff: noqa: F403,F405
from __future__ import annotations

from .shared import *


@dataclass
class ModelSpec:
    name: str
    estimator: Callable[..., Any]
    params: Dict[str, Any]


MODELS: Dict[str, ModelSpec] = {
    # ResPSANN ablations
    "res_base": ModelSpec(
        name="res_base",
        estimator=ResPSANNRegressor,
        params={"hidden_layers": 4, "hidden_units": 64, "norm": "rms", "drop_path_max": 0.0},
    ),
    "res_relu_sigmoid_psann": ModelSpec(
        name="res_relu_sigmoid_psann",
        estimator=ResPSANNRegressor,
        params={
            "hidden_layers": 4,
            "hidden_units": 64,
            "norm": "rms",
            "drop_path_max": 0.0,
            "activation_type": "relu_sigmoid_psann",
            "activation": {"slope_init": 1.0, "clip_max": 1.0},
        },
    ),
    "res_drop_path": ModelSpec(
        name="res_drop_path",
        estimator=ResPSANNRegressor,
        params={"hidden_layers": 4, "hidden_units": 64, "norm": "rms", "drop_path_max": 0.1},
    ),
    "res_no_norm": ModelSpec(
        name="res_no_norm",
        estimator=ResPSANNRegressor,
        params={"hidden_layers": 4, "hidden_units": 64, "norm": "none", "drop_path_max": 0.0},
    ),
    # WaveResNet ablations
    "wrn_base": ModelSpec(
        name="wrn_base",
        estimator=WaveResNetRegressor,
        params={
            "hidden_layers": 6,
            "hidden_units": 64,
            "norm": "rms",
            "use_film": True,
            "use_phase_shift": True,
        },
    ),
    "wrn_no_phase": ModelSpec(
        name="wrn_no_phase",
        estimator=WaveResNetRegressor,
        params={
            "hidden_layers": 6,
            "hidden_units": 64,
            "norm": "rms",
            "use_film": True,
            "use_phase_shift": False,
        },
    ),
    "wrn_no_film": ModelSpec(
        name="wrn_no_film",
        estimator=WaveResNetRegressor,
        params={
            "hidden_layers": 6,
            "hidden_units": 64,
            "norm": "rms",
            "use_film": False,
            "use_phase_shift": True,
        },
    ),
    "wrn_spec_gate_rfft": ModelSpec(
        name="wrn_spec_gate_rfft",
        estimator=WaveResNetRegressor,
        params={
            "hidden_layers": 6,
            "hidden_units": 64,
            "norm": "rms",
            "use_spectral_gate": True,
            "k_fft": 64,
            "gate_type": "rfft",
            "gate_groups": "depthwise",
            "gate_strength": 1.0,
        },
    ),
    "wrn_spec_gate_feats": ModelSpec(
        name="wrn_spec_gate_feats",
        estimator=WaveResNetRegressor,
        params={
            "hidden_layers": 6,
            "hidden_units": 64,
            "norm": "rms",
            "use_spectral_gate": True,
            "k_fft": 64,
            "gate_type": "fourier_features",
            "gate_groups": "depthwise",
            "gate_strength": 1.0,
        },
    ),
    # SGR-PSANN ablations
    "sgr_base": ModelSpec(
        name="sgr_base",
        estimator=SGRPSANNRegressor,
        params={
            "hidden_layers": 3,
            "hidden_units": 64,
            "k_fft": 64,
            "gate_type": "rfft",
            "use_spectral_gate": True,
            "phase_trainable": True,
        },
    ),
    "sgr_no_gate": ModelSpec(
        name="sgr_no_gate",
        estimator=SGRPSANNRegressor,
        params={
            "hidden_layers": 3,
            "hidden_units": 64,
            "use_spectral_gate": False,
            "phase_trainable": True,
        },
    ),
    "sgr_fourier_feats": ModelSpec(
        name="sgr_fourier_feats",
        estimator=SGRPSANNRegressor,
        params={
            "hidden_layers": 3,
            "hidden_units": 64,
            "k_fft": 64,
            "gate_type": "fourier_features",
            "use_spectral_gate": True,
            "phase_trainable": True,
        },
    ),
    "sgr_no_phase": ModelSpec(
        name="sgr_no_phase",
        estimator=SGRPSANNRegressor,
        params={
            "hidden_layers": 3,
            "hidden_units": 64,
            "k_fft": 64,
            "gate_type": "rfft",
            "use_spectral_gate": True,
            "phase_trainable": False,
        },
    ),
}
