# ruff: noqa: F403,F405
from __future__ import annotations

from .shared import *


@dataclass
class ModelSpec:
    name: str
    architecture: ArchitectureConfig
    params: Dict[str, Any]

    def build(self, **params: Any) -> PSANNRegressor:
        merged = dict(self.params)
        merged.update(params)
        return PSANNRegressor(architecture=self.architecture, **merged)


MODELS: Dict[str, ModelSpec] = {
    # ResPSANN ablations
    "res_base": ModelSpec(
        name="res_base",
        architecture=ArchitectureConfig.dense(residual=ResidualConfig(norm="rms")),
        params={"hidden_layers": 4, "hidden_units": 64},
    ),
    "res_relu_sigmoid_psann": ModelSpec(
        name="res_relu_sigmoid_psann",
        architecture=ArchitectureConfig.dense(
            activation=ActivationConfig(kind="relu-sigmoid-psann", slope_init=1.0, clip_max=1.0),
            residual=ResidualConfig(norm="rms"),
        ),
        params={
            "hidden_layers": 4,
            "hidden_units": 64,
        },
    ),
    "res_drop_path": ModelSpec(
        name="res_drop_path",
        architecture=ArchitectureConfig.dense(residual=ResidualConfig(norm="rms", drop_path=0.1)),
        params={"hidden_layers": 4, "hidden_units": 64},
    ),
    "res_no_norm": ModelSpec(
        name="res_no_norm",
        architecture=ArchitectureConfig.dense(residual=ResidualConfig(norm="none")),
        params={"hidden_layers": 4, "hidden_units": 64},
    ),
    # WaveResNet ablations
    "wrn_base": ModelSpec(
        name="wrn_base",
        architecture=ArchitectureConfig.for_wave(
            wave=WaveConfig(norm="rms", warmup=W0WarmupConfig(10.0, 0.5, 10))
        ),
        params={
            "hidden_layers": 6,
            "hidden_units": 64,
        },
    ),
    "wrn_no_phase": ModelSpec(
        name="wrn_no_phase",
        architecture=ArchitectureConfig.for_wave(
            wave=WaveConfig(norm="rms", warmup=W0WarmupConfig(10.0, 0.5, 10))
        ),
        params={
            "hidden_layers": 6,
            "hidden_units": 64,
        },
    ),
    "wrn_no_film": ModelSpec(
        name="wrn_no_film",
        architecture=ArchitectureConfig.for_wave(
            wave=WaveConfig(norm="rms", warmup=W0WarmupConfig(10.0, 0.5, 10))
        ),
        params={
            "hidden_layers": 6,
            "hidden_units": 64,
        },
    ),
    "wrn_spec_gate_rfft": ModelSpec(
        name="wrn_spec_gate_rfft",
        architecture=ArchitectureConfig.for_wave(
            wave=WaveConfig(norm="rms", warmup=W0WarmupConfig(10.0, 0.5, 10)),
            spectral=SpectralConfig(k_fft=64, gate_type="rfft"),
        ),
        params={
            "hidden_layers": 6,
            "hidden_units": 64,
        },
    ),
    "wrn_spec_gate_feats": ModelSpec(
        name="wrn_spec_gate_feats",
        architecture=ArchitectureConfig.for_wave(
            wave=WaveConfig(norm="rms", warmup=W0WarmupConfig(10.0, 0.5, 10)),
            spectral=SpectralConfig(k_fft=64, gate_type="fourier-features"),
        ),
        params={
            "hidden_layers": 6,
            "hidden_units": 64,
        },
    ),
    # SGR-PSANN ablations
    "sgr_base": ModelSpec(
        name="sgr_base",
        architecture=ArchitectureConfig.for_sequence(
            sequence=SequenceConfig(phase_trainable=True),
            spectral=SpectralConfig(k_fft=64, gate_type="rfft"),
        ),
        params={
            "hidden_layers": 3,
            "hidden_units": 64,
        },
    ),
    "sgr_no_gate": ModelSpec(
        name="sgr_no_gate",
        architecture=ArchitectureConfig.for_sequence(
            sequence=SequenceConfig(phase_trainable=True), spectral=None
        ),
        params={
            "hidden_layers": 3,
            "hidden_units": 64,
        },
    ),
    "sgr_fourier_feats": ModelSpec(
        name="sgr_fourier_feats",
        architecture=ArchitectureConfig.for_sequence(
            sequence=SequenceConfig(phase_trainable=True),
            spectral=SpectralConfig(k_fft=64, gate_type="fourier-features"),
        ),
        params={
            "hidden_layers": 3,
            "hidden_units": 64,
        },
    ),
    "sgr_no_phase": ModelSpec(
        name="sgr_no_phase",
        architecture=ArchitectureConfig.for_sequence(
            sequence=SequenceConfig(phase_trainable=False),
            spectral=SpectralConfig(k_fft=64, gate_type="rfft"),
        ),
        params={
            "hidden_layers": 3,
            "hidden_units": 64,
        },
    ),
}
