"""CUDA smoke coverage for the canonical architecture registry."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from psann.architectures import ArchitectureConfig, ContextConfig, ConvolutionConfig, GeometryConfig
from psann.estimators import PSANNRegressor
from psann.episodic import EpisodeScheduleConfig, EpisodicTrainer, HISSOConfig
from psann.lsm import LSMConv2dExpander
from psann.preprocessing import (
    LSMConfig,
    LSMPretrainingConfig,
    PreprocessorConfig,
    PreprocessorTrainingConfig,
)

pytestmark = pytest.mark.gpu


def _flat() -> tuple[np.ndarray, np.ndarray]:
    values = np.linspace(-1.0, 1.0, 24, dtype=np.float32).reshape(6, 4)
    return values, values.sum(axis=1, keepdims=True)


@pytest.mark.parametrize(
    ("architecture", "reshape"),
    [
        (ArchitectureConfig.dense(), None),
        (ArchitectureConfig.convolutional(convolution=ConvolutionConfig(channels=4)), (6, 1, 2, 2)),
        (ArchitectureConfig.for_wave(), None),
        (ArchitectureConfig.for_sequence(), (6, 2, 2)),
        (ArchitectureConfig.geometric_sparse(geometry=GeometryConfig(shape=(2, 2))), None),
    ],
)
def test_canonical_registry_architecture_fits_on_cuda(architecture, reshape) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    X, y = _flat()
    if reshape is not None:
        X = X.reshape(reshape)
    estimator = PSANNRegressor(
        architecture=architecture,
        hidden_layers=2,
        hidden_units=8,
        epochs=1,
        batch_size=3,
        device="cuda",
        random_state=17,
    ).fit(X[:4], y[:4], verbose=0)
    prediction = estimator.predict(X[4:])
    assert prediction.shape[0] == 2
    assert next(estimator.model_.parameters()).device.type == "cuda"


def test_convolutional_lsm_checkpoint_persists_on_cuda(tmp_path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    X = np.ones((8, 1, 2, 2), dtype=np.float32)
    y = X.mean(axis=(1, 2, 3))
    estimator = PSANNRegressor(
        architecture=ArchitectureConfig.convolutional(convolution=ConvolutionConfig(channels=4)),
        hidden_layers=1,
        hidden_units=4,
        epochs=1,
        batch_size=4,
        device="cuda",
        random_state=17,
        lsm=LSMConv2dExpander(out_channels=2, hidden_layers=1, conv_channels=4, epochs=1),
        lsm_train=True,
    ).fit(X, y)
    path = tmp_path / "cuda-conv-lsm.pt"
    estimator.save(str(path))
    loaded = PSANNRegressor.load(str(path), map_location="cuda")
    np.testing.assert_allclose(loaded.predict(X[:2]), estimator.predict(X[:2]), rtol=1e-6)
    assert next(loaded.model_.parameters()).device.type == "cuda"


@pytest.mark.parametrize("convolutional", [False, True])
def test_canonical_lsm_cuda_trainable_two_generation_checkpoint(
    tmp_path, convolutional: bool
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if convolutional:
        X = np.arange(72, dtype=np.float32).reshape(8, 1, 3, 3) / 10
        y = X.mean(axis=(1, 2, 3))
        architecture = ArchitectureConfig.convolutional(convolution=ConvolutionConfig(channels=3))
        component = LSMConfig.convolutional(
            output_dim=2,
            hidden_units=3,
            random_state=0,
            pretraining=LSMPretrainingConfig(epochs=0),
        )
    else:
        X, y = _flat()
        X = np.concatenate((X, X[:2]), axis=0)
        y = np.concatenate((y, y[:2]), axis=0)
        architecture = ArchitectureConfig.dense()
        component = LSMConfig.dense(
            output_dim=4,
            hidden_layers=1,
            hidden_units=5,
            random_state=0,
            pretraining=LSMPretrainingConfig(epochs=0, batch_size=4),
        )
    estimator = PSANNRegressor(
        architecture=architecture,
        preprocessor=PreprocessorConfig(
            component, PreprocessorTrainingConfig(trainable=True, lr=5e-3)
        ),
        hidden_layers=1,
        hidden_units=4,
        epochs=1,
        batch_size=4,
        device="cuda",
        random_state=0,
    ).fit(X, y)
    assert all(parameter.requires_grad for parameter in estimator.preprocessor_.parameters())
    first = tmp_path / f"canonical-{convolutional}-first.pt"
    second = tmp_path / f"canonical-{convolutional}-second.pt"
    estimator.save(str(first))
    restored = PSANNRegressor.load(str(first), map_location="cuda")
    restored.save(str(second))
    reloaded = PSANNRegressor.load(str(second), map_location="cuda")
    np.testing.assert_allclose(reloaded.predict(X[:2]), estimator.predict(X[:2]), rtol=1e-5)
    assert next(reloaded.model_.parameters()).device.type == "cuda"


@pytest.mark.parametrize("convolutional", [False, True])
def test_canonical_lsm_schema_v2_two_generations_survive_bidirectional_map_location(
    tmp_path, convolutional: bool
) -> None:
    """A CPU v2 save can rebuild on CUDA and close again on CPU."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if convolutional:
        X = np.arange(72, dtype=np.float32).reshape(8, 1, 3, 3) / 10
        y = X.mean(axis=(1, 2, 3))
        architecture = ArchitectureConfig.convolutional(convolution=ConvolutionConfig(channels=3))
        component = LSMConfig.convolutional(output_dim=2, hidden_units=3, random_state=0)
    else:
        X, y = _flat()
        X = np.concatenate((X, X[:2]), axis=0)
        y = np.concatenate((y, y[:2]), axis=0)
        architecture = ArchitectureConfig.dense()
        component = LSMConfig.dense(output_dim=4, hidden_layers=1, hidden_units=5, random_state=0)
    estimator = PSANNRegressor(
        architecture=architecture,
        preprocessor=PreprocessorConfig(component),
        hidden_layers=1,
        hidden_units=4,
        epochs=1,
        batch_size=4,
        device="cpu",
        random_state=0,
    ).fit(X, y)
    first = tmp_path / f"cpu-{convolutional}.pt"
    second = tmp_path / f"cuda-{convolutional}.pt"
    estimator.save(str(first))
    cuda_loaded = PSANNRegressor.load(str(first), map_location="cuda")
    assert next(cuda_loaded.model_.parameters()).device.type == "cuda"
    cuda_loaded.save(str(second))
    cpu_reloaded = PSANNRegressor.load(str(second), map_location="cpu")
    assert next(cpu_reloaded.model_.parameters()).device.type == "cpu"
    np.testing.assert_allclose(cpu_reloaded.predict(X[:2]), estimator.predict(X[:2]), rtol=1e-5)


def test_cuda_multichannel_channels_last_wave_context_keeps_sample_axis() -> None:
    """CUDA covers the corrected context route independently of NHWC model layout."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    X_cf = np.arange(144, dtype=np.float32).reshape(8, 2, 3, 3) / 10
    X = np.moveaxis(X_cf, 1, -1)
    y = X_cf.mean(axis=(1, 2, 3))
    estimator = PSANNRegressor(
        architecture=ArchitectureConfig.for_wave(
            convolution=ConvolutionConfig(channels=3, data_format="channels_last"),
            context=ContextConfig(builder="cosine", builder_params={"include_sin": False}),
        ),
        hidden_layers=1,
        hidden_units=4,
        epochs=1,
        batch_size=4,
        device="cuda",
        scaler="standard",
        random_state=0,
    ).fit(X, y, validation_data=(X[:2], y[:2]))
    _, meta, context = estimator._prepare_inference_inputs(X[:2])
    assert meta["n_samples"] == 2
    assert context is not None and context.shape[0] == 2
    assert estimator.predict(X[:2]).shape == (2,)


@pytest.mark.parametrize("convolutional", [False, True])
def test_canonical_episodic_cuda_two_generation_map_location(tmp_path, convolutional: bool) -> None:
    """Canonical episodic fit/evaluate closes CUDA→CPU schema-v3 generations."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if convolutional:
        X = np.arange(72, dtype=np.float32).reshape(8, 1, 3, 3) + 1
        architecture = ArchitectureConfig.convolutional(convolution=ConvolutionConfig(channels=3))
    else:
        X = np.arange(16, dtype=np.float32).reshape(8, 2) + 1
        architecture = ArchitectureConfig.dense()
    trainer = EpisodicTrainer(
        estimator=PSANNRegressor(
            architecture=architecture, epochs=1, batch_size=2, device="cuda", random_state=0
        ),
        strategy=HISSOConfig(
            schedule=EpisodeScheduleConfig(episode_length=2, batch_episodes=2),
            mixed_precision=True,
            amp_dtype="bfloat16",
        ),
    ).fit(X)
    assert trainer.profile_["amp_enabled"] is True
    assert trainer.evaluate(X[:3]) == pytest.approx(trainer.evaluate(X[:3]))
    first, second = tmp_path / "episodic-cuda.pt", tmp_path / "episodic-cpu.pt"
    trainer.save(first)
    cpu = EpisodicTrainer.load(first, map_location="cpu")
    cpu.save(second)
    loaded = EpisodicTrainer.load(second, map_location="cpu")
    assert next(loaded.estimator.model_.parameters()).device.type == "cpu"
    np.testing.assert_allclose(loaded.predict(X[:2]), cpu.predict(X[:2]), rtol=1e-5)


@pytest.mark.parametrize("convolutional", [False, True], ids=["dense-lsm", "conv2d-lsm"])
def test_canonical_episodic_bidirectional_cpu_cuda_schema_v3_closure(
    tmp_path, convolutional: bool
) -> None:
    """Canonical episodic checkpoints close CPU → CUDA → CPU with LSM metadata."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if convolutional:
        X = np.arange(72, dtype=np.float32).reshape(8, 1, 3, 3) / 10
        architecture = ArchitectureConfig.convolutional(convolution=ConvolutionConfig(channels=3))
        preprocessor = PreprocessorConfig(
            LSMConfig.convolutional(output_dim=2, hidden_units=3, random_state=0),
            PreprocessorTrainingConfig(trainable=True, lr=5e-3),
        )
    else:
        X, _ = _flat()
        X = np.concatenate((X, X[:2]), axis=0)
        architecture = ArchitectureConfig.dense()
        preprocessor = PreprocessorConfig(
            LSMConfig.dense(output_dim=4, hidden_units=5, random_state=0),
            PreprocessorTrainingConfig(trainable=True, lr=5e-3),
        )
    trainer = EpisodicTrainer(
        estimator=PSANNRegressor(
            architecture=architecture,
            preprocessor=preprocessor,
            hidden_layers=1,
            hidden_units=4,
            epochs=1,
            batch_size=2,
            device="cpu",
            random_state=0,
        ),
        strategy=HISSOConfig(schedule=EpisodeScheduleConfig(episode_length=2, batch_episodes=1)),
    ).fit(X)
    cpu_path, cuda_path = tmp_path / "cpu.pt", tmp_path / "cuda.pt"
    trainer.save(cpu_path)
    cuda = EpisodicTrainer.load(cpu_path, map_location="cuda")
    assert next(cuda.estimator.model_.parameters()).device.type == "cuda"
    assert np.isfinite(cuda.evaluate(X[:4]))
    cuda.save(cuda_path)
    restored = EpisodicTrainer.load(cuda_path, map_location="cpu")
    assert next(restored.estimator.model_.parameters()).device.type == "cpu"
    np.testing.assert_allclose(restored.predict(X[:2]), trainer.predict(X[:2]), rtol=1e-5)


@pytest.mark.parametrize(
    ("primary_width", "context_width"),
    [(2, 2), (2, 1), (1, 3)],
    ids=["exact", "singleton-broadcast", "reduced-for-scalar-action"],
)
def test_canonical_episodic_cuda_context_alignment_uses_one_strict_runtime(
    primary_width: int, context_width: int
) -> None:
    """CUDA exercises strict reward alignment in wrapper training and evaluation."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    observed: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []

    def context(inputs: torch.Tensor) -> torch.Tensor:
        base = inputs[..., :1]
        return base.expand(*base.shape[:-1], context_width)

    def reward(actions: torch.Tensor, aligned: torch.Tensor, **_kwargs: object) -> torch.Tensor:
        observed.append((actions.device.type, tuple(actions.shape), tuple(aligned.shape)))
        return -(actions - aligned).square().mean(dim=(-1, -2))

    X = np.arange(24, dtype=np.float32).reshape(12, 2) + 1
    y = np.ones((12, primary_width), dtype=np.float32)
    trainer = EpisodicTrainer(
        estimator=PSANNRegressor(
            epochs=1, batch_size=2, device="cuda", random_state=0, scaler="standard"
        ),
        strategy=HISSOConfig(
            schedule=EpisodeScheduleConfig(episode_length=3, batch_episodes=1),
            context_extractor=context,
            reward=reward,
        ),
    ).fit(X, y)
    assert trainer.predict(X[:2]).shape == (2, primary_width)
    assert np.isfinite(trainer.evaluate(X[:4]))
    assert len(observed) >= 2
    assert all(
        device == "cuda" and action_shape == context_shape
        for device, action_shape, context_shape in observed
    )
