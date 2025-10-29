from __future__ import annotations

import numpy as np
import pytest

from psann.sklearn import WaveResNetRegressor


def _make_dataset(n_samples: int = 48) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n_samples, 4)).astype(np.float32)
    context = rng.standard_normal((n_samples, 2)).astype(np.float32)
    signal = X.sum(axis=1, keepdims=True) + 0.5 * context[:, :1] - 0.25 * context[:, 1:]
    y = signal.astype(np.float32)
    return X, y, context


def test_wave_resnet_regressor_requires_context_when_configured() -> None:
    X, y, _ = _make_dataset()
    estimator = WaveResNetRegressor(hidden_layers=2, hidden_width=16, epochs=2, context_dim=2)
    with pytest.raises(ValueError, match="expects a context array"):
        estimator.fit(X, y)


def test_wave_resnet_regressor_infers_context_dimension() -> None:
    X, y, context = _make_dataset()
    estimator = WaveResNetRegressor(hidden_layers=2, hidden_width=16, epochs=2, context_dim=None)
    estimator.fit(X, y, context=context)
    assert estimator.context_dim == 2
    preds = estimator.predict(X[:4], context=context[:4])
    assert preds.shape[0] == 4
    with pytest.raises(ValueError, match="provide a matching context array"):
        estimator.predict(X[:2])


def test_wave_resnet_regressor_responds_to_context() -> None:
    X, y, context = _make_dataset()
    estimator = WaveResNetRegressor(
        hidden_layers=2,
        hidden_width=16,
        epochs=3,
        batch_size=16,
        context_dim=2,
        random_state=7,
    )
    estimator.fit(X, y, context=context)
    base = estimator.predict(X[:6], context=context[:6])
    shifted_context = context[:6] + 0.75
    shifted = estimator.predict(X[:6], context=shifted_context)
    assert not np.allclose(base, shifted)


def test_wave_resnet_regressor_cosine_builder_auto_context() -> None:
    X, y, _ = _make_dataset()
    estimator = WaveResNetRegressor(
        hidden_layers=2,
        hidden_width=16,
        epochs=2,
        batch_size=16,
        context_builder="cosine",
        context_builder_params={"frequencies": 1, "include_sin": False},
    )
    estimator.fit(X, y)
    assert estimator.context_dim == X.shape[1]
    preds = estimator.predict(X[:5])
    assert preds.shape == (5, 1)


def test_wave_resnet_w0_warmup_schedule_progresses_to_target() -> None:
    estimator = WaveResNetRegressor(
        hidden_layers=3,
        hidden_width=12,
        epochs=1,
        first_layer_w0=25.0,
        hidden_w0=1.5,
        first_layer_w0_initial=5.0,
        hidden_w0_initial=0.3,
        w0_warmup_epochs=4,
    )
    core = estimator._build_dense_core(input_dim=3, output_dim=1)
    estimator.model_ = core
    estimator._reset_w0_schedule()

    init_first, init_hidden = estimator._initial_w0_values()
    assert core.stem_w0 == pytest.approx(init_first)
    previous_blocks = [block.w0 for block in core.blocks]
    assert all(val == pytest.approx(init_hidden) for val in previous_blocks)
    previous_stem = core.stem_w0

    for step in range(1, estimator.w0_warmup_epochs + 1):
        estimator._update_w0_schedule(step)
        assert core.stem_w0 >= previous_stem - 1e-6
        current_blocks = [block.w0 for block in core.blocks]
        for prev, current in zip(previous_blocks, current_blocks):
            assert current >= prev - 1e-6
        previous_blocks = current_blocks
        previous_stem = core.stem_w0

    target_first, target_hidden = estimator._target_w0_values()
    assert core.stem_w0 == pytest.approx(target_first)
    assert all(val == pytest.approx(target_hidden) for val in previous_blocks)
    assert estimator._w0_schedule_active is False


def test_wave_resnet_progressive_depth_adds_blocks_and_optimizer_group() -> None:
    estimator = WaveResNetRegressor(
        hidden_layers=4,
        hidden_width=12,
        epochs=1,
        first_layer_w0=20.0,
        hidden_w0=1.2,
        first_layer_w0_initial=5.0,
        hidden_w0_initial=0.4,
        w0_warmup_epochs=3,
        progressive_depth_initial=2,
        progressive_depth_interval=1,
        progressive_depth_growth=1,
    )
    core = estimator._build_dense_core(input_dim=3, output_dim=1)
    estimator.model_ = core
    optimizer = estimator._build_optimizer(core)
    estimator._optimizer_ = optimizer
    estimator._reset_w0_schedule()
    estimator._reset_progressive_depth()

    assert len(core.blocks) == estimator.progressive_depth_initial
    original_group_count = len(optimizer.param_groups)
    tracked_params = {id(param) for group in optimizer.param_groups for param in group["params"]}

    estimator.epoch_callback(
        epoch=0,
        train_loss=0.0,
        val_loss=None,
        improved=False,
        patience_left=None,
    )

    assert len(core.blocks) == estimator.progressive_depth_initial + 1
    assert len(optimizer.param_groups) == original_group_count + 1

    new_group_params = optimizer.param_groups[-1]["params"]
    new_param_ids = {id(param) for param in new_group_params}
    new_block_ids = {id(param) for param in core.blocks[-1].parameters()}
    assert new_param_ids == new_block_ids
    assert new_param_ids.isdisjoint(tracked_params)

    expected_hidden = estimator._current_w0_values()[1]
    assert core.blocks[-1].w0 == pytest.approx(expected_hidden)
