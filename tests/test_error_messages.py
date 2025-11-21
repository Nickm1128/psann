import numpy as np
import pytest
import torch

from psann.episodes import multiplicative_return_reward
from psann.hisso import ensure_hisso_trainer_config
from psann.sklearn import PSANNRegressor


def _make_regressor(**kwargs):
    return PSANNRegressor(
        hidden_layers=1,
        hidden_units=8,
        epochs=1,
        batch_size=4,
        lr=1e-3,
        **kwargs,
    )


def test_fit_context_length_mismatch():
    reg = _make_regressor()
    X = np.zeros((8, 4), dtype=np.float32)
    y = np.zeros((8, 1), dtype=np.float32)
    bad_context = np.zeros((7, 2), dtype=np.float32)

    with pytest.raises(
        ValueError, match="context has 7 samples but X has 8; dimensions must match"
    ):
        reg.fit(X, y, context=bad_context)


def test_validation_channel_mismatch_message():
    reg = _make_regressor(preserve_shape=True, data_format="channels_first")
    X = np.zeros((4, 1, 5), dtype=np.float32)
    y = np.zeros((4, 5), dtype=np.float32)
    X_val = np.zeros((2, 2, 5), dtype=np.float32)  # wrong channel dimension
    y_val = np.zeros((2, 5), dtype=np.float32)

    with pytest.raises(
        ValueError, match="validation_data channels mismatch: expected 1, received 2"
    ):
        reg.fit(X, y, validation_data=(X_val, y_val))


def test_validation_per_element_ndim_message():
    reg = _make_regressor(
        preserve_shape=True,
        data_format="channels_last",
        per_element=True,
    )
    X = np.zeros((4, 5, 3), dtype=np.float32)
    y = np.zeros((4, 5, 3), dtype=np.float32)
    X_val = np.zeros((2, 5, 3), dtype=np.float32)
    y_val = np.zeros(2, dtype=np.float32)  # ndim == 1, invalid for per-element

    with pytest.raises(
        ValueError, match="validation y ndim must be 3 or 2 for data_format='channels_last'"
    ):
        reg.fit(X, y, validation_data=(X_val, y_val))


def test_multiplicative_return_reward_requires_rank3():
    actions = torch.zeros(4, 5)  # missing final dim
    context = torch.zeros(4, 5, 3)

    with pytest.raises(
        ValueError, match="actions/context must be rank-3 \\(B, T, M\\); received actions.ndim=2"
    ):
        multiplicative_return_reward(actions, context)


def test_multiplicative_return_reward_requires_matching_shapes():
    actions = torch.zeros(2, 3, 4)
    context = torch.zeros(2, 3, 5)

    with pytest.raises(ValueError, match="actions and context must align element-wise"):
        multiplicative_return_reward(actions, context)


def test_ensure_hisso_trainer_config_type_error():
    with pytest.raises(TypeError, match="received int"):
        ensure_hisso_trainer_config(123)
