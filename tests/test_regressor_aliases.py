import pytest

pytest.importorskip("torch")

from psann import PSANNRegressor


def test_set_params_accepts_hidden_width_alias():
    model = PSANNRegressor(hidden_units=16)
    with pytest.warns(DeprecationWarning, match="hidden_width.*deprecated"):
        model.set_params(hidden_width=32)
    assert model.hidden_units == 32
    assert model.hidden_width == 32
    params = model.get_params()
    assert params["hidden_units"] == 32
    assert params["hidden_width"] == 32


def test_set_params_warns_on_hidden_width_mismatch_prefers_primary():
    model = PSANNRegressor(hidden_units=16)
    with pytest.warns(UserWarning, match="hidden_units` overrides `hidden_width`"):
        model.set_params(hidden_units=24, hidden_width=48)
    assert model.hidden_units == 24
    assert model.hidden_width == 24


def test_set_params_accepts_hidden_channels_alias():
    model = PSANNRegressor(hidden_units=10)
    with pytest.warns(DeprecationWarning, match="hidden_channels.*deprecated"):
        model.set_params(hidden_channels=12)
    assert model.conv_channels == 12
    params = model.get_params()
    assert params["conv_channels"] == 12


def test_set_params_hidden_channels_mismatch_warns():
    model = PSANNRegressor(conv_channels=18, preserve_shape=True)
    with pytest.warns(UserWarning, match="conv_channels` overrides `hidden_channels`"):
        model.set_params(conv_channels=18, hidden_channels=20)
    assert model.conv_channels == 18
