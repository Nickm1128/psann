import warnings

import pytest
import torch

from psann.state import StateController


def test_state_controller_warns_on_saturation():
    ctrl = StateController(size=1, rho=0.0, beta=10.0, max_abs=1.0, init=0.1, detach=True)
    activations = torch.ones(4, 1) * 5.0
    with pytest.warns(RuntimeWarning, match="98% of max_abs"):
        ctrl.apply(activations, feature_dim=1, update=True)
    ctrl.commit()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ctrl.apply(activations, feature_dim=1, update=True)
    assert not any("98% of max_abs" in str(item.message) for item in caught)


def test_state_controller_warns_on_collapse():
    ctrl = StateController(size=1, rho=0.0, beta=0.0, max_abs=2.0, init=1.0, detach=True)
    activations = torch.zeros(4, 1)
    with pytest.warns(RuntimeWarning, match="collapsed below 1e-3"):
        ctrl.apply(activations, feature_dim=1, update=True)
