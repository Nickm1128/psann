"""Runtime invariants exposed while checking the public numerical utilities."""

import pytest
import torch
from torch import nn

from psann.state import StateController
from psann.utils.linear_probe import _unpack_batch


def test_state_controller_preserves_recursive_module_apply_and_tensor_updates():
    controller = StateController(3, init=2.0)
    model = nn.Sequential(nn.Linear(3, 3), controller)
    visited = []
    assert model.apply(visited.append) is model
    assert visited == [model[0], controller, model]
    values = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    torch.testing.assert_close(controller.apply(values, feature_dim=1), 2 * values)
    controller.commit()
    assert not torch.equal(controller.state, torch.full((3,), 2.0))


@pytest.mark.parametrize("keys", [("x", "y", "c"), ("inputs", "targets", "context")])
def test_probe_dictionary_batches_preserve_multielement_tensors(keys):
    tensors = (torch.ones(3, 2), torch.arange(3), torch.zeros(3, 1))
    result = _unpack_batch(dict(zip(keys, tensors)))
    assert all(actual is expected for actual, expected in zip(result, tensors))


def test_probe_dictionary_short_keys_take_precedence():
    x, y = torch.zeros(3, 2), torch.zeros(3, dtype=torch.long)
    result = _unpack_batch({"x": x, "y": y, "inputs": torch.ones_like(x)})
    assert result[0] is x and result[1] is y and result[2] is None
