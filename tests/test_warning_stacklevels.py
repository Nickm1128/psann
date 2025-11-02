from __future__ import annotations

import os

import numpy as np
import pytest

from psann.hisso import HISSOOptions


def test_hisso_input_noise_warning_points_to_caller() -> None:
    with pytest.warns(RuntimeWarning) as record:
        HISSOOptions.from_kwargs(
            window=None,
            reward_fn=None,
            context_extractor=None,
            primary_transform="identity",
            transition_penalty=None,
            trans_cost=None,
            input_noise=np.array([0.1, 0.2], dtype=np.float32),
            supervised=None,
        )
    assert record
    warning = record[0]
    assert os.path.basename(warning.filename) == "test_warning_stacklevels.py"
