import importlib

import pytest


def test_psann_lm_stub_guidance():
    with pytest.raises(ImportError) as exc:
        importlib.import_module("psann.lm")
    assert "psannlm" in str(exc.value)
