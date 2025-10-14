import numpy as np

from psann.lsm import LSMExpander
from psann.preproc import PreprocessorSpec, build_preprocessor


def test_build_preprocessor_from_dict_instantiates_expander():
    spec = {"output_dim": 4, "hidden_width": 4, "epochs": 0, "lr": 1e-3}
    sample = np.zeros((4, 3), dtype=np.float32)
    module, base = build_preprocessor(spec, data=sample)
    assert isinstance(module, LSMExpander)
    assert module.model is not None
    assert base is module.model


def test_build_preprocessor_trains_when_allowed():
    data = np.linspace(-1.0, 1.0, 60, dtype=np.float32).reshape(-1, 3)
    spec = PreprocessorSpec(
        name="lsmexpander",
        params={"output_dim": 5, "hidden_width": 6, "epochs": 1, "lr": 1e-3, "batch_size": 16, "random_state": 0},
    )

    module, base = build_preprocessor(
        spec,
        allow_train=True,
        pretrain_epochs=1,
        data=data,
    )

    assert isinstance(module, LSMExpander)
    assert module.model is not None
    assert base is module.model
