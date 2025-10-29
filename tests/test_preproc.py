import numpy as np

from psann.lsm import LSM, LSMConv2dExpander, LSMExpander
from psann.preproc import PreprocessorSpec, build_preprocessor


def test_build_preprocessor_from_dict_instantiates_expander():
    spec = {
        "output_dim": 4,
        "hidden_units": 4,
        "hidden_width": None,
        "epochs": 0,
        "lr": 1e-3,
    }
    sample = np.zeros((4, 3), dtype=np.float32)
    module, base = build_preprocessor(spec, data=sample)
    assert isinstance(module, LSMExpander)
    assert module.model is not None
    assert base is module.model


def test_build_preprocessor_trains_when_allowed():
    data = np.linspace(-1.0, 1.0, 60, dtype=np.float32).reshape(-1, 3)
    spec = PreprocessorSpec(
        name="lsmexpander",
        params={
            "output_dim": 5,
            "hidden_units": 6,
            "hidden_width": None,
            "epochs": 1,
            "lr": 1e-3,
            "batch_size": 16,
            "random_state": 0,
        },
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


def test_build_preprocessor_from_preprocessorspec_builds_lsm_module():
    spec = PreprocessorSpec(
        name="lsm",
        params={"input_dim": 3, "output_dim": 5, "hidden_units": 7},
    )

    module, base = build_preprocessor(spec)

    assert isinstance(module, LSM)
    assert module is base
    assert module.output_dim == 5
    assert module.hidden_units == 7


def test_build_preprocessor_from_type_dict_builds_conv_expander():
    data = np.zeros((2, 1, 4, 4), dtype=np.float32)
    spec = {
        "type": "lsmconv2dexpander",
        "out_channels": 3,
        "conv_channels": 4,
        "hidden_channels": None,
        "epochs": 0,
        "lr": 1e-3,
        "random_state": 123,
    }

    module, base = build_preprocessor(
        spec,
        allow_train=True,
        pretrain_epochs=0,
        data=data,
    )

    assert isinstance(module, LSMConv2dExpander)
    assert module.model is not None
    assert base is module.model
    assert module.model.out_channels == 3


def test_build_preprocessor_from_conv_flag_dict_uses_fallback_path():
    data = np.zeros((2, 1, 3, 3), dtype=np.float32)
    spec = {
        "conv": True,
        "out_channels": 2,
        "conv_channels": 3,
        "hidden_channels": None,
        "epochs": 0,
        "lr": 1e-3,
        "random_state": 321,
    }

    module, base = build_preprocessor(
        spec,
        allow_train=True,
        pretrain_epochs=0,
        data=data,
    )

    assert isinstance(module, LSMConv2dExpander)
    assert module.model is not None
    assert base is module.model
    assert module.model.out_channels == 2


def test_build_preprocessor_keeps_prefitted_module_without_training():
    data = np.linspace(-1.0, 1.0, 24, dtype=np.float32).reshape(-1, 3)
    expander = LSMExpander(
        output_dim=6,
        hidden_layers=1,
        hidden_units=8,
        hidden_width=None,
        epochs=1,
        lr=1e-3,
        batch_size=12,
        random_state=7,
    )
    expander.fit(data, epochs=1)

    module, base = build_preprocessor(expander, allow_train=False, data=None)

    assert module is expander
    assert base is expander.model
    assert all(param.requires_grad for param in module.parameters())
