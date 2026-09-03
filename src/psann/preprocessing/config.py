"""Immutable public preprocessing configuration and normalization."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from numbers import Real
from typing import Any, TypeAlias, cast

import torch.nn as nn


def _name(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{path} must be a string.")
    return value.strip().lower().replace("_", "-")


def _integer(value: object, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{path} must be an integer.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{path} must be at least {minimum}.")
    return value


def _finite(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{path} must be a finite real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite.")
    return result


def _positive(value: object, path: str) -> float:
    result = _finite(value, path)
    if result <= 0:
        raise ValueError(f"{path} must be positive.")
    return result


def _nonnegative(value: object, path: str) -> float:
    result = _finite(value, path)
    if result < 0:
        raise ValueError(f"{path} must be non-negative.")
    return result


def _boolean(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{path} must be a boolean.")
    return value


def _optional_integer(value: object, path: str, *, minimum: int = 0) -> int | None:
    return None if value is None else _integer(value, path, minimum=minimum)


def _optional_finite(value: object, path: str) -> float | None:
    return None if value is None else _finite(value, path)


@dataclass(frozen=True)
class LSMPretrainingConfig:
    epochs: int = 0
    lr: float = 1e-3
    ridge: float = 1e-4
    batch_size: int | None = None
    early_stopping: bool | None = None
    patience: int | None = None
    tol: float | None = None
    val_split: float | None = None
    verbose: int | None = None
    objective: str | None = None
    noisy: float | tuple[float, ...] | None = None
    noise_decay: float = 1.0
    alpha_ortho: float = 0.0
    alpha_sparse: float = 0.0
    alpha_var: float = 0.0
    target_var: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "epochs",
            _integer(self.epochs, "preprocessor.component.pretraining.epochs", minimum=0),
        )
        object.__setattr__(self, "lr", _positive(self.lr, "preprocessor.component.pretraining.lr"))
        object.__setattr__(
            self, "ridge", _nonnegative(self.ridge, "preprocessor.component.pretraining.ridge")
        )
        object.__setattr__(
            self,
            "batch_size",
            _optional_integer(
                self.batch_size, "preprocessor.component.pretraining.batch_size", minimum=1
            ),
        )
        if self.early_stopping is not None:
            _boolean(self.early_stopping, "preprocessor.component.pretraining.early_stopping")
        object.__setattr__(
            self,
            "patience",
            _optional_integer(
                self.patience, "preprocessor.component.pretraining.patience", minimum=1
            ),
        )
        object.__setattr__(
            self, "tol", _optional_finite(self.tol, "preprocessor.component.pretraining.tol")
        )
        val_split = _optional_finite(self.val_split, "preprocessor.component.pretraining.val_split")
        if val_split is not None and not 0 <= val_split < 1:
            raise ValueError(
                "preprocessor.component.pretraining.val_split must satisfy 0 <= value < 1."
            )
        object.__setattr__(self, "val_split", val_split)
        object.__setattr__(
            self,
            "verbose",
            _optional_integer(
                self.verbose, "preprocessor.component.pretraining.verbose", minimum=0
            ),
        )
        if self.objective is not None:
            objective = _name(self.objective, "preprocessor.component.pretraining.objective")
            if objective not in {"r2", "mse"}:
                raise ValueError("preprocessor.component.pretraining.objective must be r2 or mse.")
            object.__setattr__(self, "objective", objective)
        if self.noisy is not None:
            values: tuple[object, ...]
            if isinstance(self.noisy, (str, bytes)) or not isinstance(self.noisy, (Real, Sequence)):
                raise TypeError(
                    "preprocessor.component.pretraining.noisy must be a real or sequence of reals."
                )
            values = (self.noisy,) if isinstance(self.noisy, Real) else tuple(self.noisy)
            object.__setattr__(
                self,
                "noisy",
                (
                    tuple(
                        _nonnegative(v, f"preprocessor.component.pretraining.noisy[{i}]")
                        for i, v in enumerate(values)
                    )
                    if len(values) != 1
                    else _nonnegative(values[0], "preprocessor.component.pretraining.noisy")
                ),
            )
        object.__setattr__(
            self,
            "noise_decay",
            _nonnegative(self.noise_decay, "preprocessor.component.pretraining.noise_decay"),
        )
        for name in ("alpha_ortho", "alpha_sparse", "alpha_var"):
            object.__setattr__(
                self,
                name,
                _nonnegative(getattr(self, name), f"preprocessor.component.pretraining.{name}"),
            )
        object.__setattr__(
            self,
            "target_var",
            _positive(self.target_var, "preprocessor.component.pretraining.target_var"),
        )


@dataclass(frozen=True)
class LSMConfig:
    topology: str = "dense"
    output_dim: int = 128
    hidden_layers: int = 2
    hidden_units: int = 128
    kernel_size: int | None = None
    sparsity: float = 0.8
    nonlinearity: str = "sine"
    bias: bool = True
    random_state: int | None = None
    pretraining: LSMPretrainingConfig = LSMPretrainingConfig()

    def __post_init__(self) -> None:
        topology = _name(self.topology, "preprocessor.component.topology")
        if topology not in {"dense", "conv2d"}:
            raise ValueError("preprocessor.component.topology must be dense or conv2d.")
        object.__setattr__(self, "topology", topology)
        object.__setattr__(
            self,
            "output_dim",
            _integer(self.output_dim, "preprocessor.component.output_dim", minimum=1),
        )
        object.__setattr__(
            self,
            "hidden_layers",
            _integer(self.hidden_layers, "preprocessor.component.hidden_layers", minimum=1),
        )
        object.__setattr__(
            self,
            "hidden_units",
            _integer(self.hidden_units, "preprocessor.component.hidden_units", minimum=1),
        )
        if topology == "dense":
            if self.kernel_size is not None:
                raise ValueError("preprocessor.component.kernel_size requires conv2d topology.")
        else:
            object.__setattr__(
                self,
                "kernel_size",
                _integer(
                    self.kernel_size if self.kernel_size is not None else 1,
                    "preprocessor.component.kernel_size",
                    minimum=1,
                ),
            )
        sparsity = _finite(self.sparsity, "preprocessor.component.sparsity")
        if not 0 <= sparsity <= 1:
            raise ValueError("preprocessor.component.sparsity must satisfy 0 <= value <= 1.")
        object.__setattr__(self, "sparsity", sparsity)
        nonlinearity = _name(self.nonlinearity, "preprocessor.component.nonlinearity")
        if nonlinearity not in {"sine", "tanh", "relu"}:
            raise ValueError("preprocessor.component.nonlinearity must be sine, tanh, or relu.")
        object.__setattr__(self, "nonlinearity", nonlinearity)
        _boolean(self.bias, "preprocessor.component.bias")
        if self.random_state is not None:
            _integer(self.random_state, "preprocessor.component.random_state")
        if not isinstance(self.pretraining, LSMPretrainingConfig):
            raise TypeError("preprocessor.component.pretraining must be an LSMPretrainingConfig.")
        if topology == "conv2d":
            for name in (
                "batch_size",
                "early_stopping",
                "patience",
                "tol",
                "val_split",
                "verbose",
                "objective",
            ):
                if getattr(self.pretraining, name) is not None:
                    raise ValueError(
                        f"preprocessor.component.pretraining.{name} is not supported for conv2d LSM."
                    )

    @classmethod
    def dense(cls, **kwargs: object) -> "LSMConfig":
        return cls(topology="dense", **cast(Any, kwargs))

    @classmethod
    def convolutional(cls, **kwargs: object) -> "LSMConfig":
        kwargs.setdefault("hidden_layers", 1)
        kwargs.setdefault("kernel_size", 1)
        return cls(topology="conv2d", **cast(Any, kwargs))


@dataclass(frozen=True)
class ModulePreprocessorConfig:
    module: nn.Module
    input_topology: str
    output_topology: str
    output_dim: int

    def __post_init__(self) -> None:
        if not isinstance(self.module, nn.Module):
            raise TypeError("preprocessor.component.module must be a torch.nn.Module.")
        for name in ("input_topology", "output_topology"):
            value = _name(getattr(self, name), f"preprocessor.component.{name}")
            if value not in {"flat", "tokens", "spatial-1d", "spatial-2d", "spatial-3d"}:
                raise ValueError(f"preprocessor.component.{name} is not a supported topology.")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "output_dim",
            _integer(self.output_dim, "preprocessor.component.output_dim", minimum=1),
        )


@dataclass(frozen=True)
class PreprocessorTrainingConfig:
    trainable: bool = False
    lr: float | None = None

    def __post_init__(self) -> None:
        _boolean(self.trainable, "preprocessor.training.trainable")
        object.__setattr__(
            self, "lr", None if self.lr is None else _positive(self.lr, "preprocessor.training.lr")
        )


@dataclass(frozen=True)
class PreprocessorConfig:
    component: LSMConfig | ModulePreprocessorConfig
    training: PreprocessorTrainingConfig = PreprocessorTrainingConfig()

    def __post_init__(self) -> None:
        if not isinstance(self.component, (LSMConfig, ModulePreprocessorConfig)):
            raise TypeError(
                "preprocessor.component must be an LSMConfig or ModulePreprocessorConfig."
            )
        if not isinstance(self.training, PreprocessorTrainingConfig):
            raise TypeError("preprocessor.training must be a PreprocessorTrainingConfig.")


PreprocessorLike: TypeAlias = PreprocessorConfig | Mapping[str, object] | None


def _normalize_pretraining(value: object) -> LSMPretrainingConfig:
    if value is None:
        return LSMPretrainingConfig()
    if isinstance(value, LSMPretrainingConfig):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("preprocessor.component.pretraining must be a mapping.")
    raw = dict(value)
    unknown = set(raw) - {field.name for field in fields(LSMPretrainingConfig)}
    if unknown:
        raise ValueError(f"preprocessor.component.pretraining.{min(unknown)} is unknown.")
    return LSMPretrainingConfig(**cast(Any, raw))


def normalize_preprocessor(value: PreprocessorLike) -> PreprocessorConfig | None:
    """Normalize the one serializable canonical preprocessing mapping."""

    if value is None or isinstance(value, PreprocessorConfig):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("preprocessor must be a PreprocessorConfig, mapping, or None.")
    raw = dict(value)
    unknown = set(raw) - {"kind", "lsm", "training"}
    if unknown:
        raise ValueError(f"preprocessor.{min(unknown)} is unknown.")
    kind = _name(raw.get("kind", ""), "preprocessor.kind")
    if kind != "lsm":
        raise ValueError("preprocessor.kind must be lsm.")
    if "lsm" not in raw:
        raise ValueError("preprocessor.lsm is required when preprocessor.kind is lsm.")
    lsm = raw["lsm"]
    if not isinstance(lsm, Mapping):
        raise TypeError("preprocessor.lsm must be a mapping.")
    lsm_raw = dict(lsm)
    aliases = {
        "out_channels": "output_dim",
        "hidden_width": "hidden_units",
        "hidden_channels": "hidden_units",
    }
    for old, new in aliases.items():
        if old in lsm_raw:
            if new in lsm_raw and lsm_raw[old] != lsm_raw[new]:
                raise ValueError(f"preprocessor.lsm has conflicting keys {old!r} and {new!r}.")
            lsm_raw[new] = lsm_raw.pop(old)
    if "pretraining" in lsm_raw:
        lsm_raw["pretraining"] = _normalize_pretraining(lsm_raw["pretraining"])
    else:
        lsm_raw["pretraining"] = LSMPretrainingConfig()
    unknown_lsm = set(lsm_raw) - {field.name for field in fields(LSMConfig)}
    if unknown_lsm:
        raise ValueError(f"preprocessor.lsm.{min(unknown_lsm)} is unknown.")
    training_raw = raw.get("training", {})
    if isinstance(training_raw, PreprocessorTrainingConfig):
        training = training_raw
    elif isinstance(training_raw, Mapping):
        unknown_training = set(training_raw) - {
            field.name for field in fields(PreprocessorTrainingConfig)
        }
        if unknown_training:
            raise ValueError(f"preprocessor.training.{min(unknown_training)} is unknown.")
        training = PreprocessorTrainingConfig(**cast(Any, dict(training_raw)))
    else:
        raise TypeError("preprocessor.training must be a mapping.")
    return PreprocessorConfig(component=LSMConfig(**cast(Any, lsm_raw)), training=training)


def normalize_legacy_lsm(
    value: object,
    *,
    trainable: bool = False,
    pretrain_epochs: int = 0,
    training_lr: float | None = None,
) -> PreprocessorConfig | None:
    """Adapt lossless legacy mapping/spec inputs at the compatibility boundary.

    Existing module and expander objects intentionally return ``None`` here: their
    fitted graph is preserved by the retained module adapter rather than guessed
    from mutable implementation attributes.
    """

    if value is None or isinstance(value, nn.Module):
        return None
    if isinstance(value, PreprocessorConfig):
        return value
    if hasattr(value, "name") and hasattr(value, "params"):
        value = {
            "name": getattr(value, "name"),
            **dict(cast(Mapping[str, object], getattr(value, "params"))),
        }
    if not isinstance(value, Mapping):
        return None
    raw = dict(value)
    tag = _name(raw.pop("type", raw.pop("name", raw.pop("kind", "lsm"))), "lsm.kind")
    conv = bool(raw.pop("conv", False))
    if tag in {"lsmconv2d", "lsmconv2dexpander"}:
        conv = True
    elif tag not in {"lsm", "lsmexpander"}:
        raise ValueError(f"lsm.kind {tag!r} is unknown.")
    aliases = {
        "out_channels": "output_dim",
        "hidden_width": "hidden_units",
        "hidden_channels": "hidden_units",
    }
    if conv:
        aliases["conv_channels"] = "hidden_units"
    for old, new in aliases.items():
        if old in raw:
            if raw[old] is None:
                raw.pop(old)
                continue
            if new in raw and raw[old] != raw[new]:
                raise ValueError(f"lsm has conflicting keys {old!r} and {new!r}.")
            raw[new] = raw.pop(old)
    pretraining_raw = dict(cast(Mapping[str, object], raw.pop("pretraining", {})))
    for field in fields(LSMPretrainingConfig):
        if field.name in raw:
            if field.name in pretraining_raw and raw[field.name] != pretraining_raw[field.name]:
                raise ValueError(f"lsm has conflicting {field.name!r} pretraining values.")
            pretraining_raw[field.name] = raw.pop(field.name)
    if "epochs" not in pretraining_raw:
        pretraining_raw["epochs"] = pretrain_epochs
    lsm_names = {field.name for field in fields(LSMConfig)} - {"topology", "pretraining"}
    unknown = set(raw) - lsm_names
    if unknown:
        raise ValueError(f"lsm.{min(unknown)} is unknown.")
    raw["topology"] = "conv2d" if conv else "dense"
    raw["pretraining"] = LSMPretrainingConfig(**cast(Any, pretraining_raw))
    return PreprocessorConfig(
        LSMConfig(**cast(Any, raw)),
        PreprocessorTrainingConfig(trainable=trainable, lr=training_lr),
    )


def preprocessor_to_mapping(value: PreprocessorConfig) -> dict[str, object]:
    """Return a portable canonical mapping for an LSM component."""

    if not isinstance(value, PreprocessorConfig):
        raise TypeError("preprocessor must be a PreprocessorConfig.")
    if isinstance(value.component, ModulePreprocessorConfig):
        raise TypeError(
            "preprocessor.component.module cannot be represented as a portable mapping."
        )
    component = value.component
    pretraining = {
        field.name: getattr(component.pretraining, field.name)
        for field in fields(LSMPretrainingConfig)
    }
    lsm = {
        field.name: getattr(component, field.name)
        for field in fields(LSMConfig)
        if field.name != "pretraining"
    }
    lsm["pretraining"] = pretraining
    return {
        "kind": "lsm",
        "lsm": lsm,
        "training": {
            field.name: getattr(value.training, field.name)
            for field in fields(PreprocessorTrainingConfig)
        },
    }


__all__ = [
    "LSMConfig",
    "LSMPretrainingConfig",
    "ModulePreprocessorConfig",
    "PreprocessorConfig",
    "PreprocessorLike",
    "PreprocessorTrainingConfig",
    "normalize_preprocessor",
    "normalize_legacy_lsm",
    "preprocessor_to_mapping",
]
