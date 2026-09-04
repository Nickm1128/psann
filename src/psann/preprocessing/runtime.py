"""Runtime preparation for the canonical preprocessing boundary."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, cast

import numpy as np
import torch
import torch.nn as nn

from .._lsm.conv import LSMConv2d, LSMConv2dExpander
from .._lsm.dense import LSM, LSMExpander
from .config import LSMConfig, ModulePreprocessorConfig, PreprocessorConfig


@dataclass(frozen=True)
class PreprocessorCapabilities:
    input_topology: str
    output_topology: str
    output_dim: int
    supports_pretraining: bool
    supports_joint_training: bool
    serializable_kind: str


@dataclass(frozen=True)
class PreprocessorBuildRequest:
    config: PreprocessorConfig
    input_topology: str
    input_shape: tuple[int, ...]
    data: np.ndarray
    device: torch.device
    dtype: torch.dtype
    reconstruction_only: bool = False


@dataclass(frozen=True)
class PreprocessorBuildResult:
    module: nn.Module
    capabilities: PreprocessorCapabilities
    diagnostics: Mapping[str, object]
    controller: object | None = None


def declared_preprocessor_capabilities(config: PreprocessorConfig) -> PreprocessorCapabilities:
    """Return boundary metadata without constructing or pretraining a module.

    This preflight is deliberately separate from :func:`prepare_preprocessor`:
    invalid architecture compositions must fail before an expensive LSM
    reconstruction run can mutate its controller or consume accelerator time.
    """

    component = config.component
    if isinstance(component, ModulePreprocessorConfig):
        return PreprocessorCapabilities(
            component.input_topology,
            component.output_topology,
            component.output_dim,
            False,
            True,
            "module",
        )
    topology = "flat" if component.topology == "dense" else "spatial-2d"
    return PreprocessorCapabilities(
        topology,
        topology,
        component.output_dim,
        True,
        True,
        "lsm",
    )


def _lsm_expander(config: LSMConfig, device: torch.device) -> LSMExpander | LSMConv2dExpander:
    training = config.pretraining
    common: dict[str, Any] = {
        "hidden_layers": config.hidden_layers,
        "sparsity": config.sparsity,
        "nonlinearity": config.nonlinearity,
        "bias": config.bias,
        "epochs": training.epochs,
        "lr": training.lr,
        "ridge": training.ridge,
        "device": device,
        "random_state": config.random_state,
        "noisy": training.noisy,
        "noise_decay": training.noise_decay,
        "alpha_ortho": training.alpha_ortho,
        "alpha_sparse": training.alpha_sparse,
        "alpha_var": training.alpha_var,
        "target_var": training.target_var,
    }
    if config.topology == "dense":
        return LSMExpander(
            config.output_dim,
            hidden_units=config.hidden_units,
            # ``hidden_units`` is the canonical spelling.  Supplying the
            # historical alias as ``None`` keeps the underlying expander from
            # emitting a compatibility warning for a canonical configuration.
            hidden_width=None,
            batch_size=training.batch_size if training.batch_size is not None else 256,
            early_stopping=(
                training.early_stopping if training.early_stopping is not None else False
            ),
            patience=training.patience if training.patience is not None else 20,
            tol=training.tol if training.tol is not None else 1e-6,
            val_split=training.val_split,
            verbose=training.verbose if training.verbose is not None else 0,
            objective=training.objective if training.objective is not None else "r2",
            **cast(Any, common),
        )
    return LSMConv2dExpander(
        config.output_dim,
        conv_channels=config.hidden_units,
        hidden_channels=None,
        kernel_size=config.kernel_size or 1,
        **cast(Any, common),
    )


def prepare_preprocessor(request: PreprocessorBuildRequest) -> PreprocessorBuildResult:
    """Build or copy a module and initialize/pretrain it before core construction."""

    component = request.config.component
    if isinstance(component, ModulePreprocessorConfig):
        if request.input_topology != component.input_topology:
            raise ValueError(
                "preprocessor.component.input_topology conflicts with observed input topology: "
                f"declared {component.input_topology!r}, received {request.input_topology!r}."
            )
        module = deepcopy(component.module).to(device=request.device, dtype=request.dtype)
        capabilities = declared_preprocessor_capabilities(request.config)
        try:
            probe = torch.as_tensor(request.data[:1], device=request.device, dtype=request.dtype)
            with torch.no_grad():
                produced = module(probe)
        except Exception as exc:
            raise ValueError(
                "preprocessor.component.module cannot execute declared "
                f"{component.input_topology!r} input topology."
            ) from exc
        if not isinstance(produced, torch.Tensor):
            raise ValueError("preprocessor.component.module must return a torch.Tensor.")
        expected_rank = {
            "flat": 2,
            "tokens": 3,
            "spatial-1d": 3,
            "spatial-2d": 4,
            "spatial-3d": 5,
        }[component.output_topology]
        if produced.ndim != expected_rank:
            raise ValueError(
                "preprocessor.component.output_topology is incompatible with module output rank."
            )
        if produced.shape[0] != probe.shape[0]:
            raise ValueError("preprocessor.component.module must preserve batch dimension.")
        if component.output_topology == "tokens" and produced.shape[1] != probe.shape[1]:
            raise ValueError("preprocessor.component.module must preserve token length.")
        if (
            component.output_topology.startswith("spatial-")
            and produced.shape[2:] != probe.shape[2:]
        ):
            raise ValueError("preprocessor.component.module must preserve spatial dimensions.")
        if produced.shape[-1] != component.output_dim and not component.output_topology.startswith(
            "spatial-"
        ):
            raise ValueError(
                "preprocessor.component.output_dim conflicts with module output width."
            )
        if (
            component.output_topology.startswith("spatial-")
            and produced.shape[1] != component.output_dim
        ):
            raise ValueError(
                "preprocessor.component.output_dim conflicts with module output channels."
            )
        return PreprocessorBuildResult(module, capabilities, {}, None)

    expected = "flat" if component.topology == "dense" else "spatial-2d"
    if request.input_topology != expected:
        raise ValueError(
            f"preprocessor.component.topology={component.topology!r} requires {expected} input, "
            f"received {request.input_topology}."
        )
    expander = _lsm_expander(component, request.device)
    expander.fit(
        request.data,
        epochs=0 if request.reconstruction_only else component.pretraining.epochs,
    )
    module = cast(Any, expander).model
    if not isinstance(module, (LSM, LSMConv2d)):
        raise RuntimeError("preprocessor component did not create a graph module.")
    module = module.to(device=request.device, dtype=request.dtype)
    diagnostics: dict[str, object] = {}
    readout = getattr(cast(Any, expander), "W_", None)
    if isinstance(readout, torch.Tensor):
        diagnostics["ols_readout"] = readout.detach().cpu().clone()
    capabilities = declared_preprocessor_capabilities(request.config)
    return PreprocessorBuildResult(module, capabilities, diagnostics, expander)


def validate_preprocessor_capability(
    *,
    architecture_kind: str,
    attention: bool,
    convolutional: bool,
    spatial_ndim: int | None,
    capabilities: PreprocessorCapabilities,
    geometry_size: int | None = None,
) -> None:
    """Apply Phase 4 topology compatibility before predictive training."""

    output = capabilities.output_topology
    input_topology = capabilities.input_topology
    if architecture_kind == "sequence":
        if (input_topology, output) != ("tokens", "tokens"):
            raise ValueError(
                "preprocessor.component.input_topology and output_topology must both be tokens for sequence architecture."
            )
        return
    if attention:
        if (input_topology, output) != ("tokens", "tokens"):
            raise ValueError(
                "attention architecture requires preprocessor.component input_topology/output_topology tokens-to-tokens."
            )
        return
    if convolutional:
        expected = f"spatial-{spatial_ndim}d"
        if (input_topology, output) != (expected, expected):
            raise ValueError(
                f"convolutional architecture requires preprocessor.component input_topology/output_topology {expected}."
            )
        return
    if (input_topology, output) != ("flat", "flat"):
        raise ValueError(
            f"{architecture_kind} architecture requires preprocessor.component input_topology/output_topology flat."
        )
    if architecture_kind == "geometric-sparse" and geometry_size is not None:
        if capabilities.output_dim != geometry_size:
            raise ValueError(
                "geometric-sparse preprocessor output width must equal geometry.shape product."
            )


__all__ = [
    "PreprocessorBuildRequest",
    "PreprocessorBuildResult",
    "PreprocessorCapabilities",
    "declared_preprocessor_capabilities",
    "prepare_preprocessor",
    "validate_preprocessor_capability",
]
