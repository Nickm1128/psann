"""Runtime preparation for the canonical preprocessing boundary."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Mapping

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


def _lsm_expander(config: LSMConfig, device: torch.device) -> nn.Module:
    training = config.pretraining
    common = {
        "hidden_layers": config.hidden_layers,
        "sparsity": config.sparsity,
        "nonlinearity": config.nonlinearity,
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
            batch_size=training.batch_size if training.batch_size is not None else 256,
            early_stopping=(
                training.early_stopping if training.early_stopping is not None else False
            ),
            patience=training.patience if training.patience is not None else 20,
            tol=training.tol if training.tol is not None else 1e-6,
            val_split=training.val_split,
            verbose=training.verbose if training.verbose is not None else 0,
            objective=training.objective if training.objective is not None else "r2",
            **common,
        )
    return LSMConv2dExpander(
        config.output_dim,
        conv_channels=config.hidden_units,
        kernel_size=config.kernel_size or 1,
        **common,
    )


def prepare_preprocessor(request: PreprocessorBuildRequest) -> PreprocessorBuildResult:
    """Build or copy a module and initialize/pretrain it before core construction."""

    component = request.config.component
    if isinstance(component, ModulePreprocessorConfig):
        module = deepcopy(component.module).to(device=request.device, dtype=request.dtype)
        capabilities = PreprocessorCapabilities(
            component.input_topology,
            component.output_topology,
            component.output_dim,
            False,
            True,
            "module",
        )
        return PreprocessorBuildResult(module, capabilities, {})

    expected = "flat" if component.topology == "dense" else "spatial-2d"
    if request.input_topology != expected:
        raise ValueError(
            f"preprocessor.component.topology={component.topology!r} requires {expected} input, "
            f"received {request.input_topology}."
        )
    expander = _lsm_expander(component, request.device)
    expander.fit(request.data, epochs=component.pretraining.epochs)
    module = expander.model
    if not isinstance(module, (LSM, LSMConv2d)):
        raise RuntimeError("preprocessor component did not create a graph module.")
    module = module.to(device=request.device, dtype=request.dtype)
    diagnostics: dict[str, object] = {}
    if getattr(expander, "W_", None) is not None:
        diagnostics["ols_readout"] = expander.W_.detach().cpu().clone()
    capabilities = PreprocessorCapabilities(
        expected,
        expected,
        component.output_dim,
        True,
        True,
        "lsm",
    )
    return PreprocessorBuildResult(module, capabilities, diagnostics)


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
    if architecture_kind == "sequence":
        if output != "tokens":
            raise ValueError(
                "preprocessor output topology must be tokens for sequence architecture."
            )
        return
    if attention:
        if output != "tokens":
            raise ValueError("attention architecture requires a tokens-to-tokens preprocessor.")
        return
    if convolutional:
        expected = f"spatial-{spatial_ndim}d"
        if output != expected:
            raise ValueError(f"convolutional architecture requires {expected} preprocessor output.")
        return
    if output != "flat":
        raise ValueError(f"{architecture_kind} architecture requires flat preprocessor output.")
    if architecture_kind == "geometric-sparse" and geometry_size is not None:
        if capabilities.output_dim != geometry_size:
            raise ValueError(
                "geometric-sparse preprocessor output width must equal geometry.shape product."
            )


__all__ = [
    "PreprocessorBuildRequest",
    "PreprocessorBuildResult",
    "PreprocessorCapabilities",
    "prepare_preprocessor",
    "validate_preprocessor_capability",
]
