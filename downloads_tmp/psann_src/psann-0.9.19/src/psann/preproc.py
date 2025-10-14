from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .lsm import LSM, LSMExpander, LSMConv2d, LSMConv2dExpander


@dataclass
class PreprocessorSpec:
    name: str
    params: Dict[str, Any]


def _instantiated(module: Any) -> Tuple[nn.Module | None, Optional[nn.Module]]:
    if isinstance(module, nn.Module):
        return module, getattr(module, 'model', None)
    return None, None


def _ensure_numpy(data: Any) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return np.asarray(data, dtype=np.float32)
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data, dtype=np.float32)


def _maybe_init_expander(expander: Any, *, allow_train: bool, pretrain_epochs: int, data: Any) -> None:
    if data is None and getattr(expander, 'model', None) is not None:
        return
    epochs = int(pretrain_epochs or 0)
    needs_init = getattr(expander, 'model', None) is None
    should_train = allow_train and epochs > 0
    if not needs_init and not should_train:
        return
    if data is None:
        raise RuntimeError("Preprocessor requires data to initialise but no data was provided")
    arr = _ensure_numpy(data)
    expander.fit(arr, epochs=epochs if allow_train else 0)


def _build_lsm(params: Dict[str, Any]) -> nn.Module:
    p = dict(params)
    width = int(p.pop('hidden_units', p.pop('hidden_width', 128)))
    p.setdefault('hidden_units', width)
    p.setdefault('hidden_width', width)
    return LSM(**p)


def _build_lsm_expander(params: Dict[str, Any]) -> LSMExpander:
    p = dict(params)
    width = int(p.pop('hidden_units', p.pop('hidden_width', 128)))
    p.setdefault('hidden_units', width)
    p.setdefault('hidden_width', width)
    return LSMExpander(**p)


def _build_lsm_conv(params: Dict[str, Any]) -> LSMConv2d:
    p = dict(params)
    channels = int(p.pop('conv_channels', p.pop('hidden_channels', 128)))
    p.setdefault('conv_channels', channels)
    p.setdefault('hidden_channels', channels)
    return LSMConv2d(**p)


def _build_lsm_conv_expander(params: Dict[str, Any]) -> LSMConv2dExpander:
    p = dict(params)
    channels = int(p.pop('conv_channels', p.pop('hidden_channels', 128)))
    p.setdefault('conv_channels', channels)
    p.setdefault('hidden_channels', channels)
    return LSMConv2dExpander(**p)


_BUILDERS: Dict[str, Any] = {
    'lsm': _build_lsm,
    'lsmexpander': _build_lsm_expander,
    'lsmconv2d': _build_lsm_conv,
    'lsmconv2dexpander': _build_lsm_conv_expander,
}


def _resolve_spec(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, PreprocessorSpec):
        spec = dict(value.params)
        builder = _BUILDERS.get(value.name.lower())
        if builder is None:
            raise ValueError(f"Unknown preprocessor type '{value.name}'")
        return builder(spec)
    if isinstance(value, dict):
        spec = dict(value)
        name = spec.get('type') or spec.get('kind') or spec.get('name')
        if name:
            spec.pop('type', None)
            spec.pop('kind', None)
            spec.pop('name', None)
            builder = _BUILDERS.get(str(name).lower())
            if builder is None:
                raise ValueError(f"Unknown preprocessor type '{name}'")
            return builder(spec)
        out_dim = int(spec.pop('output_dim', spec.pop('out_channels', 128)))
        if spec.pop('conv', False):
            return LSMConv2dExpander(out_channels=out_dim, **spec)
        return LSMExpander(output_dim=out_dim, **spec)
    return value


def build_preprocessor(
    value: Any,
    *,
    allow_train: bool = False,
    pretrain_epochs: int = 0,
    data: Optional[Any] = None,
) -> Tuple[nn.Module | None, Optional[nn.Module]]:
    """Normalise user-supplied preprocessor specifications.

    Args:
        value: Existing module, `PreprocessorSpec`, config dictionary, or `None`.
        allow_train: When `True`, allow expanders to run their own `fit` routine.
        pretrain_epochs: Number of epochs to run when training is allowed.
        data: Optional raw array used to initialise or warm-start expanders (passed to `fit`).

    Returns:
        Tuple `(module, base_model)` where `module` plugs into PSANN estimators and `base_model` exposes
        transform helpers (e.g. the underlying LSMExpander model) when available.

    Raises:
        ValueError: If the specification type is unknown or incompatible with the requested mode.

    Notes:
        Expanders are fitted in place when `allow_train` is true; provide data with the same shape you plan to
        feed into the estimator to avoid shape mismatches.
    """
    resolved = _resolve_spec(value)
    if resolved is None:
        return None, None

    module, model = _instantiated(resolved)
    if module is not None:
        base = model if model is not None else module
        return module, base

    if isinstance(resolved, (LSMExpander, LSMConv2dExpander)):
        _maybe_init_expander(resolved, allow_train=allow_train, pretrain_epochs=pretrain_epochs, data=data)
        return resolved, getattr(resolved, 'model', None)

    raise ValueError("Unsupported preprocessor specification")


__all__ = ['PreprocessorSpec', 'build_preprocessor']
