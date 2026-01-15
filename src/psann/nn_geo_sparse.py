from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn

from .activations import MixedActivation, PhaseSineParam, SineParam
from .layers.geo_sparse import GeoSparseLinear, build_geo_connectivity
from .layers.sine_residual import RMSNorm
from .nn import DropPath
from .types import ActivationConfig


class GeoSparseResidualBlock(nn.Module):
    """Pre-norm residual block using sparse geometric linear layers."""

    def __init__(
        self,
        features: int,
        in_index_per_out: torch.Tensor,
        *,
        activation_type: str = "psann",
        activation_config: Optional[ActivationConfig | Mapping[str, Any]] = None,
        norm: str = "rms",
        drop_path: float = 0.0,
        residual_alpha_init: float = 0.0,
        bias: bool = True,
        compute_mode: str = "gather",
    ) -> None:
        super().__init__()
        if features <= 0:
            raise ValueError("features must be positive.")
        if in_index_per_out.shape[0] != features:
            raise ValueError("in_index_per_out first dimension must match features.")
        self.features = int(features)
        self.norm = _build_norm(norm, features)
        self.fc1 = GeoSparseLinear(
            features, features, in_index_per_out, bias=bias, compute_mode=compute_mode
        )
        self.act = _build_activation(activation_type, features, activation_config)
        self.fc2 = GeoSparseLinear(
            features, features, in_index_per_out, bias=bias, compute_mode=compute_mode
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.alpha = nn.Parameter(torch.full((1,), float(residual_alpha_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.act(self.fc1(h))
        h = self.fc2(h)
        h = self.drop_path(h)
        return x + self.alpha * h


class GeoSparseNet(nn.Module):
    """Deep residual network built from geometric sparse layers."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        shape: Tuple[int, int],
        depth: int = 4,
        k: int = 8,
        pattern: str = "local",
        radius: int = 1,
        offsets: Optional[Sequence[Tuple[int, int]]] = None,
        wrap_mode: str = "clamp",
        activation_type: str = "psann",
        activation_config: Optional[ActivationConfig | Mapping[str, Any]] = None,
        norm: str = "rms",
        drop_path_max: float = 0.0,
        residual_alpha_init: float = 0.0,
        bias: bool = True,
        compute_mode: str = "gather",
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError("depth must be positive.")
        height, width = int(shape[0]), int(shape[1])
        if height <= 0 or width <= 0:
            raise ValueError("shape dimensions must be positive.")
        self.shape = (height, width)
        self.features = height * width
        if int(input_dim) != self.features:
            raise ValueError("input_dim must match shape height * width.")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive.")

        self.depth = int(depth)
        self.k = int(k)
        self.pattern = str(pattern)
        self.radius = int(radius)
        self.wrap_mode = str(wrap_mode)
        self.activation_type = str(activation_type)
        self.activation_config = deepcopy(activation_config) if activation_config else None
        self.compute_mode = str(compute_mode)
        self.seed = seed

        blocks = []
        for idx in range(self.depth):
            block_seed = None if seed is None else int(seed) + idx * 9973
            indices = build_geo_connectivity(
                self.shape,
                k=self.k,
                pattern=self.pattern,
                radius=self.radius,
                offsets=offsets,
                wrap_mode=self.wrap_mode,
                seed=block_seed,
            )
            block_activation_config = deepcopy(activation_config) if activation_config else None
            if isinstance(block_activation_config, Mapping):
                block_activation_config = dict(block_activation_config)
                block_activation_config.setdefault("mix_seed", block_seed)
            drop_path = (
                float(drop_path_max) * (idx / max(1, self.depth - 1))
                if self.depth > 1
                else 0.0
            )
            block = GeoSparseResidualBlock(
                self.features,
                indices,
                activation_type=self.activation_type,
                activation_config=block_activation_config,
                norm=norm,
                drop_path=drop_path,
                residual_alpha_init=residual_alpha_init,
                bias=bias,
                compute_mode=self.compute_mode,
            )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(self.features, int(output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _flatten_geo_input(x, self.shape)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def _build_activation(
    activation_type: str,
    out_features: int,
    activation_config: Optional[ActivationConfig | Mapping[str, Any]],
) -> nn.Module:
    act = activation_type.lower()
    cfg, phase_cfg = _normalize_activation_config(out_features, activation_config)
    if act == "mixed":
        raw = _as_mapping(activation_config)
        types_raw = raw.get("activation_types", raw.get("types"))
        if not isinstance(types_raw, (list, tuple)) or not types_raw:
            raise ValueError(
                "activation_type='mixed' requires activation_config['activation_types'] "
                "(a non-empty list of activation names)"
            )
        ratios_raw = raw.get("activation_ratios", raw.get("ratios"))
        ratios = None
        if ratios_raw is not None:
            if not isinstance(ratios_raw, (list, tuple)):
                raise ValueError("activation_ratios must be a list of floats when provided")
            ratios = [float(x) for x in ratios_raw]
        ratio_sum_tol = float(raw.get("ratio_sum_tol", 1e-3))
        layout = str(raw.get("mix_layout", raw.get("layout", "random")))
        seed = raw.get("mix_seed", None)
        if seed is None:
            seed = raw.get("seed", None)
        seed_int = None if seed is None else int(seed)

        def _build_psann(n: int) -> nn.Module:
            cfg_n, _ = _normalize_activation_config(n, activation_config)
            return SineParam(n, **cfg_n)

        def _build_phase_psann(n: int) -> nn.Module:
            cfg_n, phase_n = _normalize_activation_config(n, activation_config)
            return PhaseSineParam(
                n,
                phase_init=phase_n["phase_init"],
                phase_trainable=phase_n["phase_trainable"],
                **cfg_n,
            )

        builders = {
            "psann": _build_psann,
            "phase_psann": _build_phase_psann,
        }
        return MixedActivation(
            out_features,
            activation_types=[str(t) for t in types_raw],
            activation_ratios=ratios,
            ratio_sum_tol=ratio_sum_tol,
            seed=seed_int,
            layout=layout,
            feature_dim=int(raw.get("feature_dim", -1)),
            builders=builders,
        )
    if act == "psann":
        return SineParam(out_features, **cfg)
    if act == "phase_psann":
        return PhaseSineParam(
            out_features,
            phase_init=phase_cfg["phase_init"],
            phase_trainable=phase_cfg["phase_trainable"],
            **cfg,
        )
    if act == "relu":
        return nn.ReLU()
    if act == "tanh":
        return nn.Tanh()
    raise ValueError(
        "activation_type must be one of: 'psann', 'phase_psann', 'mixed', 'relu', 'tanh'"
    )


def _build_norm(norm: str, features: int) -> nn.Module:
    key = norm.lower()
    if key == "none":
        return nn.Identity()
    if key == "layer":
        return nn.LayerNorm(features)
    if key == "rms":
        return RMSNorm(features)
    raise ValueError("norm must be one of: 'none', 'layer', 'rms'")


def _flatten_geo_input(x: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    height, width = shape
    if x.ndim >= 3 and int(x.shape[-2]) == height and int(x.shape[-1]) == width:
        return x.reshape(*x.shape[:-2], height * width)
    if x.ndim >= 2 and int(x.shape[-1]) == height * width:
        return x
    raise ValueError("Input must have shape (..., H, W) or (..., H*W).")


def _normalize_activation_config(
    out_features: int,
    activation_config: Optional[ActivationConfig | Mapping[str, Any]],
) -> Tuple[dict, dict]:
    raw = _as_mapping(activation_config)
    cfg: dict = {}

    amp_init = raw.get("amplitude_init", raw.get("amp_init"))
    freq_init = raw.get("frequency_init", raw.get("freq_init"))
    damp_init = raw.get("decay_init", raw.get("damp_init"))

    amp_init = _maybe_sample_init(out_features, amp_init, raw, "amp_init_std", "amp_range")
    freq_init = _maybe_sample_init(out_features, freq_init, raw, "freq_init_std", "freq_range")
    damp_init = _maybe_sample_init(out_features, damp_init, raw, "damp_init_std", "damp_range")

    if amp_init is not None:
        cfg["amplitude_init"] = amp_init
    if freq_init is not None:
        cfg["frequency_init"] = freq_init
    if damp_init is not None:
        cfg["decay_init"] = damp_init

    if "learnable" in raw:
        cfg["learnable"] = raw["learnable"]
    elif "trainable" in raw:
        cfg["learnable"] = ("amplitude", "frequency", "decay") if raw["trainable"] else ()

    if "decay_mode" in raw:
        cfg["decay_mode"] = raw["decay_mode"]
    if "feature_dim" in raw:
        cfg["feature_dim"] = raw["feature_dim"]

    bounds = {}
    if isinstance(raw.get("bounds"), Mapping):
        bounds.update(raw["bounds"])
    if raw.get("amp_bounds") is not None:
        bounds["amplitude"] = raw["amp_bounds"]
    if raw.get("freq_bounds") is not None:
        bounds["frequency"] = raw["freq_bounds"]
    if raw.get("damp_bounds") is not None:
        bounds["decay"] = raw["damp_bounds"]
    if bounds:
        cfg["bounds"] = bounds

    phase_cfg = {
        "phase_init": float(raw.get("phase_init", 0.0)),
        "phase_trainable": bool(raw.get("phase_trainable", True)),
    }
    return cfg, phase_cfg


def _as_mapping(cfg: Optional[ActivationConfig | Mapping[str, Any]]) -> dict:
    if cfg is None:
        return {}
    if isinstance(cfg, Mapping):
        return dict(cfg)
    if hasattr(cfg, "__dict__"):
        return {k: v for k, v in vars(cfg).items() if not k.startswith("_")}
    return {}


def _maybe_sample_init(
    out_features: int,
    base_value: Any,
    raw: Mapping[str, Any],
    std_key: str,
    range_key: str,
):
    std = raw.get(std_key)
    if std is not None and float(std) > 0:
        mean = 0.0 if base_value is None else float(base_value)
        return torch.randn(out_features, dtype=torch.float32) * float(std) + mean
    rng = raw.get(range_key)
    if rng is not None:
        lo, hi = float(rng[0]), float(rng[1])
        if hi < lo:
            lo, hi = hi, lo
        return torch.empty(1, dtype=torch.float32).uniform_(lo, hi).item()
    return base_value
