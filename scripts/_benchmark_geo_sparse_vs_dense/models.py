# ruff: noqa: F403,F405
from __future__ import annotations

from .shared import *
from .shared import _build_geo_activation


class DenseMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dim: int,
        depth: int,
        activation_type: str = "relu",
        activation_config: Optional[Dict[str, Any]] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError("depth must be positive.")
        layers = []
        in_dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim, bias=bias))
            layers.append(_build_geo_activation(activation_type, hidden_dim, activation_config))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim, bias=bias))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.reshape(x.size(0), -1))


class DenseResidualBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        activation_type: str,
        activation_config: Optional[Dict[str, Any]] = None,
        norm: str = "rms",
        residual_alpha_init: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive.")
        norm_key = str(norm).lower()
        if norm_key == "none":
            self.norm = nn.Identity()
        elif norm_key == "layer":
            self.norm = nn.LayerNorm(int(dim))
        elif norm_key == "rms":
            self.norm = RMSNorm(int(dim))
        else:
            raise ValueError("norm must be one of: 'none', 'layer', 'rms'")

        self.fc1 = nn.Linear(int(dim), int(dim), bias=bool(bias))
        self.act = _build_geo_activation(str(activation_type), int(dim), activation_config)
        self.fc2 = nn.Linear(int(dim), int(dim), bias=bool(bias))
        self.alpha = nn.Parameter(torch.full((1,), float(residual_alpha_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.act(self.fc1(h))
        h = self.fc2(h)
        return x + self.alpha * h


class DenseResidualNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dim: int,
        depth: int,
        activation_type: str,
        activation_config: Optional[Dict[str, Any]] = None,
        norm: str = "rms",
        residual_alpha_init: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError("depth must be positive.")
        self.in_proj = nn.Linear(int(input_dim), int(hidden_dim), bias=bool(bias))
        self.blocks = nn.ModuleList(
            [
                DenseResidualBlock(
                    int(hidden_dim),
                    activation_type=str(activation_type),
                    activation_config=activation_config,
                    norm=norm,
                    residual_alpha_init=residual_alpha_init,
                    bias=bias,
                )
                for _ in range(int(depth))
            ]
        )
        self.head = nn.Linear(int(hidden_dim), int(output_dim), bias=bool(bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.in_proj(x.reshape(x.size(0), -1))
        for block in self.blocks:
            z = block(z)
        return self.head(z)


def _activation_param_multiplier(activation_type: str) -> int:
    key = str(activation_type).lower()
    if key in {"psann", "sine", "respsann"}:
        return 3
    if key == "phase_psann":
        return 4
    if key in {"relu_sigmoid_psann", "rspsann", "rsp", "clipped_psann"}:
        return 4
    return 0


def _dense_mlp_params_with_activation(
    *,
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    depth: int,
    activation_type: str,
    bias: bool = True,
) -> int:
    base = dense_mlp_params(
        input_dim=int(input_dim),
        output_dim=int(output_dim),
        hidden_dim=int(hidden_dim),
        depth=int(depth),
        bias=bool(bias),
    )
    extra = _activation_param_multiplier(activation_type) * int(hidden_dim) * int(depth)
    return int(base + extra)


def _match_dense_width_with_activation(
    *,
    target_params: int,
    input_dim: int,
    output_dim: int,
    depth: int,
    activation_type: str,
    bias: bool = True,
    max_width: int = 8192,
) -> Tuple[int, int]:
    if target_params <= 0:
        raise ValueError("target_params must be positive.")
    lo, hi = 1, int(max_width)
    best_width = 1
    best_mismatch = abs(
        _dense_mlp_params_with_activation(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=best_width,
            depth=depth,
            activation_type=activation_type,
            bias=bias,
        )
        - target_params
    )
    while lo <= hi:
        mid = (lo + hi) // 2
        params = _dense_mlp_params_with_activation(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=mid,
            depth=depth,
            activation_type=activation_type,
            bias=bias,
        )
        mismatch = abs(params - target_params)
        if mismatch < best_mismatch:
            best_mismatch = mismatch
            best_width = mid
        if params < target_params:
            lo = mid + 1
        else:
            hi = mid - 1
    return int(best_width), int(best_mismatch)


def _norm_param_count(norm: str, dim: int) -> int:
    key = str(norm).lower()
    if key == "none":
        return 0
    if key == "layer":
        return 2 * int(dim)  # weight + bias
    if key == "rms":
        return int(dim)  # RMSNorm scale
    raise ValueError("norm must be one of: 'none', 'layer', 'rms'")


def _dense_linear_param_count(in_features: int, out_features: int, *, bias: bool) -> int:
    return int(in_features) * int(out_features) + (int(out_features) if bias else 0)


def _dense_residual_params_with_activation(
    *,
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    depth: int,
    activation_type: str,
    norm: str = "rms",
    bias: bool = True,
) -> int:
    if depth <= 0:
        raise ValueError("depth must be positive.")
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive.")
    total = _dense_linear_param_count(input_dim, hidden_dim, bias=bias)
    total += _dense_linear_param_count(hidden_dim, output_dim, bias=bias)
    block = 0
    block += _norm_param_count(norm, hidden_dim)
    block += _dense_linear_param_count(hidden_dim, hidden_dim, bias=bias)
    block += _activation_param_multiplier(activation_type) * int(hidden_dim)
    block += _dense_linear_param_count(hidden_dim, hidden_dim, bias=bias)
    block += 1  # residual alpha
    total += int(depth) * int(block)
    return int(total)


def _match_dense_residual_width_with_activation(
    *,
    target_params: int,
    input_dim: int,
    output_dim: int,
    depth: int,
    activation_type: str,
    norm: str = "rms",
    bias: bool = True,
    max_width: int = 8192,
) -> Tuple[int, int]:
    if target_params <= 0:
        raise ValueError("target_params must be positive.")
    lo, hi = 1, int(max_width)
    best_width = 1
    best_mismatch = abs(
        _dense_residual_params_with_activation(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=best_width,
            depth=depth,
            activation_type=activation_type,
            norm=norm,
            bias=bias,
        )
        - target_params
    )
    while lo <= hi:
        mid = (lo + hi) // 2
        params = _dense_residual_params_with_activation(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=mid,
            depth=depth,
            activation_type=activation_type,
            norm=norm,
            bias=bias,
        )
        mismatch = abs(params - target_params)
        if mismatch < best_mismatch:
            best_mismatch = mismatch
            best_width = mid
        if params < target_params:
            lo = mid + 1
        else:
            hi = mid - 1
    return int(best_width), int(best_mismatch)
