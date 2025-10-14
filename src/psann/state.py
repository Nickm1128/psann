from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Union

import torch
import torch.nn as nn

__all__ = [
    "StateConfig",
    "ensure_state_config",
    "StateController",
]


@dataclass(frozen=True)
class StateConfig:
    """Immutable configuration for PSANN state controllers."""

    rho: float = 0.95
    beta: float = 1.0
    init: float = 1.0
    max_abs: float = 5.0
    detach: bool = True

    def __post_init__(self) -> None:
        if not (0.0 <= float(self.rho) < 1.0):
            raise ValueError("StateConfig.rho must satisfy 0 <= rho < 1.")
        if float(self.max_abs) <= 0.0:
            raise ValueError("StateConfig.max_abs must be positive.")
        if not math.isfinite(float(self.init)):
            raise ValueError("StateConfig.init must be finite.")
        if float(self.beta) < 0.0:
            raise ValueError("StateConfig.beta must be non-negative.")

    def to_kwargs(self) -> dict[str, Any]:
        """Return kwargs consumable by :class:`StateController`."""

        return {
            "rho": float(self.rho),
            "beta": float(self.beta),
            "init": float(self.init),
            "max_abs": float(self.max_abs),
            "detach": bool(self.detach),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "StateConfig":
        """Coerce a mapping of controller parameters into ``StateConfig``."""

        allowed = {"rho", "beta", "init", "max_abs", "detach"}
        unknown = set(mapping.keys()) - allowed
        if unknown:
            unknown_list = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown StateConfig keys: {unknown_list}")
        return cls(
            rho=mapping.get("rho", cls.rho),
            beta=mapping.get("beta", cls.beta),
            init=mapping.get("init", cls.init),
            max_abs=mapping.get("max_abs", cls.max_abs),
            detach=mapping.get("detach", cls.detach),
        )


StateConfigLike = Union[StateConfig, Mapping[str, Any]]


def ensure_state_config(value: Optional[StateConfigLike]) -> Optional[StateConfig]:
    """Normalise user-specified ``state`` arguments to ``StateConfig``."""

    if value is None:
        return None
    if isinstance(value, StateConfig):
        return value
    if isinstance(value, Mapping):
        return StateConfig.from_mapping(value)
    raise TypeError("state must be a mapping or a StateConfig instance.")


class StateController(nn.Module):
    """Persistent per-feature state controller used by stateful PSANN blocks.

    Args:
        size: Number of features; must match the length of the feature axis being modulated.
        init: Initial state value applied to every feature.
        rho: Exponential decay coefficient in `[0, 1)` that controls persistence.
        beta: Multiplier for the mean activation magnitude used to refresh the state.
        max_abs: Soft bound applied via `tanh` to keep the state in `[-max_abs, max_abs]`.
        detach: When true, forward passes use a detached view of the state to avoid autograd warnings.

    The controller multiplies activations by the current state and stages updates computed from the mean absolute
    value of `y` across all axes except the feature dimension supplied to `apply()`. Call `commit()` after the
    optimiser step to apply buffered updates, or `reset`/`reset_like_init` to reinitialise the state.
    """

    def __init__(
        self,
        size: int,
        *,
        init: float = 1.0,
        rho: float = 0.95,
        beta: float = 1.0,
        max_abs: float = 5.0,
        detach: bool = True,
    ) -> None:
        super().__init__()
        assert size > 0
        self.size = int(size)
        self.rho = float(rho)
        self.beta = float(beta)
        self.max_abs = float(max_abs)
        self.detach = bool(detach)

        state = torch.full((self.size,), float(init))
        self.register_buffer("state", state)
        self._warn_high_emitted: bool = False
        self._warn_low_emitted: bool = False

    def reset(self, value: Optional[float] = None) -> None:
        v = float(value) if value is not None else float(self.state.mean().item())
        with torch.no_grad():
            self.state.fill_(v)
        self._warn_high_emitted = False
        self._warn_low_emitted = False

    def reset_like_init(self, init: float = 1.0) -> None:
        with torch.no_grad():
            self.state.fill_(float(init))
        self._warn_high_emitted = False
        self._warn_low_emitted = False

    def apply(self, y: torch.Tensor, feature_dim: int, update: bool = True) -> torch.Tensor:
        """Scale activations with the persisted state and optionally stage updates.

        Args:
            y: Tensor of activations to modulate.
            feature_dim: Axis index for the feature dimension; negative indices are supported.
            update: When true, accumulate a pending update using the mean absolute value of `y`.

        Returns:
            Tensor with the same shape as the input after scaling by the current state.
        """
        # Use a detached copy of state for forward to avoid versioning issues
        fd = feature_dim if feature_dim >= 0 else (y.ndim + feature_dim)
        shape = [1] * y.ndim
        shape[fd] = self.size
        # Use detached state for forward if detach=True, else use live buffer
        s = (self.state.detach() if self.detach else self.state).view(*shape)
        y_scaled = y * s

        if update:
            # Compute proposed new state based on scaled activation magnitude
            reduce_dims = [d for d in range(y_scaled.ndim) if d != fd]
            if reduce_dims:
                m = y_scaled.abs().mean(dim=reduce_dims)
            else:
                m = y_scaled.abs()
            new_s = self.rho * self.state + (1.0 - self.rho) * (self.beta * m)
            new_s = self.max_abs * torch.tanh(new_s / max(self.max_abs, 1e-6))
            new_s = new_s.clamp(-self.max_abs, self.max_abs)
            self._maybe_warn_bounds(new_s)
            if self.detach:
                new_s = new_s.detach()
            # Defer commit until after backward
            self._pending_state = new_s
        return y_scaled

    def commit(self) -> None:
        """Apply any pending state updates staged during forward passes."""
        new_s = getattr(self, "_pending_state", None)
        if new_s is None:
            return
        with torch.no_grad():
            self.state.copy_(new_s)
        self._pending_state = None

    def _maybe_warn_bounds(self, new_state: torch.Tensor) -> None:
        """Emit warnings if the proposed state approaches saturation or collapse."""

        if not torch.is_tensor(new_state):
            return
        with torch.no_grad():
            abs_vals = new_state.abs()
            if abs_vals.numel() == 0:
                return
            max_val = float(abs_vals.max().item())
            mean_val = float(abs_vals.mean().item())

        high_threshold = float(self.max_abs) * 0.98
        if max_val >= high_threshold and not self._warn_high_emitted:
            warnings.warn(
                "StateController persistent state magnitude reached 98% of max_abs; "
                "consider reducing beta or increasing rho to avoid saturation.",
                RuntimeWarning,
                stacklevel=3,
            )
            self._warn_high_emitted = True

        low_threshold = 1e-3
        if mean_val <= low_threshold and not self._warn_low_emitted:
            warnings.warn(
                "StateController persistent state magnitude collapsed below 1e-3; "
                "consider increasing beta or the initial state.",
                RuntimeWarning,
                stacklevel=3,
            )
            self._warn_low_emitted = True
