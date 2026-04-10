from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .._aliases import resolve_int_alias
from ..utils import choose_device, seed_all
from .common import TensorLike, _tensor_to_output, _to_float_tensor


class MaskedConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        bias: bool = True,
        sparsity: float = 0.8,
        random_state: Optional[int] = None,
    ):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, padding=0, bias=bias)
        rs = torch.Generator()
        if random_state is not None:
            rs.manual_seed(int(random_state))
        density = max(0.0, min(1.0, 1.0 - float(sparsity)))
        mask = (
            torch.rand(
                (out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]),
                generator=rs,
            )
            < density
        ).float()
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight * self.mask
        return F.conv2d(
            x,
            w,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class LSMConv2d(nn.Module):
    """Conv2d expander using masked convolutions with configurable nonlinearity."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        hidden_layers: int = 1,
        conv_channels: Optional[int] = None,
        hidden_channels: Optional[int] = 128,
        kernel_size: int = 1,
        sparsity: float = 0.8,
        nonlinearity: str = "sine",
        bias: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        channels_res = resolve_int_alias(
            primary_value=conv_channels,
            alias_value=hidden_channels,
            primary_name="conv_channels",
            alias_name="hidden_channels",
            context="LSMConv2d",
            default=128,
        )
        channels = channels_res.value if channels_res.value is not None else 128
        self.hidden_layers = int(hidden_layers)
        self.conv_channels = channels
        self.hidden_channels = channels
        self.kernel_size = int(kernel_size)
        act = {"sine": torch.sin, "tanh": torch.tanh, "relu": F.relu}.get(nonlinearity)
        if act is None:
            raise ValueError("nonlinearity must be one of: sine, tanh, relu")
        self._act = act

        layers = []
        current = in_channels
        for i in range(hidden_layers):
            layers.append(
                MaskedConv2d(
                    current,
                    channels,
                    kernel_size=self.kernel_size,
                    bias=bias,
                    sparsity=sparsity,
                    random_state=None if random_state is None else random_state + i,
                )
            )
            current = channels
        self.body = nn.Sequential(*layers)
        self.head = MaskedConv2d(
            current,
            out_channels,
            kernel_size=self.kernel_size,
            bias=bias,
            sparsity=sparsity,
            random_state=None if random_state is None else random_state + 777,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x
        for layer in self.body:
            z = self._act(layer(z))
        return self.head(z)


class LSMConv2dExpander(nn.Module):
    """Pretraining interface for :class:`LSMConv2d` with OLS per-pixel objective."""

    def __init__(
        self,
        out_channels: int,
        *,
        hidden_layers: int = 1,
        conv_channels: Optional[int] = None,
        hidden_channels: Optional[int] = 128,
        kernel_size: int = 1,
        sparsity: float = 0.8,
        nonlinearity: str = "sine",
        epochs: int = 50,
        lr: float = 1e-3,
        ridge: float = 1e-4,
        device: str | torch.device = "auto",
        random_state: Optional[int] = None,
        noisy: Optional[float] = None,
        noise_decay: float = 1.0,
        alpha_ortho: float = 0.0,
        alpha_sparse: float = 0.0,
        alpha_var: float = 0.0,
        target_var: float = 1.0,
    ) -> None:
        super().__init__()
        self.out_channels = int(out_channels)
        channels_res = resolve_int_alias(
            primary_value=conv_channels,
            alias_value=hidden_channels,
            primary_name="conv_channels",
            alias_name="hidden_channels",
            context="LSMConv2dExpander",
            default=128,
        )
        channels = channels_res.value if channels_res.value is not None else 128
        self.hidden_layers = int(hidden_layers)
        self.conv_channels = channels
        self.hidden_channels = channels
        self.kernel_size = int(kernel_size)
        self.sparsity = sparsity
        self.nonlinearity = nonlinearity
        self.epochs = epochs
        self.lr = lr
        self.ridge = ridge
        self.device = device
        self.random_state = random_state
        self.model: Optional[LSMConv2d] = None
        self.W_: Optional[torch.Tensor] = None
        self.noisy = noisy
        self.noise_decay = float(noise_decay)
        self.alpha_ortho = float(alpha_ortho)
        self.alpha_sparse = float(alpha_sparse)
        self.alpha_var = float(alpha_var)
        self.target_var = float(target_var)

    def _device(self) -> torch.device:
        return choose_device(self.device)

    def _ols_readout(self, Z: torch.Tensor, X: torch.Tensor, ridge: float) -> torch.Tensor:
        _, C_out, _, _ = Z.shape
        Zf = Z.permute(0, 2, 3, 1).reshape(-1, C_out)
        Cin = X.shape[1]
        ones = torch.ones((Zf.shape[0], 1), dtype=Z.dtype, device=Z.device)
        Zb = torch.cat([Zf, ones], dim=1)
        A = Zb.T @ Zb
        Xf = X.permute(0, 2, 3, 1).reshape(-1, Cin)
        b = Zb.T @ Xf
        eye = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
        lam = float(ridge if ridge is not None else 0.0)
        lam = lam if lam > 0 else 1e-8
        W = None
        for _ in range(6):
            try:
                W = torch.linalg.solve(A + lam * eye, b)
                break
            except Exception:
                lam *= 10.0
        if W is None:
            try:
                W = torch.linalg.lstsq(Zb, Xf, rcond=None).solution
            except Exception:
                W = torch.linalg.pinv(Zb) @ Xf
        return W

    def fit(self, X: TensorLike, epochs: Optional[int] = None) -> "LSMConv2dExpander":
        seed_all(self.random_state)
        device = self._device()
        X_all_t, _, _, _ = _to_float_tensor(X, device=device)
        if X_all_t.ndim != 4:
            raise AssertionError("Expected channels-first (N, C, H, W) input")
        Cin = int(X_all_t.shape[1])
        self.model = LSMConv2d(
            Cin,
            self.out_channels,
            hidden_layers=self.hidden_layers,
            hidden_channels=self.hidden_channels,
            kernel_size=self.kernel_size,
            sparsity=self.sparsity,
            nonlinearity=self.nonlinearity,
            random_state=self.random_state,
        ).to(device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        E = int(epochs) if epochs is not None else self.epochs

        noise_std_t = None
        if self.noisy is not None and float(self.noise_decay) >= 0.0:
            if np.isscalar(self.noisy):
                std = np.full((1, Cin, 1, 1), float(self.noisy), dtype=np.float32)
            else:
                arr = np.asarray(self.noisy, dtype=np.float32)
                if arr.ndim == 1 and arr.shape[0] == Cin:
                    std = arr.reshape(1, Cin, 1, 1)
                elif arr.shape == (Cin, 1, 1):
                    std = arr.reshape(1, Cin, 1, 1)
                else:
                    raise ValueError(f"noisy shape {arr.shape} incompatible with channels={Cin}")
            noise_std_t = torch.from_numpy(std).to(device)

        for epoch in range(E):
            self.model.train()
            opt.zero_grad()
            if noise_std_t is not None:
                factor = float(max(self.noise_decay, 0.0) ** epoch)
                X_in = X_all_t + torch.randn_like(X_all_t) * (noise_std_t * factor)
            else:
                X_in = X_all_t
            Z = self.model(X_in)
            W = self._ols_readout(Z, X_all_t, ridge=self.ridge)
            Zf = Z.permute(0, 2, 3, 1).reshape(-1, Z.shape[1])
            ones = torch.ones((Zf.shape[0], 1), dtype=Z.dtype, device=Z.device)
            Zbf = torch.cat([Zf, ones], dim=1)
            Xf_hat = Zbf @ W
            Xf = X_all_t.permute(0, 2, 3, 1).reshape(-1, Cin)
            ss_res = ((Xf - Xf_hat) ** 2).sum()
            ss_tot = ((Xf - Xf.mean(dim=0, keepdim=True)) ** 2).sum().clamp_min(1e-8)
            reg = 0.0
            if self.alpha_ortho > 0.0:
                Zc = Zf - Zf.mean(dim=0, keepdim=True)
                Cz = (Zc.T @ Zc) / max(1, Zf.shape[0] - 1)
                offdiag = Cz - torch.diag(torch.diag(Cz))
                reg = reg + self.alpha_ortho * (offdiag.pow(2).sum() / (Zf.shape[1] ** 2))
            if self.alpha_sparse > 0.0:
                reg = reg + self.alpha_sparse * Zf.abs().mean()
            if self.alpha_var > 0.0:
                var = Zf.var(dim=0, unbiased=False)
                reg = reg + self.alpha_var * ((var - self.target_var) ** 2).mean()
            (ss_res / ss_tot + reg).backward()
            opt.step()

        self.model.eval()
        with torch.no_grad():
            Z = self.model(X_all_t)
            self.W_ = self._ols_readout(Z, X_all_t, ridge=self.ridge).detach().cpu()
        return self

    def transform(self, X: TensorLike) -> TensorLike:
        if self.model is None:
            raise RuntimeError("LSMConv2dExpander not fitted")
        model_device = next(self.model.parameters()).device
        X_t, is_tensor, orig_device, orig_dtype = _to_float_tensor(X, device=model_device)
        if X_t.ndim != 4:
            raise ValueError("Expected channels-first (N, C, H, W) input")
        with torch.no_grad():
            Z = self.model(X_t)
        return _tensor_to_output(
            Z, return_tensor=is_tensor, target_device=orig_device, target_dtype=orig_dtype
        )

    def fit_transform(self, X: TensorLike) -> TensorLike:
        self.fit(X)
        return self.transform(X)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if not isinstance(X, torch.Tensor):
            raise TypeError("forward expects a torch.Tensor input")
        if self.model is None:
            raise RuntimeError("LSMConv2dExpander not fitted; call fit() before forward().")
        device = next(self.model.parameters()).device
        return self.model(X.to(device=device, dtype=torch.float32))

    def to(self, *args, **kwargs) -> "LSMConv2dExpander":
        super().to(*args, **kwargs)
        if self.model is not None:
            self.model = self.model.to(*args, **kwargs)
        if self.W_ is not None:
            self.W_ = self.W_.to(*args, **kwargs)
        return self

    def train(self, mode: bool = True) -> "LSMConv2dExpander":
        super().train(mode)
        if self.model is not None:
            self.model.train(mode)
        return self

    def eval(self) -> "LSMConv2dExpander":
        return self.train(False)

    def score_reconstruction(self, X: TensorLike) -> float:
        if self.model is None or self.W_ is None:
            raise RuntimeError("LSMConv2dExpander not fitted")
        device = next(self.model.parameters()).device
        X_t, _, _, _ = _to_float_tensor(X, device=device)
        if X_t.ndim != 4:
            raise ValueError("Expected channels-first (N, C, H, W) input")
        with torch.no_grad():
            Z = self.model(X_t)
            Zf = Z.permute(0, 2, 3, 1).reshape(-1, Z.shape[1])
            ones = torch.ones((Zf.shape[0], 1), dtype=Z.dtype, device=Z.device)
            Zbf = torch.cat([Zf, ones], dim=1)
            W = self.W_.to(Z.device, dtype=Z.dtype)
            Xf_hat = Zbf @ W
            Xf = X_t.permute(0, 2, 3, 1).reshape(-1, X_t.shape[1])
            ss_res = ((Xf - Xf_hat) ** 2).sum()
            ss_tot = ((Xf - Xf.mean(dim=0, keepdim=True)) ** 2).sum().clamp_min(1e-8)
            return 1.0 - float(ss_res.cpu() / ss_tot.cpu())
