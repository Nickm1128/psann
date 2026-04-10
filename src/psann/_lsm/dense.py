from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .._aliases import resolve_int_alias
from ..utils import choose_device, seed_all
from .common import TensorLike, _tensor_to_output, _to_float_tensor


class MaskedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        sparsity: float = 0.8,
        random_state: Optional[int] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        rs = torch.Generator()
        if random_state is not None:
            rs.manual_seed(int(random_state))
        density = max(0.0, min(1.0, 1.0 - float(sparsity)))
        mask = (torch.rand((out_features, in_features), generator=rs) < density).float()
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.linear.weight * self.mask
        return F.linear(x, w, self.linear.bias)


class LSM(nn.Module):
    """Liquid State Machine-inspired expander (feed-forward, sparse layers)."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_layers: int = 2,
        hidden_units: Optional[int] = None,
        hidden_width: Optional[int] = 128,
        sparsity: float = 0.8,
        nonlinearity: str = "sine",
        bias: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        width_res = resolve_int_alias(
            primary_value=hidden_units,
            alias_value=hidden_width,
            primary_name="hidden_units",
            alias_name="hidden_width",
            context="LSM",
            default=128,
        )
        width = width_res.value if width_res.value is not None else 128
        self.hidden_layers = int(hidden_layers)
        self.hidden_units = width
        self.hidden_width = width
        self.sparsity = float(sparsity)
        self.nonlinearity = nonlinearity

        act = {"sine": torch.sin, "tanh": torch.tanh, "relu": F.relu}.get(nonlinearity)
        if act is None:
            raise ValueError("nonlinearity must be one of: sine, tanh, relu")
        self._act = act

        layers = []
        in_dim = self.input_dim
        for i in range(self.hidden_layers):
            layers.append(
                MaskedLinear(
                    in_dim,
                    self.hidden_width,
                    bias=bias,
                    sparsity=self.sparsity,
                    random_state=None if random_state is None else random_state + i,
                )
            )
            in_dim = self.hidden_width
        self.body = nn.Sequential(*layers)
        self.head = MaskedLinear(
            in_dim,
            self.output_dim,
            bias=bias,
            sparsity=self.sparsity,
            random_state=None if random_state is None else random_state + 999,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x
        for layer in self.body:
            z = self._act(layer(z))
        return self.head(z)


class LSMExpander(nn.Module):
    """Pretraining interface for :class:`LSM` with an OLS-in-the-loop objective."""

    def __init__(
        self,
        output_dim: int,
        *,
        hidden_layers: int = 2,
        hidden_units: Optional[int] = None,
        hidden_width: Optional[int] = 128,
        sparsity: float = 0.8,
        nonlinearity: str = "sine",
        epochs: int = 100,
        lr: float = 1e-3,
        ridge: float = 1e-4,
        batch_size: int = 256,
        device: str | torch.device = "auto",
        random_state: Optional[int] = None,
        early_stopping: bool = False,
        patience: int = 20,
        tol: float = 1e-6,
        val_split: Optional[float] = None,
        verbose: int = 0,
        objective: str = "r2",
        alpha_ortho: float = 0.0,
        alpha_sparse: float = 0.0,
        alpha_var: float = 0.0,
        target_var: float = 1.0,
        noisy: Optional[float] = None,
        noise_decay: float = 1.0,
    ) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        width_res = resolve_int_alias(
            primary_value=hidden_units,
            alias_value=hidden_width,
            primary_name="hidden_units",
            alias_name="hidden_width",
            context="LSMExpander",
            default=128,
        )
        width = width_res.value if width_res.value is not None else 128
        self.hidden_layers = int(hidden_layers)
        self.hidden_units = width
        self.hidden_width = width
        self.sparsity = sparsity
        self.nonlinearity = nonlinearity
        self.epochs = epochs
        self.lr = lr
        self.ridge = ridge
        self.batch_size = batch_size
        self.device = device
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.patience = patience
        self.tol = tol
        self.val_split = val_split
        self.verbose = verbose
        self.objective = objective
        self.alpha_ortho = float(alpha_ortho)
        self.alpha_sparse = float(alpha_sparse)
        self.alpha_var = float(alpha_var)
        self.target_var = float(target_var)
        self.noisy = noisy
        self.noise_decay = float(noise_decay)
        self.model: Optional[LSM] = None
        self.W_: Optional[torch.Tensor] = None

    def _device(self) -> torch.device:
        return choose_device(self.device)

    def _ols_readout(self, Z: torch.Tensor, X: torch.Tensor, ridge: float) -> torch.Tensor:
        ones = torch.ones((Z.shape[0], 1), dtype=Z.dtype, device=Z.device)
        Zb = torch.cat([Z, ones], dim=1)
        A = Zb.T @ Zb
        b = Zb.T @ X
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
                W = torch.linalg.lstsq(Zb, X, rcond=None).solution
            except Exception:
                W = torch.linalg.pinv(Zb) @ X
        return W

    def fit(
        self,
        X: TensorLike,
        epochs: Optional[int] = None,
        *,
        validation_data: Optional[TensorLike] = None,
        val_split: Optional[float] = None,
        early_stopping: Optional[bool] = None,
        patience: Optional[int] = None,
        tol: Optional[float] = None,
        verbose: Optional[int] = None,
    ) -> "LSMExpander":
        seed_all(self.random_state)
        device = self._device()
        X_all_t, _, _, _ = _to_float_tensor(X, device=device)
        if X_all_t.ndim != 2:
            raise ValueError("Expected 2D input shaped (N, D)")
        n_features = int(X_all_t.shape[1])
        self.model = LSM(
            n_features,
            self.output_dim,
            hidden_layers=self.hidden_layers,
            hidden_width=self.hidden_width,
            sparsity=self.sparsity,
            nonlinearity=self.nonlinearity,
            random_state=self.random_state,
        ).to(device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        es = self.early_stopping if early_stopping is None else bool(early_stopping)
        pat = int(self.patience if patience is None else patience)
        tolerance = float(self.tol if tol is None else tol)
        verb = int(self.verbose if verbose is None else verbose)
        vs = self.val_split if val_split is None else val_split
        E = int(epochs) if epochs is not None else (10000 if es else self.epochs)

        if validation_data is not None:
            X_tr_t = X_all_t
            X_va_t, _, _, _ = _to_float_tensor(validation_data, device=device)
            if X_va_t.ndim != 2 or int(X_va_t.shape[1]) != n_features:
                raise ValueError("validation_data shape mismatch; expected (N, D)")
        elif vs is not None or es:
            frac = 0.1 if vs is None else float(vs)
            frac = min(max(frac, 0.0), 0.9)
            n = X_all_t.shape[0]
            if n <= 1:
                X_tr_t = X_all_t
                X_va_t = None
            else:
                n_val = max(1, int(n * frac))
                n_val = min(n_val, n - 1)
                perm = torch.randperm(int(n), device=device)
                va_idx = perm[:n_val]
                tr_idx = perm[n_val:]
                X_va_t = X_all_t.index_select(0, va_idx)
                X_tr_t = X_all_t.index_select(0, tr_idx)
        else:
            X_tr_t = X_all_t
            X_va_t = None

        noise_std_t = None
        if self.noisy is not None and float(self.noise_decay) >= 0.0:
            if np.isscalar(self.noisy):
                std = np.full((1, n_features), float(self.noisy), dtype=np.float32)
            else:
                arr = np.asarray(self.noisy, dtype=np.float32).reshape(1, -1)
                if arr.shape[1] != n_features:
                    raise ValueError(f"noisy has {arr.shape[1]} features, expected {n_features}")
                std = arr
            noise_std_t = torch.from_numpy(std).to(device)

        best_r2 = -float("inf")
        best_state = None
        patience_left = pat

        n_train = int(X_tr_t.shape[0])
        batch_size = int(self.batch_size) if self.batch_size is not None else n_train
        if batch_size <= 0:
            batch_size = n_train
        if n_train > 0:
            batch_size = max(1, min(batch_size, n_train))
        else:
            batch_size = 0

        stop_training = False
        for epoch in range(E):
            self.model.train()
            if n_train == 0:
                break
            perm = torch.randperm(n_train, device=device)
            factor = float(max(self.noise_decay, 0.0) ** epoch)
            for start_idx in range(0, n_train, batch_size):
                idx = perm[start_idx : start_idx + batch_size]
                X_batch = X_tr_t.index_select(0, idx)
                opt.zero_grad()
                X_in = (
                    X_batch + torch.randn_like(X_batch) * (noise_std_t * factor)
                    if noise_std_t is not None
                    else X_batch
                )
                Z_batch = self.model(X_in)
                W = self._ols_readout(Z_batch, X_batch, ridge=self.ridge)
                ones_batch = torch.ones(
                    (Z_batch.shape[0], 1), dtype=Z_batch.dtype, device=Z_batch.device
                )
                Zb_batch = torch.cat([Z_batch, ones_batch], dim=1)
                X_hat_batch = Zb_batch @ W
                if self.objective == "mse":
                    base = F.mse_loss(X_hat_batch, X_batch)
                else:
                    ss_res = ((X_batch - X_hat_batch) ** 2).sum()
                    ss_tot = (
                        ((X_batch - X_batch.mean(dim=0, keepdim=True)) ** 2).sum().clamp_min(1e-8)
                    )
                    base = ss_res / ss_tot
                reg = 0.0
                if self.alpha_ortho > 0.0:
                    Zc = Z_batch - Z_batch.mean(dim=0, keepdim=True)
                    Cz = (Zc.T @ Zc) / max(1, Z_batch.shape[0] - 1)
                    offdiag = Cz - torch.diag(torch.diag(Cz))
                    reg = reg + self.alpha_ortho * (offdiag.pow(2).sum() / (Z_batch.shape[1] ** 2))
                if self.alpha_sparse > 0.0:
                    reg = reg + self.alpha_sparse * Z_batch.abs().mean()
                if self.alpha_var > 0.0:
                    var = Z_batch.var(dim=0, unbiased=False)
                    reg = reg + self.alpha_var * ((var - self.target_var) ** 2).mean()
                (base + reg).backward()
                opt.step()

            if X_va_t is not None and n_train > 0:
                self.model.eval()
                with torch.no_grad():
                    Z_tr_eval = self.model(X_tr_t)
                    W_eval = self._ols_readout(Z_tr_eval, X_tr_t, ridge=self.ridge)
                    Z_va = self.model(X_va_t)
                    ones_va = torch.ones((Z_va.shape[0], 1), dtype=Z_va.dtype, device=Z_va.device)
                    Zb_va = torch.cat([Z_va, ones_va], dim=1)
                    X_hat_va = Zb_va @ W_eval
                    ss_res_va = ((X_va_t - X_hat_va) ** 2).sum()
                    ss_tot_va = (
                        ((X_va_t - X_va_t.mean(dim=0, keepdim=True)) ** 2).sum().clamp_min(1e-8)
                    )
                    r2_va = float(1.0 - (ss_res_va / ss_tot_va))
                if verb:
                    print(f"LSMExpander epoch {epoch+1}/{E} - val R^2: {r2_va:.6f}")
                if r2_va > best_r2 + tolerance:
                    best_r2 = r2_va
                    best_state = {
                        k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                    }
                    patience_left = pat
                else:
                    patience_left -= 1
                    if es and patience_left <= 0:
                        if verb:
                            print(
                                f"Early stopping LSM at epoch {epoch+1} (best R^2: {best_r2:.6f})"
                            )
                        stop_training = True
                if stop_training:
                    break

            if stop_training:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        with torch.no_grad():
            Z_all = self.model(X_all_t)
            self.W_ = self._ols_readout(Z_all, X_all_t, ridge=self.ridge).detach().cpu()
        return self

    def transform(self, X: TensorLike) -> TensorLike:
        if self.model is None:
            raise RuntimeError("LSMExpander not fitted; call fit() first or set .model externally.")
        model_device = next(self.model.parameters()).device
        X_t, is_tensor, orig_device, orig_dtype = _to_float_tensor(X, device=model_device)
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
            raise RuntimeError("LSMExpander not fitted; call fit() before forward().")
        device = next(self.model.parameters()).device
        return self.model(X.to(device=device, dtype=torch.float32))

    def to(self, *args, **kwargs) -> "LSMExpander":
        super().to(*args, **kwargs)
        if self.model is not None:
            self.model = self.model.to(*args, **kwargs)
        if self.W_ is not None:
            self.W_ = self.W_.to(*args, **kwargs)
        return self

    def train(self, mode: bool = True) -> "LSMExpander":
        super().train(mode)
        if self.model is not None:
            self.model.train(mode)
        return self

    def eval(self) -> "LSMExpander":
        return self.train(False)

    def score_reconstruction(self, X: TensorLike) -> float:
        if self.model is None or self.W_ is None:
            raise RuntimeError("LSMExpander not fitted")
        model_device = next(self.model.parameters()).device
        X_t, _, _, _ = _to_float_tensor(X, device=model_device)
        with torch.no_grad():
            Z = self.model(X_t)
            ones = torch.ones((Z.shape[0], 1), dtype=Z.dtype, device=Z.device)
            Zb = torch.cat([Z, ones], dim=1)
            W = self.W_.to(Z.device, dtype=Z.dtype)
            X_hat = Zb @ W
            ss_res = ((X_t - X_hat) ** 2).sum()
            ss_tot = ((X_t - X_t.mean(dim=0, keepdim=True)) ** 2).sum().clamp_min(1e-8)
            return 1.0 - float(ss_res.cpu() / ss_tot.cpu())
