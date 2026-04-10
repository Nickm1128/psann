from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from ..training import TrainingLoopConfig, run_training_loop
from .amp import _guard_cuda_capture
from .config import HISSOWarmStartConfig

if TYPE_CHECKING:
    from ..sklearn import PSANNRegressor


def run_hisso_supervised_warmstart(
    estimator: "PSANNRegressor",
    X_flat: np.ndarray,
    *,
    primary_dim: int,
    config: Optional[HISSOWarmStartConfig],
    lsm_module: Optional[torch.nn.Module],
) -> None:
    """Run a supervised warm start against primary targets before HISSO."""

    if config is None:
        return

    y_vec = np.asarray(config.targets, dtype=np.float32)
    if y_vec.ndim == 1:
        y_vec = y_vec.reshape(-1, 1)
    if y_vec.ndim != 2:
        raise ValueError("hisso_supervised['y'] must be 2D with shape (N, primary_dim).")
    if y_vec.shape[0] != X_flat.shape[0]:
        raise ValueError("hisso_supervised['y'] length must match X.")
    if y_vec.shape[1] != int(primary_dim):
        raise ValueError("hisso_supervised['y'] column count must equal primary_dim.")

    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_flat.astype(np.float32)),
        torch.from_numpy(y_vec.astype(np.float32)),
    )

    shuffle = not (estimator.stateful and estimator.state_reset in ("epoch", "none"))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(config.batch_size or estimator.batch_size),
        shuffle=shuffle,
        num_workers=int(estimator.num_workers),
    )

    device = estimator._device()
    estimator._ensure_model_device(device)

    optimizer = estimator._build_optimizer(estimator.model_)
    if config.lr is not None:
        for group in optimizer.param_groups:
            group["lr"] = float(config.lr)
    if config.weight_decay is not None:
        for group in optimizer.param_groups:
            group["weight_decay"] = float(config.weight_decay)
    if lsm_module is not None and config.lsm_lr is not None:
        for group in optimizer.param_groups:
            if any(param in group["params"] for param in lsm_module.parameters()):
                group["lr"] = float(config.lsm_lr)

    loop_cfg = TrainingLoopConfig(
        epochs=int(config.epochs or estimator.epochs),
        patience=1,
        early_stopping=False,
        stateful=bool(estimator.stateful),
        state_reset=str(estimator.state_reset),
        verbose=int(config.verbose),
        lr_max=None,
        lr_min=None,
    )

    with _guard_cuda_capture():
        run_training_loop(
            estimator.model_,
            optimizer=optimizer,
            loss_fn=estimator._make_loss(),
            train_loader=dataloader,
            device=device,
            cfg=loop_cfg,
        )
    estimator.model_.eval()
