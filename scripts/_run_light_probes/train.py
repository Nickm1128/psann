# ruff: noqa: F403,F405
from __future__ import annotations

from .metrics import _grad_norm
from .shared import *


@dataclass
class TrainSpec:
    model: str
    hidden: int
    depth: int
    epochs: int
    kernel_size: int = 5
    lr: float = 1e-3
    batch_size: int = 256


def train_regressor(
    model: nn.Module, train_X, train_y, val_X, val_y, spec: TrainSpec, device: torch.device
):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=spec.lr)
    loss_fn = nn.MSELoss()
    tX = torch.from_numpy(train_X).float().to(device)
    ty = torch.from_numpy(train_y).float().to(device)
    vX = torch.from_numpy(val_X).float().to(device)
    vy = torch.from_numpy(val_y).float().to(device)
    total_steps = 0
    history: List[dict] = []
    total_start = time.perf_counter()
    for epoch in range(spec.epochs):
        model.train()
        epoch_start = time.perf_counter()
        perm = torch.randperm(tX.size(0), device=device)
        epoch_loss_sum = 0.0
        epoch_samples = 0
        step_times: List[float] = []
        grad_norms: List[float] = []
        loss_nonfinite = 0
        grad_nonfinite = 0
        for i in range(0, len(perm), spec.batch_size):
            idx = perm[i : i + spec.batch_size]
            xb, yb = tX.index_select(0, idx), ty.index_select(0, idx)
            opt.zero_grad()
            if device.type == "cuda":
                torch.cuda.synchronize()
            step_start = time.perf_counter()
            loss = loss_fn(model(xb), yb)
            loss_value = float(loss.detach().item())
            if not math.isfinite(loss_value):
                loss_nonfinite += 1
            loss.backward()
            grad_norm = _grad_norm(model)
            if grad_norm is not None:
                grad_norms.append(grad_norm)
                if not math.isfinite(grad_norm):
                    grad_nonfinite += 1
            opt.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            step_times.append(time.perf_counter() - step_start)
            bs = xb.shape[0]
            epoch_loss_sum += loss_value * bs
            epoch_samples += bs
            total_steps += 1
        train_loss = epoch_loss_sum / max(epoch_samples, 1)
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(vX), vy).item())
        epoch_time = time.perf_counter() - epoch_start
        finite_grad_norms = [g for g in grad_norms if math.isfinite(g)]
        grad_norm_mean = (
            float(sum(finite_grad_norms) / len(finite_grad_norms)) if finite_grad_norms else None
        )
        grad_norm_max = float(max(finite_grad_norms)) if finite_grad_norms else None
        step_time_mean = float(sum(step_times) / len(step_times)) if step_times else None
        history.append(
            {
                "epoch": int(epoch + 1),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "epoch_time_s": float(epoch_time),
                "step_time_s_mean": step_time_mean,
                "steps": int(len(step_times)),
                "samples": int(epoch_samples),
                "grad_norm_mean": grad_norm_mean,
                "grad_norm_max": grad_norm_max,
                "loss_nonfinite_steps": int(loss_nonfinite),
                "grad_nonfinite_steps": int(grad_nonfinite),
            }
        )
    model.eval()
    with torch.no_grad():
        vloss = loss_fn(model(vX), vy).item()
    total_time = time.perf_counter() - total_start
    if history:
        history[-1]["train_time_s_total"] = float(total_time)
    return model, {
        "epochs": spec.epochs,
        "val_loss": float(vloss),
        "steps": total_steps,
        "train_size": int(len(train_X)),
        "history": history,
    }
