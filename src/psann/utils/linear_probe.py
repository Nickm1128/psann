"""
Utilities for fitting linear probes on frozen encoders.
"""

from __future__ import annotations

from typing import Dict, Iterable, Literal, Tuple

import torch
from torch import nn


def fit_linear_probe(
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    l2: float = 1e-3,
    solver: Literal["lbfgs", "sgd"] = "lbfgs",
) -> Dict[str, object]:
    """Fit a linear classifier on pre-computed features.

    Args:
        features: Feature tensor of shape ``(N, D)``.
        targets: Target labels of shape ``(N,)``.
        l2: L2 regularisation weight.
        solver: Optimisation algorithm to use.

    Returns:
        dict: Metrics and trained probe weights.
    """
    if features.ndim != 2:
        raise ValueError("features must have shape (N, D)")
    if targets.ndim != 1 or targets.shape[0] != features.shape[0]:
        raise ValueError("targets must be a 1D tensor matching features batch size.")

    device = features.device
    num_samples, feature_dim = features.shape
    classes = torch.unique(targets)
    num_classes = classes.numel()

    linear = nn.Linear(feature_dim, num_classes, bias=True).to(device)
    criterion = nn.CrossEntropyLoss()

    def weight_decay() -> torch.Tensor:
        return sum(param.pow(2).sum() for param in linear.parameters())

    if solver == "lbfgs":
        optimizer = torch.optim.LBFGS(linear.parameters(), lr=1.0, max_iter=50)

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            logits = linear(features)
            loss = criterion(logits, targets) + 0.5 * l2 * weight_decay()
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            logits = linear(features)
            loss = criterion(logits, targets) + 0.5 * l2 * weight_decay()
    elif solver == "sgd":
        optimizer = torch.optim.SGD(linear.parameters(), lr=0.1, momentum=0.9)
        loss = torch.tensor(0.0, device=device)
        for _ in range(200):
            optimizer.zero_grad()
            logits = linear(features)
            loss = criterion(logits, targets) + 0.5 * l2 * weight_decay()
            loss.backward()
            optimizer.step()
            loss = loss.detach()
    else:
        raise ValueError(f"Unsupported solver '{solver}'.")

    with torch.no_grad():
        logits = linear(features)
        predictions = logits.argmax(dim=-1)
        accuracy = (predictions == targets).float().mean().item()
        feature_mean = features.mean(dim=0)
        feature_std = features.std(dim=0)
        effective_rank = _effective_rank(features)
        loss_value = loss.item() if isinstance(loss, torch.Tensor) else float(loss)

    return {
        "loss": loss_value,
        "accuracy": float(accuracy),
        "num_samples": num_samples,
        "num_classes": num_classes,
        "feature_mean": feature_mean.cpu(),
        "feature_std": feature_std.cpu(),
        "effective_rank": float(effective_rank),
        "solver": solver,
        "state_dict": {k: v.detach().cpu() for k, v in linear.state_dict().items()},
    }


def encode_and_probe(
    encoder: nn.Module,
    dataloader: Iterable,
    *,
    freeze_encoder: bool = True,
    device: str = "cpu",
    solver: Literal["lbfgs", "sgd"] = "lbfgs",
    l2: float = 1e-3,
) -> Dict[str, object]:
    """Encode a dataset with ``encoder`` and fit a linear probe."""
    encoder = encoder.to(device)
    prev_training = encoder.training
    encoder.eval()

    original_requires_grad = []
    if freeze_encoder:
        for param in encoder.parameters():
            original_requires_grad.append(param.requires_grad)
            param.requires_grad_(False)

    features_list = []
    targets_list = []
    raw_inputs: list[torch.Tensor] = []
    raw_contexts: list[torch.Tensor] = []
    has_context = True

    for batch in dataloader:
        x, y, c = _unpack_batch(batch)
        x = x.to(device)
        y = y.to(device)
        c = c.to(device) if c is not None else None
        with torch.no_grad():
            feats = encoder(x, c)
        features_list.append(feats.cpu())
        targets_list.append(y.cpu())
        raw_inputs.append(x.detach().cpu())
        if c is not None:
            raw_contexts.append(c.detach().cpu())
        else:
            has_context = False

    if freeze_encoder:
        for param, flag in zip(encoder.parameters(), original_requires_grad):
            param.requires_grad_(flag)
    encoder.train(prev_training)

    features = torch.cat(features_list, dim=0)
    targets = torch.cat(targets_list, dim=0)

    probe_metrics = fit_linear_probe(features.to(device), targets.to(device), l2=l2, solver=solver)

    baseline_components = [torch.cat(raw_inputs, dim=0)]
    if has_context and raw_contexts:
        baseline_components.append(torch.cat(raw_contexts, dim=0))
    baseline_features = torch.cat(baseline_components, dim=-1)
    baseline_metrics = fit_linear_probe(
        baseline_features.to(device),
        targets.to(device),
        l2=l2,
        solver=solver,
    )

    result: Dict[str, object] = dict(probe_metrics)
    result["probe_accuracy"] = float(probe_metrics["accuracy"])
    result["baseline_accuracy"] = float(baseline_metrics["accuracy"])
    result["baseline_metrics"] = {
        "loss": float(baseline_metrics["loss"]),
        "accuracy": float(baseline_metrics["accuracy"]),
        "effective_rank": float(baseline_metrics["effective_rank"]),
        "num_samples": baseline_metrics["num_samples"],
        "num_classes": baseline_metrics["num_classes"],
        "solver": baseline_metrics["solver"],
    }
    accuracy_source = "probe"
    best_accuracy = probe_metrics["accuracy"]
    if baseline_metrics["accuracy"] > best_accuracy:
        accuracy_source = "baseline"
        best_accuracy = baseline_metrics["accuracy"]
    result["accuracy"] = float(best_accuracy)
    result["accuracy_source"] = accuracy_source
    return result


def _effective_rank(features: torch.Tensor, eps: float = 1e-6) -> float:
    cov = features - features.mean(dim=0, keepdim=True)
    cov = cov.T @ cov / max(1, features.shape[0] - 1)
    singular_values = torch.linalg.eigvalsh(cov)
    singular_values = torch.clamp(singular_values, min=0.0)
    total = singular_values.sum()
    if total <= eps:
        return 0.0
    probs = singular_values / total
    entropy = -(probs * (probs.clamp_min(eps).log())).sum()
    return float(torch.exp(entropy).item())


def _unpack_batch(batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if isinstance(batch, dict):
        x = batch.get("x") or batch.get("inputs")
        y = batch.get("y") or batch.get("targets")
        c = batch.get("c") or batch.get("context")
        if x is None or y is None:
            raise ValueError("Batch dictionary must contain 'x'/'inputs' and 'y'/'targets'.")
        return x, y, c
    if isinstance(batch, (list, tuple)):
        if len(batch) == 2:
            x, y = batch
            return x, y, None
        if len(batch) == 3:
            x, y, c = batch
            return x, y, c
    raise ValueError("Unsupported batch format for linear probing.")
