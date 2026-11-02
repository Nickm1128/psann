"""Bounded activation and gradient measurements for PSANN LM experiments."""

from __future__ import annotations

import math
import time

import torch
from torch.nn import functional as F

from psann.activations import SineParam


def causality_probe(model, input_ids, max_tokens=32, atol=3e-5, rtol=3e-4):
    """Reject future-token dependence and context-length-dependent prefix logits."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    ids = input_ids[:1, :max_tokens].to(device)
    if ids.shape[1] < 2:
        model.train(was_training)
        raise ValueError("Causality checks need at least two context tokens.")
    split = ids.shape[1] // 2
    try:
        with torch.inference_mode():
            full = model(ids)[:, :split].float()
            prefix = model(ids[:, :split]).float()
            changed = ids.clone()
            changed[:, split:] = (changed[:, split:] + 1) % model.lm_config.vocab_size
            perturbed = model(changed)[:, :split].float()
        return {
            "passed": bool(
                torch.allclose(full, prefix, atol=atol, rtol=rtol)
                and torch.allclose(full, perturbed, atol=atol, rtol=rtol)
            ),
            "tokens": ids.shape[1],
            "prefix_tokens": split,
            "prefix_max_abs_delta": float((full - prefix).abs().max()),
            "future_max_abs_delta": float((full - perturbed).abs().max()),
            "atol": atol,
            "rtol": rtol,
        }
    finally:
        model.train(was_training)


def distribution(value):
    value = value.detach().float().reshape(-1)
    finite = value[torch.isfinite(value)]
    result = {"finite_fraction": float(torch.isfinite(value).float().mean())}
    if finite.numel():
        result.update(
            mean=float(finite.mean()),
            rms=float(finite.square().mean().sqrt()),
            min=float(finite.min()),
            max=float(finite.max()),
        )
    return result


def gradient_groups(model):
    groups = {}
    for name, parameter in model.named_parameters():
        key = (
            "activation"
            if any(piece in name for piece in ("._A", "._f", "._d", "._phi"))
            else (
                "embedding"
                if name.startswith("embed.")
                else "readout" if name.startswith("lm_head.") else "backbone"
            )
        )
        group = groups.setdefault(
            key,
            {"parameters": 0, "weight_squared": 0.0, "gradient_squared": 0.0, "with_gradient": 0},
        )
        group["parameters"] += parameter.numel()
        group["weight_squared"] += float(parameter.detach().float().square().sum())
        if parameter.grad is not None:
            group["with_gradient"] += parameter.numel()
            group["gradient_squared"] += float(parameter.grad.detach().float().square().sum())
    for group in groups.values():
        group["weight_norm"] = math.sqrt(group.pop("weight_squared"))
        group["gradient_norm"] = math.sqrt(group.pop("gradient_squared"))
        group["gradient_to_weight_ratio"] = group["gradient_norm"] / max(
            group["weight_norm"], 1e-12
        )
    return groups


class ActivationProbe:
    def __init__(self, model, max_elements=4096):
        self.activations, self.handles, self.representation = {}, [], {}
        self.max_elements = max_elements
        for name, module in model.named_modules():
            if isinstance(module, SineParam):
                self.handles.append(module.register_forward_hook(self._hook(name)))
        if hasattr(model, "ln_f"):
            self.handles.append(model.ln_f.register_forward_hook(self._features))

    def _hook(self, name):
        def hook(module, args, output):
            with torch.no_grad():
                x = args[0].detach().movedim(module.feature_dim, -1)
                width = x.shape[-1]
                sample = x.reshape(-1, width)[: max(1, self.max_elements // width)].float()
                params = F.softplus(torch.stack((module._A, module._f, module._d)).detach().float())
                amplitude = module._apply_bounds(params[0], "amplitude")
                frequency = module._apply_bounds(params[1] + 1e-6, "frequency")
                decay = module._apply_bounds(params[2], "decay")
                phase = sample * frequency
                if hasattr(module, "_phi"):
                    phase = phase + module._phi.detach().float()
                if module.decay_mode == "abs":
                    g, dg = sample.abs(), sample.sign()
                elif module.decay_mode == "relu":
                    g, dg = sample.relu(), (sample > 0).float()
                else:
                    g, dg = torch.zeros_like(sample), torch.zeros_like(sample)
                envelope = torch.exp(-decay * g)
                derivative = (
                    amplitude * envelope * (frequency * phase.cos() - decay * dg * phase.sin())
                )
                self.activations[name] = {
                    "sample_elements": sample.numel(),
                    "input": distribution(sample),
                    "output": distribution(output.detach().reshape(-1)[: self.max_elements]),
                    "amplitude": distribution(amplitude),
                    "frequency": distribution(frequency),
                    "decay": distribution(decay),
                    "phase": distribution(phase),
                    "derivative": distribution(derivative),
                    "damped_fraction": float((envelope < 0.05).float().mean()),
                    "tiny_derivative_fraction": float((derivative.abs() < 0.001).float().mean()),
                }

        return hook

    def _features(self, module, args, output):
        with torch.no_grad():
            features = output.detach().reshape(-1, output.shape[-1])[:64, :256].float()
            centered = features - features.mean(dim=0)
            covariance = centered.T @ centered / max(1, features.shape[0] - 1)
            denominator = float(covariance.square().sum())
            self.representation = {
                "sample_rows": features.shape[0],
                "sample_dimensions": features.shape[1],
                "participation_ratio": float(covariance.trace().square()) / max(denominator, 1e-12),
            }

    def close(self):
        for handle in self.handles:
            handle.remove()


def probe(model, batch):
    """One deterministic forward/backward pass; use a fresh experiment model."""
    was_training = model.training
    model.eval()
    observer = ActivationProbe(model)
    device = next(model.parameters()).device
    ids, labels = batch["input_ids"].to(device), batch["labels"].to(device)
    model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    try:
        logits = model(ids)
        loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.monotonic() - started
        groups = gradient_groups(model)
        return {
            "loss": float(loss.detach()),
            "finite_logits": bool(torch.isfinite(logits).all()),
            "forward_backward_s": elapsed,
            "tokens": ids.numel(),
            "forward_backward_tokens_per_s": ids.numel() / max(elapsed, 1e-12),
            "peak_cuda_allocated_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            ),
            "gradients": groups,
            "finite_gradients": all(
                math.isfinite(group["gradient_norm"]) for group in groups.values()
            ),
            "activations": observer.activations,
            "representation": observer.representation,
        }
    finally:
        observer.close()
        model.zero_grad(set_to_none=True)
        model.train(was_training)
