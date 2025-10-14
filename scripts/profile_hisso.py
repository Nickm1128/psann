#!/usr/bin/env python
"""Quick HISSO profiling harness for the primary-only trainer."""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from psann.hisso import HISSOTrainer, HISSOTrainerConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_series(n_steps: int, features: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(size=(n_steps, features)).astype(np.float32)


def _build_trainer(device: str | torch.device) -> HISSOTrainer:
    device_obj = torch.device(device)
    cfg = HISSOTrainerConfig(
        episode_length=64,
        episodes_per_batch=8,
        primary_dim=8,
        primary_transform="softmax",
        transition_penalty=0.0,
    )
    model = torch.nn.Sequential(
        torch.nn.Linear(cfg.primary_dim, 64),
        torch.nn.GELU(),
        torch.nn.Linear(64, cfg.primary_dim),
    )
    return HISSOTrainer(
        model,
        cfg=cfg,
        device=device_obj,
        lr=1e-3,
        reward_fn=lambda alloc, ctx: alloc.mean(dim=-1),
        context_extractor=None,
        input_noise_std=None,
    )


def profile(device: str | torch.device, epochs: int) -> dict:
    trainer = _build_trainer(device)
    series = _make_series(2048, trainer.cfg.primary_dim)
    t0 = time.perf_counter()
    trainer.train(series, epochs=epochs, verbose=0, lr_max=None, lr_min=None)
    total = time.perf_counter() - t0
    summary = dict(trainer.profile)
    summary["wall_time_s"] = total
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=4, help="Number of epochs to profile")
    args = parser.parse_args()

    cpu_summary = profile("cpu", epochs=args.epochs)
    print("CPU profile:", cpu_summary)
    if torch.cuda.is_available():
        cuda_summary = profile("cuda", epochs=args.epochs)
        print("CUDA profile:", cuda_summary)
    else:
        print("CUDA not available; skipping GPU profile")


if __name__ == "__main__":
    main()
