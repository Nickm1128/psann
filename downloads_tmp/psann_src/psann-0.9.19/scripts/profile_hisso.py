#!/usr/bin/env python
"""Quick HISSO profiling harness for local benchmarking."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from psann.augmented import PredictiveExtrasConfig, PredictiveExtrasTrainer


def _make_series(n_steps: int, features: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(size=(n_steps, features)).astype(np.float32)


def _build_trainer(device: str | torch.device) -> PredictiveExtrasTrainer:
    cfg = PredictiveExtrasConfig(episode_length=64, batch_episodes=8, primary_dim=8, extras_dim=4)
    model = torch.nn.Sequential(
        torch.nn.Linear(cfg.primary_dim + cfg.extras_dim, 64),
        torch.nn.GELU(),
        torch.nn.Linear(64, cfg.primary_dim + cfg.extras_dim),
    )
    return PredictiveExtrasTrainer(
        model,
        reward_fn=lambda alloc, ctx: alloc.mean(dim=-1),
        cfg=cfg,
        device=device,
    )


def profile(device: str | torch.device, epochs: int) -> dict:
    trainer = _build_trainer(device)
    series = _make_series(2048, 8)
    t0 = time.perf_counter()
    trainer.train(series, epochs=epochs, verbose=0)
    total = time.perf_counter() - t0
    summary = trainer.profile_summary()
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
