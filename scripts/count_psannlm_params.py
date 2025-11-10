#!/usr/bin/env python
"""Count PSANN-LM model parameters for a given configuration."""

from __future__ import annotations

import argparse

import torch

from psann.lm.models.registry import get_base
from psann.lm.models.sine import SineConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Count parameters for a PSANN-LM configuration")
    p.add_argument("--base", type=str, default="waveresnet", choices=["waveresnet", "respsann"])
    p.add_argument("--vocab-size", type=int, default=50257)
    p.add_argument("--d-model", type=int, default=2048)
    p.add_argument("--n-layers", type=int, default=22)
    p.add_argument("--n-heads", type=int, default=16)
    p.add_argument("--d-mlp", type=int, default=None, help="Defaults to 4*d_model if not provided")
    p.add_argument("--pos-enc", type=str, default="rope", choices=["rope", "alibi", "sinusoidal"])
    p.add_argument("--wave-interleave", action="store_true")
    p.add_argument("--wave-kernel-size", type=int, default=3)
    p.add_argument("--wave-dilation-growth", type=int, default=1)
    p.add_argument("--wave-dropout", type=float, default=0.0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    d_mlp = args.d_mlp if args.d_mlp is not None else 4 * int(args.d_model)
    factory = get_base(args.base)
    sine = SineConfig()
    model = factory(
        vocab_size=int(args.vocab_size),
        d_model=int(args.d_model),
        n_layers=int(args.n_layers),
        n_heads=int(args.n_heads),
        d_mlp=int(d_mlp),
        positional_encoding=str(args.pos_enc),
        sine=sine,
        wave_interleave=bool(args.wave_interleave),
        wave_kernel_size=int(args.wave_kernel_size),
        wave_dilation_growth=int(args.wave_dilation_growth),
        wave_dropout=float(args.wave_dropout),
    )
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
