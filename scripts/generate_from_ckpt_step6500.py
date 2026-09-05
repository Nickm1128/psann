#!/usr/bin/env python3
"""Compatibility diagnostic; see docs/migration.md for canonical workflows.


Quick text-generation driver for the ckpt_step006500 checkpoint.

Usage:
  python scripts/generate_from_ckpt_step6500.py \\
    --ckpt runs/ckpt_step006500.pt \\
    --tokenizer-dir runs/tokenizer_300m \\
    --prompt "The future of PSANN-LM is"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from psannlm.lm import PSANNLM  # type: ignore  # noqa: E402
from psannlm.lm.data.tokenizer import Tokenizer, TokenizerConfig  # type: ignore  # noqa: E402


def _infer_dims(state_dict: dict) -> Tuple[int, int, int, int]:
    """Infer vocab_size, d_model, d_mlp, n_layers from the state dict."""
    vocab_size, d_model = state_dict["embed.weight"].shape
    d_mlp = state_dict["blocks.0.mlp.fc1.weight"].shape[0]
    layers = [int(k.split(".")[1]) for k in state_dict if k.startswith("blocks.")]
    n_layers = max(layers) + 1 if layers else 0
    return int(vocab_size), int(d_model), int(d_mlp), int(n_layers)


def _load_tokenizer(tokenizer_dir: Path) -> Tokenizer:
    cfg = TokenizerConfig(
        backend="tokenizers",
        model_path=str(tokenizer_dir / "tokenizer.json"),
        special_tokens_map_path=str(tokenizer_dir / "special_tokens_map.json"),
        hf_passthrough_ids=True,
    )
    tok = Tokenizer(cfg)
    tok.fit([])  # loads from the serialized tokenizer.json
    return tok


def _default_prompts() -> List[str]:
    return [
        "The quick brown fox jumps over",
        "Write a short summary of PSANN-LM:",
        "User: Give me a haiku about GPUs.\nAssistant:",
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate text from ckpt_step006500.")
    p.add_argument("--ckpt", type=str, default="runs/ckpt_step006500.pt")
    p.add_argument("--tokenizer-dir", type=str, default="runs/tokenizer_300m")
    p.add_argument(
        "--prompt",
        action="append",
        help="Prompt to generate from (can be passed multiple times).",
    )
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--n-heads", type=int, default=None, help="Override head count if needed.")
    p.add_argument("--attn-impl", type=str, default="sdpa", help="math|sdpa|auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.ckpt)
    from psannlm.cli import _load_model

    model, config, _ = _load_model(
        ckpt_path,
        base="waveresnet",
        pos_enc="rope",
        n_heads=args.n_heads,
        attn_impl=args.attn_impl,
        device=device,
    )
    vocab_size, d_model, d_mlp, n_layers, n_heads = (
        config.vocab_size,
        config.d_model,
        config.d_mlp,
        config.n_layers,
        config.n_heads,
    )
    tokenizer = _load_tokenizer(Path(args.tokenizer_dir))

    lm = PSANNLM(config=config, device=device)
    lm._model = model  # reuse loaded model with the desired attn_impl
    lm._tokenizer = tokenizer

    prompts: Iterable[str] = args.prompt if args.prompt else _default_prompts()
    print(
        f"[info] device={device} vocab_size={vocab_size} d_model={d_model} "
        f"n_layers={n_layers} n_heads={n_heads} d_mlp={d_mlp}"
    )
    for p in prompts:
        out = lm.generate(
            p,
            max_new_tokens=args.max_new_tokens,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
        )
        print(f"\n[prompt]\n{p}\n[output]\n{out}")


if __name__ == "__main__":
    main()
