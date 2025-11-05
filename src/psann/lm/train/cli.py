"""Command-line interface for PSANN-LM training.

Loads a YAML config and runs training with periodic checkpointing.
"""

from __future__ import annotations

import argparse
from typing import List
import os

import yaml

from ..api import psannLM, psannLMDataPrep


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="psann-lm-train", description="Train PSANN-LM")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    return p


def _load_texts(paths: List[str]) -> list[str]:
    texts: list[str] = []
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p, "r", encoding="utf-8") as fh:
            texts.extend([ln.rstrip("\n") for ln in fh.readlines() if ln.strip()])
    return texts


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI wiring
    parser = build_parser()
    args = parser.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})

    # Prepare data
    sources = []
    if isinstance(data_cfg.get("sources"), list):
        for ent in data_cfg["sources"]:
            if isinstance(ent, dict) and "path" in ent:
                sources.append(str(ent["path"]))
            elif isinstance(ent, str):
                sources.append(ent)
    texts = _load_texts(sources)
    if not texts:
        raise SystemExit(
            "No training texts were loaded. Ensure data.sources paths exist and contain text.\n"
            "For the tiny-corpus benchmark, create datasets/lm/tiny_books.txt (~50MB, one paragraph per line)."
        )
    dp = psannLMDataPrep(
        texts,
        tokenizer=str(data_cfg.get("tokenizer", "auto")),
        tokenizer_model_path=str(data_cfg.get("tokenizer_model_path")) if data_cfg.get("tokenizer_model_path") is not None else None,
        max_length=int(data_cfg.get("max_length", 1024)),
        pack_sequences=bool(data_cfg.get("pack_sequences", True)),
        val_split=float(data_cfg.get("val_split", 0.0)) if data_cfg.get("val_split") is not None else None,
        seed=int(data_cfg.get("seed", 1337)),
    )

    # Build model
    sp = model_cfg.get("sine_params", {}) or {}
    _vocab = model_cfg.get("vocab_size", None)
    vocab_size = dp.vocab_size if _vocab is None else int(_vocab)
    model = psannLM(
        base=str(model_cfg.get("base", "waveresnet")),
        d_model=int(model_cfg.get("d_model", 512)),
        n_layers=int(model_cfg.get("n_layers", 8)),
        n_heads=int(model_cfg.get("n_heads", 8)),
        d_mlp=int(model_cfg.get("d_mlp", 2048)),
        vocab_size=vocab_size,
        sine_params=dict(
            amp_init=float(sp.get("amp_init", 1.0)),
            freq_init=float(sp.get("freq_init", 1.0)),
            damp_init=float(sp.get("damp_init", 0.01)),
            trainable=bool(sp.get("trainable", True)),
        ),
        rope=bool(model_cfg.get("rope", True)),
    )

    # Train
    model.fit(
        dp,
        epochs=int(train_cfg.get("epochs", 1)),
        batch_tokens=int(train_cfg.get("batch_tokens", 131072)),
        lr=float(train_cfg.get("lr", 2e-4)),
        amp=str(train_cfg.get("amp", "bf16")),
        ddp=str(train_cfg.get("ddp", "auto")),
    )

    # Save final artifact
    ckpt_dir = str(train_cfg.get("checkpoint_dir", "runs/lm/exp"))
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save(os.path.join(ckpt_dir, "final_model.pt"))
    return 0


if __name__ == "__main__":  # pragma: no cover - manual test
    raise SystemExit(main())
