#!/usr/bin/env python
"""One-command training script for PSANN-LM (scales to multi-GPU with FSDP).

Example (~3B sizing sketch):

  python scripts/train_psann_lm.py \
    --base waveresnet --d-model 3072 --n-layers 30 --n-heads 24 \
    --tokenizer-backend tokenizers --train-tokenizer \
    --hf-dataset allenai/c4 --hf-name en --hf-split train --hf-text-key text \
    --hf-keep-ascii-only --hf-lang en \
    --batch-tokens 65536 --grad-accum-steps 8 --amp bf16 --grad-checkpoint \
    --fsdp full_shard --epochs 1 --save-interval-steps 2000 \
    --checkpoint-dir runs/lm/3b_en \
    --export-dir artifacts/psannlm_3b_run

Notes:
- Provide either a local manifest (`--data-manifest`) or stream directly from Hugging Face (`--hf-dataset`).
- `--train-tokenizer` trains a tokenizer before model training and saves artifacts alongside checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Iterator, List, Optional

import torch

# Local imports from package
from psann.lm.data.tokenizer import Tokenizer, TokenizerConfig
from psann.lm.data.dataset import (
    StreamingLMDataset,
    PackingConfig,
    HFTextStreamingLMDataset,
    build_text_filter,
)
from psann.lm.models.registry import get_base
from psann.lm.train.trainer import Trainer
from psann.lm.config import TrainConfig
from psann.lm.models.sine import SineConfig


def _read_manifest(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as fh:
        return [ln.strip() for ln in fh.readlines() if ln.strip()]


def _iter_manifest_texts(paths: list[str], limit: Optional[int]) -> Iterator[str]:
    yielded = 0
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                s = line.strip()
                if not s:
                    continue
                yield s
                yielded += 1
                if limit is not None and yielded >= limit:
                    return


def _iter_hf_texts(args: argparse.Namespace, limit: Optional[int]) -> Iterator[str]:
    from datasets import load_dataset  # type: ignore

    text_filter = build_text_filter(
        ascii_only=bool(args.hf_keep_ascii_only),
        languages=[s for s in (args.hf_lang or [])],
        lang_threshold=float(args.hf_lang_threshold),
    )
    stream = load_dataset(
        args.hf_dataset,
        name=args.hf_name,
        split=args.hf_split,
        streaming=True,
        revision=args.hf_revision,
    )
    if args.hf_shuffle:
        try:
            stream = stream.shuffle(seed=int(args.seed), buffer_size=int(args.hf_shuffle_buffer))
        except Exception:
            pass
    yielded = 0
    for row in stream:
        try:
            text = str(row.get(args.hf_text_key, "")).strip()
        except Exception:
            text = ""
        if not text:
            continue
        if not text_filter(text):
            continue
        yield text
        yielded += 1
        if limit is not None and yielded >= limit:
            break


def _tokenizer_sample_iterator(args: argparse.Namespace, shard_paths: list[str], limit: Optional[int]) -> Iterator[str]:
    if args.hf_dataset:
        yield from _iter_hf_texts(args, limit)
    else:
        yield from _iter_manifest_texts(shard_paths, limit)


def _ensure_tokenizer_config(special_map_path: Optional[str], max_length: int) -> Optional[str]:
    if not special_map_path:
        return None
    special = Path(special_map_path)
    if not special.exists():
        return None
    cfg_path = special.with_name("tokenizer_config.json")
    if cfg_path.exists():
        return str(cfg_path)
    try:
        with special.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return None
    config = dict(data)
    config["model_max_length"] = int(max_length)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg_path.open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    return str(cfg_path)


def _prepare_tokenizer(
    args: argparse.Namespace,
    shard_paths: list[str],
) -> tuple[Tokenizer, dict]:
    backend = str(args.tokenizer_backend or "auto").lower()
    if args.train_tokenizer and backend != "tokenizers":
        raise SystemExit("--train-tokenizer currently supports --tokenizer-backend tokenizers.")

    tok_model = args.tokenizer_model_path
    tok_special = args.tokenizer_special_map_path
    if args.train_tokenizer:
        tok_model = None
        tok_special = None

    artifacts: dict[str, Optional[str] | bool] = {
        "model": tok_model,
        "special_map": tok_special,
        "config": None,
        "dir": None,
        "trained": bool(args.train_tokenizer),
    }

    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    is_rank0 = rank == 0

    if args.train_tokenizer:
        limit = None if args.tokenizer_sample_limit is None or int(args.tokenizer_sample_limit) <= 0 else int(args.tokenizer_sample_limit)
        save_dir = Path(args.tokenizer_save_dir or os.path.join(args.checkpoint_dir, "tokenizer"))
        tok_json = save_dir / "tokenizer.json"
        special_map = save_dir / "special_tokens_map.json"
        done_flag = save_dir / ".done"
        save_dir.mkdir(parents=True, exist_ok=True)

        if is_rank0:
            train_cfg = TokenizerConfig(
                backend=backend,
                model_path=None,
                special_tokens_map_path=None,
                vocab_size=int(args.tokenizer_vocab_size),
                min_frequency=int(args.tokenizer_min_frequency),
                hf_passthrough_ids=(backend == "tokenizers"),
            )
            trainer_tok = Tokenizer(train_cfg)
            samples = _tokenizer_sample_iterator(args, shard_paths, limit)
            trainer_tok.fit(samples)
            trainer_tok.save(str(tok_json), special_tokens_map_path=str(special_map))
            artifacts["model"] = str(tok_json)
            artifacts["special_map"] = str(special_map)
            artifacts["dir"] = str(save_dir)
            cfg_path = _ensure_tokenizer_config(artifacts["special_map"], args.max_length)
            if cfg_path:
                artifacts["config"] = cfg_path
            done_flag.write_text("ok", encoding="utf-8")
            print(f"[tokenizer] Trained tokenizer saved to {tok_json}")
        else:
            wait_paths = [tok_json, special_map]
            while not all(p.exists() for p in wait_paths):
                if done_flag.exists():
                    break
                time.sleep(1.0)
            artifacts["model"] = str(tok_json)
            artifacts["special_map"] = str(special_map)
            artifacts["dir"] = str(save_dir)
            cfg_path = _ensure_tokenizer_config(artifacts["special_map"], args.max_length)
            if cfg_path:
                artifacts["config"] = cfg_path

    final_cfg = TokenizerConfig(
        backend=backend,
        model_path=artifacts["model"],
        special_tokens_map_path=artifacts["special_map"],
        vocab_size=int(args.tokenizer_vocab_size),
        min_frequency=int(args.tokenizer_min_frequency),
        hf_passthrough_ids=(backend == "tokenizers"),
    )
    tokenizer = Tokenizer(final_cfg)
    if artifacts["model"] is None:
        try:
            tokenizer.fit([""])
        except Exception:
            pass
    cfg_path = _ensure_tokenizer_config(artifacts["special_map"], args.max_length)
    if cfg_path:
        artifacts["config"] = cfg_path

    return tokenizer, artifacts


def _export_bundle(
    args: argparse.Namespace,
    *,
    final_ckpt: Path,
    tokenizer_artifacts: dict,
    shard_paths: list[str],
) -> None:
    if not args.export_dir:
        return
    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    def _copy(src: Path, dst_name: Optional[str] = None) -> Optional[Path]:
        if not src.exists():
            return None
        target = export_dir / (dst_name or src.name)
        shutil.copy2(src, target)
        return target

    copied = []
    model_copy = _copy(final_ckpt, "model.pt")
    if model_copy:
        copied.append(str(model_copy))
    for key in ("model", "special_map", "config"):
        path = tokenizer_artifacts.get(key)
        if not path:
            continue
        copied_path = _copy(Path(path))
        if copied_path:
            copied.append(str(copied_path))

    meta = {
        "model": {
            "base": args.base,
            "d_model": args.d_model,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "d_mlp": args.d_mlp if args.d_mlp is not None else 4 * args.d_model,
            "positional_encoding": args.pos_enc,
        },
        "tokenizer": {
            "backend": args.tokenizer_backend,
            "trained": bool(tokenizer_artifacts.get("trained")),
            "files": {k: Path(v).name for k, v in tokenizer_artifacts.items() if k in {"model", "special_map", "config"} and v},
        },
        "data": (
            {
                "type": "hf_dataset",
                "dataset": args.hf_dataset,
                "name": args.hf_name,
                "split": args.hf_split,
                "revision": args.hf_revision,
                "text_key": args.hf_text_key,
                "filters": {
                    "ascii_only": bool(args.hf_keep_ascii_only),
                    "languages": args.hf_lang or [],
                    "lang_threshold": args.hf_lang_threshold,
                },
            }
            if args.hf_dataset
            else {
                "type": "manifest",
                "path": args.data_manifest,
                "num_shards": len(shard_paths),
            }
        ),
        "training": {
            "epochs": args.epochs,
            "batch_tokens": args.batch_tokens,
            "grad_accum_steps": args.grad_accum_steps,
            "optimizer": args.optimizer,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "amp": args.amp,
            "fsdp": args.fsdp,
            "grad_checkpoint": bool(args.grad_checkpoint),
            "max_length": args.max_length,
        },
        "artifacts": copied,
    }
    meta_path = export_dir / "psann_artifacts.json"
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[export] Assets copied to {export_dir} (metadata: {meta_path})")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train PSANN-LM with streaming data and FSDP")

    # Model
    p.add_argument("--base", type=str, default="waveresnet")
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--d-mlp", type=int, default=None)
    p.add_argument("--vocab-size", type=int, default=None)
    p.add_argument("--pos-enc", type=str, default="rope", choices=["rope", "alibi", "sinusoidal"])

    # Tokenizer
    p.add_argument("--tokenizer-backend", type=str, default="auto", choices=["auto", "simple", "sentencepiece", "tokenizers"])
    p.add_argument("--tokenizer-model-path", type=str, default=None)
    p.add_argument("--tokenizer-special-map-path", type=str, default=None)
    p.add_argument("--hf-tokenizer-repo", type=str, default=None)
    p.add_argument("--hf-tokenizer-filename", type=str, default=None)
    p.add_argument("--hf-tokenizer-revision", type=str, default=None)
    p.add_argument("--train-tokenizer", action="store_true", help="Train a tokenizer before model training (tokenizers backend only)")
    p.add_argument("--tokenizer-save-dir", type=str, default=None, help="Directory to save newly trained tokenizer artifacts")
    p.add_argument("--tokenizer-vocab-size", type=int, default=50257)
    p.add_argument("--tokenizer-min-frequency", type=int, default=2)
    p.add_argument("--tokenizer-sample-limit", type=int, default=200000, help="Maximum number of documents to use for tokenizer training (0 = all)")

    # Data (choose one: manifest or HF dataset)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--data-manifest", type=str, help="Path to a newline-separated list of text shard files")
    g.add_argument("--hf-dataset", type=str, help="Hugging Face dataset repo id (e.g., allenai/c4)")
    p.add_argument("--hf-name", type=str, default=None, help="HF dataset subset/config name")
    p.add_argument("--hf-split", type=str, default="train", help="HF dataset split (default: train)")
    p.add_argument("--hf-revision", type=str, default=None, help="HF dataset revision (branch/tag/commit)")
    p.add_argument("--hf-text-key", type=str, default="text", help="Column containing raw text (default: text)")
    p.add_argument("--hf-shuffle", action="store_true", help="Shuffle streaming HF dataset with a buffer")
    p.add_argument("--hf-shuffle-buffer", type=int, default=10000, help="Streaming shuffle buffer size")
    p.add_argument("--hf-keep-ascii-only", action="store_true", help="Filter rows to ASCII-only text")
    p.add_argument("--hf-lang", action="append", default=None, help="Language code to keep (repeatable, requires langdetect)")
    p.add_argument("--hf-lang-threshold", type=float, default=0.8, help="Minimum langdetect probability to accept")
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--shuffle-docs", action="store_true")
    p.add_argument("--seed", type=int, default=1337)

    # Training
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-tokens", type=int, default=131072)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adamw8bit", "adafactor"])
    p.add_argument("--betas", type=str, default="0.9,0.95")
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--amp", type=str, default="bf16", choices=["bf16", "fp16", "fp32", "none"])
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--grad-checkpoint", action="store_true")
    p.add_argument("--ddp", type=str, default="auto", choices=["auto", "on", "off"])
    p.add_argument("--fsdp", type=str, default="off", choices=["off", "full_shard"])
    p.add_argument("--fsdp-auto-wrap", type=str, default="size", choices=["size", "none"])
    p.add_argument("--fsdp-min-params", type=int, default=1_000_000)
    p.add_argument("--fsdp-cpu-offload", action="store_true")
    p.add_argument("--steps-per-epoch", type=int, default=None)
    p.add_argument("--save-interval-steps", type=int, default=500)
    p.add_argument("--log-interval-steps", type=int, default=50)

    # Outputs
    p.add_argument("--checkpoint-dir", type=str, default="runs/lm/exp")
    p.add_argument("--out-ckpt", type=str, default=None)
    p.add_argument("--export-dir", type=str, default=None, help="Optional directory to gather model+tokenizer artifacts for Hugging Face upload")

    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    # Resolve tokenizer assets (optionally from HF Hub)
    tok_model = args.tokenizer_model_path
    tok_special = args.tokenizer_special_map_path
    if args.hf_tokenizer_repo and args.hf_tokenizer_filename and not tok_model:
        try:
            from huggingface_hub import hf_hub_download  # type: ignore

            tok_model = hf_hub_download(
                repo_id=args.hf_tokenizer_repo,
                filename=args.hf_tokenizer_filename,
                revision=args.hf_tokenizer_revision,
            )
            if tok_special is None:
                from pathlib import PurePosixPath

                special_name = str(PurePosixPath(args.hf_tokenizer_filename).with_name("special_tokens_map.json"))
                try:
                    tok_special = hf_hub_download(
                        repo_id=args.hf_tokenizer_repo,
                        filename=special_name,
                        revision=args.hf_tokenizer_revision,
                    )
                except Exception:
                    tok_special = None
        except Exception as e:
            raise SystemExit(f"Failed to download tokenizer assets: {e}")
    args.tokenizer_model_path = tok_model
    args.tokenizer_special_map_path = tok_special

    shard_paths: list[str] = []
    if args.data_manifest:
        shard_paths = _read_manifest(args.data_manifest)
        if not shard_paths:
            raise SystemExit("No data shards found in --data-manifest.")

    tokenizer, tokenizer_artifacts = _prepare_tokenizer(args, shard_paths)

    # Data
    pack = PackingConfig(max_length=int(args.max_length), pack_sequences=True)
    if args.hf_dataset:
        dataset = HFTextStreamingLMDataset(
            dataset=str(args.hf_dataset),
            tokenizer=tokenizer,
            cfg=pack,
            split=str(args.hf_split or "train"),
            text_key=str(args.hf_text_key or "text"),
            name=(str(args.hf_name) if args.hf_name else None),
            revision=(str(args.hf_revision) if args.hf_revision else None),
            shuffle=bool(args.hf_shuffle),
            seed=int(args.seed),
            shuffle_buffer=int(args.hf_shuffle_buffer),
            ascii_only=bool(args.hf_keep_ascii_only),
            languages=[s for s in args.hf_lang] if args.hf_lang else None,
            lang_threshold=float(args.hf_lang_threshold),
        )
    else:
        if not shard_paths:
            raise SystemExit("No data shards found in --data-manifest.")
        dataset = StreamingLMDataset(shard_paths, tokenizer, pack, shuffle_docs=bool(args.shuffle_docs), seed=int(args.seed))

    # Model
    vocab_size = int(args.vocab_size) if args.vocab_size is not None else int(tokenizer.vocab_size)
    d_mlp = int(args.d_mlp) if args.d_mlp is not None else 4 * int(args.d_model)
    factory = get_base(str(args.base))
    sine = SineConfig()
    model = factory(
        vocab_size=vocab_size,
        d_model=int(args.d_model),
        n_layers=int(args.n_layers),
        n_heads=int(args.n_heads),
        d_mlp=int(d_mlp),
        dropout=0.0,
        positional_encoding=str(args.pos_enc),
        mlp_activation="sine",
        sine=sine,
    )

    # Trainer
    # Parse betas
    try:
        b0, b1 = [float(x.strip()) for x in str(args.betas).split(",", 1)]
        betas = (b0, b1)
    except Exception:
        betas = (0.9, 0.95)

    tcfg = TrainConfig(
        epochs=int(args.epochs),
        batch_tokens=int(args.batch_tokens),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        optimizer=str(args.optimizer),
        betas=betas,
        eps=float(args.eps),
        amp=str(args.amp),
        grad_clip=float(args.grad_clip),
        grad_accum_steps=int(args.grad_accum_steps),
        ddp=str(args.ddp),
        fsdp=str(args.fsdp),
        fsdp_cpu_offload=bool(args.fsdp_cpu_offload),
        fsdp_auto_wrap_policy=str(args.fsdp_auto_wrap),
        fsdp_min_params=int(args.fsdp_min_params),
        steps_per_epoch=(int(args.steps_per_epoch) if args.steps_per_epoch is not None else None),
        checkpoint_dir=str(args.checkpoint_dir),
        log_interval_steps=int(args.log_interval_steps),
        save_interval_steps=int(args.save_interval_steps),
        grad_checkpoint=bool(args.grad_checkpoint),
    )
    trainer = Trainer(tcfg)
    trainer.train(model, dataset, max_length=int(args.max_length), val_dataset=None)

    # Copy final artifact if requested
    final_ckpt = Path(tcfg.checkpoint_dir) / "final.pt"
    if args.out_ckpt:
        dst = Path(args.out_ckpt)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if final_ckpt.exists():
            shutil.copy2(final_ckpt, dst)
            print(f"Copied final checkpoint -> {dst}")
        else:
            print(f"[warn] Expected final checkpoint not found at {final_ckpt}")

    _export_bundle(
        args,
        final_ckpt=final_ckpt,
        tokenizer_artifacts=tokenizer_artifacts,
        shard_paths=shard_paths,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
