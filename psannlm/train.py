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
import re
import shutil
import time
from pathlib import Path
from typing import Iterator, Optional, Any


# Local imports from package
from psannlm.lm.data.tokenizer import Tokenizer, TokenizerConfig
from psannlm.lm.data.dataset import (
    StreamingLMDataset,
    PackingConfig,
    HFTextStreamingLMDataset,
    build_text_filter,
)
from psannlm.lm.models.registry import get_base
from psannlm.lm.train.trainer import Trainer
from psannlm.lm.config import TrainConfig
from psannlm.lm.models.sine import SineConfig
from .data_stream import streamed_token_iterator
from .data_loader import build_stream_loader


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


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


def _tokenizer_sample_iterator(
    args: argparse.Namespace, shard_paths: list[str], limit: Optional[int]
) -> Iterator[str]:
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
        limit = (
            None
            if args.tokenizer_sample_limit is None or int(args.tokenizer_sample_limit) <= 0
            else int(args.tokenizer_sample_limit)
        )
        save_dir = Path(args.tokenizer_save_dir or os.path.join(args.checkpoint_dir, "tokenizer"))
        tok_json = save_dir / "tokenizer.json"
        special_map = save_dir / "special_tokens_map.json"
        done_flag = save_dir / ".done"
        save_dir.mkdir(parents=True, exist_ok=True)

        already_trained = tok_json.exists() and special_map.exists() and done_flag.exists()
        if already_trained:
            artifacts["model"] = str(tok_json)
            artifacts["special_map"] = str(special_map)
            artifacts["dir"] = str(save_dir)
            cfg_path = _ensure_tokenizer_config(artifacts["special_map"], args.max_length)
            if cfg_path:
                artifacts["config"] = cfg_path
            if is_rank0:
                print(f"[tokenizer] Reusing existing tokenizer at {tok_json}")
        elif is_rank0:
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
            "files": {
                k: Path(v).name
                for k, v in tokenizer_artifacts.items()
                if k in {"model", "special_map", "config"} and v
            },
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
    p.add_argument(
        "--attn-impl",
        type=str,
        default="math",
        choices=["math", "sdpa", "auto"],
        help="Attention implementation: 'math' = explicit matmul+softmax, 'sdpa'/'auto' use torch.scaled_dot_product_attention when available.",
    )
    p.add_argument("--sine-amp-init", type=float, default=None)
    p.add_argument("--sine-amp-init-std", type=float, default=None)
    p.add_argument("--sine-freq-init", type=float, default=None)
    p.add_argument("--sine-freq-init-std", type=float, default=None)
    p.add_argument("--sine-damp-init", type=float, default=None)
    p.add_argument("--sine-damp-init-std", type=float, default=None)
    p.add_argument(
        "--sine-trainable",
        type=str2bool,
        default=True,
        help="Whether sine parameters are trainable (default: true).",
    )

    # Tokenizer
    p.add_argument(
        "--tokenizer-backend",
        type=str,
        default="auto",
        choices=["auto", "simple", "sentencepiece", "tokenizers"],
    )
    p.add_argument("--tokenizer-model-path", type=str, default=None)
    p.add_argument("--tokenizer-special-map-path", type=str, default=None)
    p.add_argument("--hf-tokenizer-repo", type=str, default=None)
    p.add_argument("--hf-tokenizer-filename", type=str, default=None)
    p.add_argument("--hf-tokenizer-revision", type=str, default=None)
    p.add_argument(
        "--train-tokenizer",
        action="store_true",
        help="Train a tokenizer before model training (tokenizers backend only)",
    )
    p.add_argument(
        "--tokenizer-save-dir",
        type=str,
        default=None,
        help="Directory to save newly trained tokenizer artifacts",
    )
    p.add_argument("--tokenizer-vocab-size", type=int, default=50257)
    p.add_argument("--tokenizer-min-frequency", type=int, default=2)
    p.add_argument(
        "--tokenizer-sample-limit",
        type=int,
        default=200000,
        help="Maximum number of documents to use for tokenizer training (0 = all)",
    )

    # Data (choose one: manifest or HF dataset)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--data-manifest", type=str, help="Path to a newline-separated list of text shard files"
    )
    g.add_argument(
        "--hf-dataset",
        "--dataset-name",
        "--dataset_name",
        dest="hf_dataset",
        type=str,
        help="Hugging Face dataset repo id (e.g., allenai/c4)",
    )
    p.add_argument(
        "--hf-name",
        "--dataset-config",
        "--dataset_config",
        type=str,
        default=None,
        help="HF dataset subset/config name",
    )
    p.add_argument(
        "--hf-split",
        "--dataset-split",
        "--dataset_split",
        type=str,
        default="train",
        help="HF dataset split (default: train)",
    )
    p.add_argument(
        "--hf-revision", type=str, default=None, help="HF dataset revision (branch/tag/commit)"
    )
    p.add_argument(
        "--hf-text-key",
        "--dataset-text-field",
        "--dataset_text_field",
        type=str,
        default="text",
        help="Column containing raw text (default: text)",
    )
    p.add_argument(
        "--hf-shuffle", action="store_true", help="Shuffle streaming HF dataset with a buffer"
    )
    p.add_argument(
        "--hf-shuffle-buffer", type=int, default=10000, help="Streaming shuffle buffer size"
    )
    p.add_argument(
        "--hf-keep-ascii-only", action="store_true", help="Filter rows to ASCII-only text"
    )
    p.add_argument(
        "--hf-lang",
        action="append",
        default=None,
        help="Language code to keep (repeatable, requires langdetect)",
    )
    p.add_argument(
        "--hf-lang-threshold",
        type=float,
        default=0.8,
        help="Minimum langdetect probability to accept",
    )
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument(
        "--seq-len",
        "--seq_len",
        type=int,
        default=None,
        help="Override sequence length (defaults to --max-length)",
    )
    p.add_argument(
        "--dataset-streaming",
        "--dataset_streaming",
        type=str,
        default="false",
        help="true/false to enable HF streaming iterator",
    )
    p.add_argument(
        "--pack-buffer-tokens",
        "--pack_buffer_tokens",
        type=int,
        default=2_000_000,
        help="Soft cap for streaming tokenizer buffer",
    )
    p.add_argument(
        "--hf-cache-limit-gb",
        type=float,
        default=None,
        help="If set, periodically trim the HF datasets cache to this size (in GB).",
    )
    p.add_argument("--shuffle-docs", action="store_true")
    p.add_argument("--seed", type=int, default=1337)

    # Training
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-tokens", type=int, default=131072)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument(
        "--warmup-steps",
        type=int,
        default=2000,
        help="Number of optimizer steps to linearly warm up the learning rate.",
    )
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument(
        "--optimizer", type=str, default="adamw", choices=["adamw", "adamw8bit", "adafactor"]
    )
    p.add_argument("--betas", type=str, default="0.9,0.95")
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--amp", type=str, default="bf16", choices=["bf16", "fp16", "fp32", "none"])
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--grad-checkpoint", action="store_true")
    p.add_argument(
        "--cuda-memory-fraction",
        type=float,
        default=None,
        help="If set (0-1], caps this process to a fraction of total VRAM (leaves headroom for other processes).",
    )
    p.add_argument(
        "--cuda-empty-cache-after-init",
        action="store_true",
        help="Call torch.cuda.empty_cache() once after init/compile (reduces reserved VRAM; may reduce performance).",
    )
    p.add_argument(
        "--cuda-empty-cache-interval-steps",
        type=int,
        default=0,
        help="If >0, call torch.cuda.empty_cache() every N optimizer steps (reduces reserved VRAM; may reduce performance).",
    )
    p.add_argument(
        "--eval-data-files",
        type=str,
        default=None,
        help="Local JSONL file used for periodic held-out perplexity evaluation (expects a 'text' field by default).",
    )
    p.add_argument("--eval-text-key", type=str, default="text")
    p.add_argument(
        "--eval-interval-steps",
        type=int,
        default=0,
        help="If >0, run eval every N optimizer steps (in addition to eval at checkpoint saves).",
    )
    p.add_argument(
        "--eval-max-batches",
        type=int,
        default=512,
        help="Maximum eval batches per evaluation run (0 = default cap).",
    )
    p.add_argument(
        "--eval-create-shard",
        action="store_true",
        help="If set and --eval-data-files does not exist, create it by sampling from the training HF dataset.",
    )
    p.add_argument(
        "--eval-target-tokens",
        type=int,
        default=10_000_000,
        help="When creating an eval shard, target this many tokens (approx; uses the active tokenizer).",
    )
    p.add_argument(
        "--eval-hf-split",
        type=str,
        default=None,
        help="HF split used when creating eval shard (default: same as --hf-split).",
    )
    p.add_argument(
        "--torch-compile",
        action="store_true",
        help="Enable torch.compile for the model (single GPU only; skipped under DDP/FSDP).",
    )
    p.add_argument(
        "--torch-compile-mode",
        type=str,
        default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode.",
    )
    p.add_argument(
        "--torch-compile-fullgraph",
        action="store_true",
        help="Pass fullgraph=True to torch.compile (can be more brittle).",
    )
    p.add_argument(
        "--torch-compile-dynamic",
        action="store_true",
        help="Pass dynamic=True to torch.compile (for dynamic shapes; can reduce performance).",
    )
    p.add_argument(
        "--log-gpu-mem",
        action="store_true",
        help="Log basic GPU memory stats (allocated/reserved/max, in GB) at each log interval on the main rank.",
    )
    p.add_argument("--ddp", type=str, default="auto", choices=["auto", "on", "off"])
    p.add_argument("--fsdp", type=str, default="off", choices=["off", "full_shard"])
    p.add_argument("--fsdp-auto-wrap", type=str, default="size", choices=["size", "none"])
    p.add_argument("--fsdp-min-params", type=int, default=1_000_000)
    p.add_argument("--fsdp-cpu-offload", action="store_true")
    p.add_argument("--steps-per-epoch", type=int, default=None)
    p.add_argument(
        "--max-steps",
        "--max_steps",
        type=int,
        default=0,
        help="Total optimizer steps override (required for streaming mode)",
    )
    p.add_argument("--save-interval-steps", type=int, default=500)
    p.add_argument("--log-interval-steps", type=int, default=50)
    p.add_argument(
        "--tokens-target",
        "--tokens_target",
        type=int,
        default=0,
        help="Approximate token budget to derive max_steps",
    )
    p.add_argument(
        "--tokens-per-step",
        "--tokens_per_step",
        type=int,
        default=0,
        help="Override tokens processed per optimizer step",
    )
    p.add_argument(
        "--num-workers", "--dataloader-num-workers", "--dataloader_num_workers", type=int, default=8
    )
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--no-persistent-workers", action="store_true")

    # Outputs
    p.add_argument("--checkpoint-dir", type=str, default="runs/lm/exp")
    p.add_argument("--out-ckpt", type=str, default=None)
    p.add_argument(
        "--export-dir",
        type=str,
        default=None,
        help="Optional directory to gather model+tokenizer artifacts for Hugging Face upload",
    )
    p.add_argument(
        "--resume-ckpt",
        type=str,
        default=None,
        help="Optional path to a checkpoint to resume training (loads model + optimizer state)",
    )

    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    seq_len = int(args.seq_len) if args.seq_len is not None else int(args.max_length)
    args.max_length = seq_len
    micro_batch = max(1, int(args.batch_tokens) // seq_len)
    tokens_per_micro = micro_batch * seq_len
    tokens_per_step = (
        int(args.tokens_per_step)
        if int(args.tokens_per_step) > 0
        else tokens_per_micro * max(1, int(args.grad_accum_steps))
    )
    tokens_target = max(0, int(args.tokens_target))
    if tokens_target > 0 and int(args.max_steps) <= 0 and tokens_per_step > 0:
        args.max_steps = max(1, tokens_target // tokens_per_step)
    use_streaming = str2bool(args.dataset_streaming)
    if use_streaming and int(args.max_steps) <= 0:
        raise SystemExit(
            "--max_steps (or --tokens-target) must be set when --dataset-streaming true."
        )
    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    if rank == 0:
        target_msg = f"{tokens_target:,}" if tokens_target > 0 else "n/a"
        max_steps_msg = f"{int(args.max_steps):,}" if int(args.max_steps) > 0 else "auto"
        global_tokens_per_step = tokens_per_step * world_size
        print(
            f"[budget] seq_len={seq_len} micro_batch={micro_batch} tokens_per_step={tokens_per_step:,} "
            f"tokens_per_step_global≈{global_tokens_per_step:,} target_tokens≈{target_msg} max_steps={max_steps_msg}"
        )

    if args.cuda_memory_fraction is not None:
        frac = float(args.cuda_memory_fraction)
        if not (0.0 < frac <= 1.0):
            raise SystemExit("--cuda-memory-fraction must be in (0, 1].")
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(frac)
                if rank == 0:
                    print(f"[gpu] set_per_process_memory_fraction={frac:.3f}")
        except Exception as exc:
            if rank == 0:
                print(f"[warn] Failed to set CUDA memory fraction: {exc}")

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

                special_name = str(
                    PurePosixPath(args.hf_tokenizer_filename).with_name("special_tokens_map.json")
                )
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

    # Optional: create a held-out eval shard (JSONL with a 'text' field) so eval can be reused.
    if args.eval_data_files:
        eval_path = Path(str(args.eval_data_files))
        if args.eval_create_shard and not eval_path.exists():
            if not args.hf_dataset:
                raise SystemExit("--eval-create-shard requires --hf-dataset (HF source).")
            tmp_path = eval_path.with_suffix(eval_path.suffix + ".tmp")
            done_path = eval_path.with_suffix(eval_path.suffix + ".done")
            if rank == 0:
                from datasets import load_dataset  # type: ignore

                eval_path.parent.mkdir(parents=True, exist_ok=True)
                eval_split = str(args.eval_hf_split or args.hf_split or "train")
                eval_seed = int(args.seed) + 1009
                eval_buffer = int(args.hf_shuffle_buffer) if bool(args.hf_shuffle) else 0
                text_filter = build_text_filter(
                    ascii_only=bool(args.hf_keep_ascii_only),
                    languages=[s for s in (args.hf_lang or [])],
                    lang_threshold=float(args.hf_lang_threshold),
                )
                ds = load_dataset(
                    str(args.hf_dataset),
                    name=(str(args.hf_name) if args.hf_name else None),
                    split=eval_split,
                    streaming=True,
                    revision=(str(args.hf_revision) if args.hf_revision else None),
                )
                if eval_buffer > 0:
                    try:
                        ds = ds.shuffle(seed=eval_seed, buffer_size=eval_buffer)
                    except Exception:
                        pass

                target_tokens = max(1, int(args.eval_target_tokens))
                total_tokens = 0
                docs = 0
                with tmp_path.open("w", encoding="utf-8") as f:
                    for row in ds:
                        try:
                            text = str(row.get(str(args.hf_text_key or "text"), "")).strip()
                        except Exception:
                            text = ""
                        if not text:
                            continue
                        if not text_filter(text):
                            continue
                        try:
                            ids = tokenizer.encode(text, add_specials=True)
                        except Exception:
                            continue
                        if not ids:
                            continue
                        f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                        total_tokens += len(ids)
                        docs += 1
                        if total_tokens >= target_tokens:
                            break
                tmp_path.replace(eval_path)
                done_path.write_text("ok", encoding="utf-8")
                print(f"[eval] Wrote {docs} docs, {total_tokens:,} tokens -> {eval_path}")
            else:
                while not eval_path.exists():
                    if done_path.exists():
                        break
                    time.sleep(1.0)

    # Data
    pack = PackingConfig(max_length=seq_len, pack_sequences=True)
    stream_loader = None
    effective_steps_per_epoch: Optional[int] = (
        int(args.steps_per_epoch) if args.steps_per_epoch is not None else None
    )
    if use_streaming:
        if not args.hf_dataset:
            raise SystemExit("--dataset-streaming true requires --hf-dataset/--dataset-name.")
        text_filter = build_text_filter(
            ascii_only=bool(args.hf_keep_ascii_only),
            languages=[s for s in (args.hf_lang or [])],
            lang_threshold=float(args.hf_lang_threshold),
        )

        resume_step = 0
        if args.resume_ckpt:
            m = re.search(r"ckpt_step(\\d+)", os.path.basename(str(args.resume_ckpt)))
            if m:
                try:
                    resume_step = int(m.group(1))
                except Exception:
                    resume_step = 0
            if resume_step <= 0:
                try:
                    import torch

                    payload = torch.load(str(args.resume_ckpt), map_location="cpu")
                    ckpt_state = payload.get("state", {}) if isinstance(payload, dict) else {}
                    resume_step = int(ckpt_state.get("step", 0) or 0)
                except Exception:
                    resume_step = 0

        skip_sequences = 0
        if resume_step > 0:
            skip_sequences = int(resume_step) * int(micro_batch) * max(1, int(args.grad_accum_steps))
            if rank == 0:
                print(
                    f"[resume] step={resume_step} -> skip_sequences={skip_sequences:,} (seq_len={seq_len})"
                )

        def _iterator(worker_info: Optional[Any] = None):
            worker_id = getattr(worker_info, "id", None) if worker_info else None
            num_workers = getattr(worker_info, "num_workers", None) if worker_info else None
            return streamed_token_iterator(
                dataset_name=str(args.hf_dataset),
                split=str(args.hf_split or "train"),
                tokenizer=tokenizer,
                dataset_config=(str(args.hf_name) if args.hf_name else None),
                dataset_revision=(str(args.hf_revision) if args.hf_revision else None),
                text_field=str(args.hf_text_key or "text"),
                seq_len=seq_len,
                shuffle_seed=int(args.seed),
                shuffle_buffer=(int(args.hf_shuffle_buffer) if bool(args.hf_shuffle) else 0),
                pack_buffer_tokens=int(args.pack_buffer_tokens),
                skip_sequences=int(skip_sequences),
                worker_id=worker_id,
                num_workers=num_workers,
                text_filter=text_filter,
            )

        num_workers = int(args.num_workers)
        if skip_sequences > 0 and num_workers > 0:
            raise SystemExit(
                "--resume-ckpt in streaming mode requires --num-workers 0 for deterministic shuffling/skip. "
                "Set --num-workers 0 from the start (initial run and every resume) to avoid repeating data."
            )

        stream_loader = build_stream_loader(
            _iterator,
            batch_size=micro_batch,
            num_workers=num_workers,
        )
        dataset = stream_loader.dataset  # type: ignore[assignment]
        effective_steps_per_epoch = int(args.max_steps)
        args.epochs = 1
    elif args.hf_dataset:
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
        dataset = StreamingLMDataset(
            shard_paths, tokenizer, pack, shuffle_docs=bool(args.shuffle_docs), seed=int(args.seed)
        )

    if effective_steps_per_epoch is None and int(args.max_steps) > 0:
        effective_steps_per_epoch = int(args.max_steps)

    # Model
    vocab_size = int(args.vocab_size) if args.vocab_size is not None else int(tokenizer.vocab_size)
    d_mlp = int(args.d_mlp) if args.d_mlp is not None else 4 * int(args.d_model)
    factory = get_base(str(args.base))
    sine = SineConfig()
    if args.sine_amp_init is not None:
        sine.amp_init = float(args.sine_amp_init)
    if args.sine_amp_init_std is not None:
        sine.amp_init_std = float(args.sine_amp_init_std)
    if args.sine_freq_init is not None:
        sine.freq_init = float(args.sine_freq_init)
    if args.sine_freq_init_std is not None:
        sine.freq_init_std = float(args.sine_freq_init_std)
    if args.sine_damp_init is not None:
        sine.damp_init = float(args.sine_damp_init)
    if args.sine_damp_init_std is not None:
        sine.damp_init_std = float(args.sine_damp_init_std)
    sine.trainable = bool(args.sine_trainable)
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
        attn_impl=str(args.attn_impl),
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
        warmup_steps=int(args.warmup_steps),
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
        steps_per_epoch=effective_steps_per_epoch,
        checkpoint_dir=str(args.checkpoint_dir),
        log_interval_steps=int(args.log_interval_steps),
        save_interval_steps=int(args.save_interval_steps),
        grad_checkpoint=bool(args.grad_checkpoint),
        eval_interval_steps=int(args.eval_interval_steps),
        eval_max_batches=int(args.eval_max_batches),
        torch_compile=bool(getattr(args, "torch_compile", False)),
        torch_compile_mode=str(getattr(args, "torch_compile_mode", "default")),
        torch_compile_fullgraph=bool(getattr(args, "torch_compile_fullgraph", False)),
        torch_compile_dynamic=bool(getattr(args, "torch_compile_dynamic", False)),
        log_gpu_mem=bool(getattr(args, "log_gpu_mem", False)),
        cuda_empty_cache_after_init=bool(getattr(args, "cuda_empty_cache_after_init", False)),
        cuda_empty_cache_interval_steps=int(getattr(args, "cuda_empty_cache_interval_steps", 0)),
        dataloader_num_workers=int(args.num_workers),
        dataloader_prefetch_factor=int(args.prefetch_factor),
        dataloader_persistent_workers=not bool(args.no_persistent_workers),
        hf_cache_limit_gb=(
            float(args.hf_cache_limit_gb) if args.hf_cache_limit_gb is not None else None
        ),
    )
    trainer = Trainer(tcfg)

    val_dataset = None
    if args.eval_data_files:
        eval_path = Path(str(args.eval_data_files))
        if eval_path.exists():
            val_dataset = HFTextStreamingLMDataset(
                dataset="json",
                data_files=str(eval_path),
                tokenizer=tokenizer,
                cfg=pack,
                split="train",
                text_key=str(args.eval_text_key or "text"),
                shuffle=False,
                seed=int(args.seed),
                shuffle_buffer=0,
            )
        elif rank == 0:
            print(f"[warn] --eval-data-files set but file not found: {eval_path}")

    trainer.train(
        model,
        dataset,
        max_length=seq_len,
        val_dataset=val_dataset,
        data_loader=stream_loader,
        resume_checkpoint=args.resume_ckpt,
    )

    stream_exhausted_early = (
        use_streaming and int(args.max_steps) > 0 and int(trainer.state.step) < int(args.max_steps)
    )
    if stream_exhausted_early and rank == 0:
        global_tokens_per_step = tokens_per_step * world_size
        trained_tokens = int(trainer.state.step) * int(global_tokens_per_step)
        target_msg = f"{tokens_target:,}" if tokens_target > 0 else "n/a"
        print(
            "[warn] Streaming dataset exhausted early "
            f"(step={trainer.state.step} < max_steps={int(args.max_steps):,}; "
            f"trained_tokens≈{trained_tokens:,} vs target_tokens≈{target_msg})."
        )

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

    return 2 if stream_exhausted_early else 0


if __name__ == "__main__":
    raise SystemExit(main())
