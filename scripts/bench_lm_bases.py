#!/usr/bin/env python
"""Benchmark PSANN-LM base estimators on a quick WikiText-103 budget.

Runs a short training/eval loop for each base (e.g., respsann, waveresnet),
captures validation loss/perplexity/top-1 accuracy plus throughput, and
produces a leaderboard report under reports/benchmarks/.

Example:
  python scripts/bench_lm_bases.py \
    --config examples/lm/configs/base_compare_quick.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

try:
    import yaml
except Exception as exc:  # pragma: no cover - import guard
    raise SystemExit("Missing dependency 'pyyaml'. Install via `pip install pyyaml`.") from exc

import torch

from psann.lm.api import psannLM
from psann.lm.config import TrainConfig
from psann.lm.data.dataset import HFTextStreamingLMDataset, PackingConfig, build_text_filter
from psann.lm.data.tokenizer import Tokenizer, TokenizerConfig
from psann.lm.models.registry import get_base, list_bases
from psann.lm.models.sine import SineConfig
from psann.lm.train.trainer import Trainer


def _now_utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _system_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version.split(" ")[0],
        "platform": sys.platform,
        "torch": torch.__version__,
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append(
                {
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                }
            )
        info["gpus"] = gpus
    return info


def _default_config() -> Dict[str, Any]:
    return {
        "bench": {
            "run_name": "base_shootout",
            "out_dir": "reports/benchmarks",
            "bases": [],
            "seeds": [1337],
            "reuse_tokenizer": True,
            "with_lm_eval": False,
            "lm_eval_tasks": ["lambada_openai", "hellaswag"],
            "lm_eval_limit": 256,
            "lm_eval_num_fewshot": 0,
        },
        "data": {
            "dataset": "iohadrubin/wikitext-103-raw-v1",
            "name": None,
            "revision": None,
            "train_split": "train",
            "val_split": "validation",
            "text_key": "text",
            "streaming": True,
            "shuffle": True,
            "shuffle_buffer": 10000,
            "ascii_only": False,
            "languages": [],
            "lang_threshold": 0.8,
            "max_length": 512,
        },
        "tokenizer": {
            "backend": "tokenizers",
            "vocab_size": 16384,
            "min_frequency": 2,
            "sample_limit": 20000,
            "save_dir": "runs/tokenizers/base_compare_quick",
            "model_path": None,
            "special_tokens_map_path": None,
        },
        "train": {
            "d_model": 256,
            "n_layers": 4,
            "n_heads": 4,
            "d_mlp": 1024,
            "dropout": 0.0,
            "positional_encoding": "rope",
            "attn_impl": "auto",
            "batch_tokens": 32768,
            "grad_accum_steps": 1,
            "lr": 2e-4,
            "weight_decay": 0.01,
            "warmup_steps": 200,
            "amp": "bf16",
            "grad_checkpoint": False,
            "ddp": "off",
            "max_steps": 300,
            "log_interval_steps": 50,
            "save_interval_steps": 500,
            "sine_params": {
                "amp_init": 1.0,
                "freq_init": 1.0,
                "damp_init": 0.01,
                "trainable": True,
            },
        },
        "eval": {
            "batch_tokens": 32768,
            "max_tokens": 200000,
            "max_batches": 0,
        },
    }


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)  # type: ignore[index]
        else:
            base[key] = value
    return base


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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


def _iter_hf_texts(
    data_cfg: Dict[str, Any],
    limit: Optional[int],
    seed: int,
) -> Iterable[str]:
    from datasets import load_dataset  # type: ignore

    text_filter = build_text_filter(
        ascii_only=bool(data_cfg.get("ascii_only", False)),
        languages=list(data_cfg.get("languages") or []),
        lang_threshold=float(data_cfg.get("lang_threshold", 0.8)),
    )
    stream = load_dataset(
        data_cfg["dataset"],
        name=data_cfg.get("name"),
        split=data_cfg.get("train_split", "train"),
        streaming=True,
        revision=data_cfg.get("revision"),
    )
    if bool(data_cfg.get("shuffle", True)):
        try:
            stream = stream.shuffle(
                seed=int(seed), buffer_size=int(data_cfg.get("shuffle_buffer", 10000))
            )
        except Exception:
            pass
    yielded = 0
    for row in stream:
        try:
            text = str(row.get(data_cfg.get("text_key", "text"), "")).strip()
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


def _ensure_tokenizer(
    cfg: Dict[str, Any],
    *,
    outdir: Path,
    seed: int,
) -> tuple[Tokenizer, Dict[str, Any]]:
    tok_cfg = cfg.get("tokenizer", {})
    data_cfg = cfg.get("data", {})
    backend = str(tok_cfg.get("backend", "tokenizers")).lower()
    reuse = bool(cfg.get("bench", {}).get("reuse_tokenizer", True))
    save_dir = Path(tok_cfg.get("save_dir") or outdir / "tokenizer")
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = tok_cfg.get("model_path")
    special_map_path = tok_cfg.get("special_tokens_map_path")
    tok_json = save_dir / "tokenizer.json"
    special_map = save_dir / "special_tokens_map.json"

    if not model_path and reuse and tok_json.exists():
        model_path = str(tok_json)
        if special_map.exists():
            special_map_path = str(special_map)

    if backend == "tokenizers" and not model_path:
        limit = tok_cfg.get("sample_limit")
        limit_val = None if limit is None or int(limit) <= 0 else int(limit)
        train_cfg = TokenizerConfig(
            backend=backend,
            model_path=None,
            special_tokens_map_path=None,
            vocab_size=int(tok_cfg.get("vocab_size", 16384)),
            min_frequency=int(tok_cfg.get("min_frequency", 2)),
            hf_passthrough_ids=True,
        )
        trainer_tok = Tokenizer(train_cfg)
        samples = _iter_hf_texts(data_cfg, limit=limit_val, seed=seed)
        trainer_tok.fit(samples)
        trainer_tok.save(str(tok_json), special_tokens_map_path=str(special_map))
        model_path = str(tok_json)
        special_map_path = str(special_map)

    if backend == "tokenizers":
        hf_passthrough = True
    else:
        hf_passthrough = False

    final_cfg = TokenizerConfig(
        backend=backend,
        model_path=str(model_path) if model_path else None,
        special_tokens_map_path=str(special_map_path) if special_map_path else None,
        vocab_size=int(tok_cfg.get("vocab_size", 16384)),
        min_frequency=int(tok_cfg.get("min_frequency", 2)),
        hf_passthrough_ids=hf_passthrough,
    )
    tokenizer = Tokenizer(final_cfg)
    try:
        tokenizer.fit([""])
    except Exception:
        pass

    cfg_path = _ensure_tokenizer_config(
        special_map_path,
        int(data_cfg.get("max_length", 512)),
    )
    meta = {
        "backend": backend,
        "model_path": model_path,
        "special_tokens_map_path": special_map_path,
        "tokenizer_config_path": cfg_path,
        "vocab_size": tokenizer.vocab_size,
        "save_dir": str(save_dir),
    }
    return tokenizer, meta


def _build_stream_dataset(
    data_cfg: Dict[str, Any],
    tokenizer: Tokenizer,
    *,
    split: str,
    shuffle: bool,
    seed: int,
) -> HFTextStreamingLMDataset:
    pack = PackingConfig(
        max_length=int(data_cfg.get("max_length", 512)),
        pack_sequences=True,
    )
    return HFTextStreamingLMDataset(
        dataset=str(data_cfg.get("dataset")),
        name=data_cfg.get("name"),
        revision=data_cfg.get("revision"),
        split=str(split),
        text_key=str(data_cfg.get("text_key", "text")),
        shuffle=bool(shuffle),
        seed=int(seed),
        shuffle_buffer=int(data_cfg.get("shuffle_buffer", 10000)),
        tokenizer=tokenizer,
        cfg=pack,
        ascii_only=bool(data_cfg.get("ascii_only", False)),
        languages=list(data_cfg.get("languages") or []),
        lang_threshold=float(data_cfg.get("lang_threshold", 0.8)),
    )


def _eval_model(
    model: torch.nn.Module,
    dataset: HFTextStreamingLMDataset,
    *,
    max_tokens: int,
    max_batches: int,
    batch_tokens: int,
    seq_len: int,
    amp_mode: str,
) -> Dict[str, Any]:
    from torch.nn import functional as F
    from torch.utils.data import DataLoader

    device = next(model.parameters()).device
    batch_size = max(1, int(batch_tokens) // int(seq_len))

    use_amp = device.type == "cuda" and amp_mode in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if amp_mode == "bf16" else torch.float16
    autocast_ctx = torch.amp.autocast("cuda", dtype=amp_dtype) if use_amp else torch.no_grad()

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    batches = 0

    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in dl:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            with autocast_ctx:
                logits = model(input_ids)
                B, T, V = logits.shape
                loss = F.cross_entropy(
                    logits.view(B * T, V), labels.view(B * T), reduction="sum"
                )
            preds = logits.argmax(dim=-1)
            total_correct += int((preds == labels).sum().item())
            total_loss += float(loss.detach().float().item())
            total_tokens += int(B * T)
            batches += 1
            if max_batches > 0 and batches >= max_batches:
                break
            if max_tokens > 0 and total_tokens >= max_tokens:
                break

    if total_tokens <= 0:
        return {
            "val_loss": float("nan"),
            "val_ppl": float("nan"),
            "val_top1_acc": float("nan"),
            "val_tokens": 0,
            "val_batches": batches,
        }

    val_loss = total_loss / float(total_tokens)
    val_ppl = math.exp(val_loss)
    val_acc = float(total_correct) / float(total_tokens)
    return {
        "val_loss": round(val_loss, 6),
        "val_ppl": round(val_ppl, 6),
        "val_top1_acc": round(val_acc, 6),
        "val_tokens": int(total_tokens),
        "val_batches": int(batches),
    }


def _run_lm_eval(
    run_dir: Path,
    *,
    ckpt_path: str,
    tokenizer_meta: Dict[str, Any],
    tasks: List[str],
    limit: int,
    num_fewshot: int,
    device: str,
) -> Dict[str, Any]:
    import subprocess

    tok_backend = str(tokenizer_meta.get("backend", "auto"))
    tok_model = tokenizer_meta.get("model_path")
    tok_special = tokenizer_meta.get("special_tokens_map_path")

    cmd = [
        sys.executable,
        "scripts/run_lm_eval_psann.py",
        "--ckpt",
        str(ckpt_path),
        "--tokenizer-backend",
        tok_backend,
        "--tasks",
        ",".join(tasks),
        "--limit",
        str(int(limit)),
        "--num-fewshot",
        str(int(num_fewshot)),
        "--device",
        device,
        "--output",
        str(run_dir / "lm_eval.json"),
    ]
    if tok_model:
        cmd.extend(["--tokenizer-model-path", str(tok_model)])
    if tok_special:
        cmd.extend(["--tokenizer-special-map-path", str(tok_special)])

    try:
        subprocess.run(cmd, check=True)
    except Exception as exc:
        return {"status": "error", "error": str(exc), "cmd": " ".join(cmd)}
    return {"status": "ok", "output": str(run_dir / "lm_eval.json"), "cmd": " ".join(cmd)}


def _parse_bases(cli_value: Optional[str], cfg: Dict[str, Any]) -> List[str]:
    if cli_value:
        bases = [b.strip() for b in cli_value.split(",") if b.strip()]
        return bases
    bench_bases = cfg.get("bench", {}).get("bases") or []
    if bench_bases:
        return [str(b).strip() for b in bench_bases if str(b).strip()]
    discovered = list_bases()
    return discovered if discovered else ["respsann", "waveresnet"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark PSANN-LM bases quickly on WikiText-103")
    ap.add_argument("--config", type=str, default=None, help="Optional YAML config path")
    ap.add_argument("--out", type=str, default=None, help="Override output directory")
    ap.add_argument("--run-name", type=str, default=None, help="Override run name suffix")
    ap.add_argument("--bases", type=str, default=None, help="Comma-separated base list")
    ap.add_argument("--seeds", type=str, default=None, help="Comma-separated seed list")
    ap.add_argument("--max-steps", type=int, default=None, help="Override training steps")
    ap.add_argument(
        "--tokens-target",
        type=int,
        default=None,
        help="Approximate token budget per run (overrides max-steps)",
    )
    ap.add_argument(
        "--with-lm-eval",
        action="store_true",
        help="Run lm-eval (opt-in; expects lm_eval installed)",
    )
    ap.add_argument("--lm-eval-tasks", type=str, default=None)
    ap.add_argument("--lm-eval-limit", type=int, default=None)
    args = ap.parse_args()

    cfg = _default_config()
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            raise SystemExit(f"Config not found: {cfg_path}")
        with cfg_path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        _deep_update(cfg, loaded)

    if args.out:
        cfg["bench"]["out_dir"] = args.out
    if args.run_name:
        cfg["bench"]["run_name"] = args.run_name
    if args.max_steps is not None:
        cfg["train"]["max_steps"] = int(args.max_steps)
    if args.tokens_target is not None:
        cfg["train"]["tokens_target"] = int(args.tokens_target)
    if args.with_lm_eval:
        cfg["bench"]["with_lm_eval"] = True
    if args.lm_eval_tasks:
        cfg["bench"]["lm_eval_tasks"] = [
            t.strip() for t in args.lm_eval_tasks.split(",") if t.strip()
        ]
    if args.lm_eval_limit is not None:
        cfg["bench"]["lm_eval_limit"] = int(args.lm_eval_limit)
    if args.seeds:
        cfg["bench"]["seeds"] = [int(s) for s in args.seeds.split(",") if s.strip()]

    bench_cfg = cfg.get("bench", {})
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    eval_cfg = cfg.get("eval", {})

    bases = _parse_bases(args.bases, cfg)
    seeds = bench_cfg.get("seeds") or [1337]
    if not isinstance(seeds, list):
        seeds = [int(seeds)]

    out_root = Path(str(bench_cfg.get("out_dir", "reports/benchmarks"))).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    tag = _now_utc_tag()
    run_name = str(bench_cfg.get("run_name", "base_shootout"))
    outdir = out_root / f"{tag}_{run_name}"
    outdir.mkdir(parents=True, exist_ok=True)

    (outdir / "config_used.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )
    _write_json(outdir / "system.json", _system_info())

    # Ensure tokenizer once (reuse across bases).
    tok_seed = int(seeds[0]) if seeds else 1337
    tokenizer, tokenizer_meta = _ensure_tokenizer(cfg, outdir=outdir, seed=tok_seed)
    _write_json(outdir / "tokenizer_meta.json", tokenizer_meta)

    results: List[Dict[str, Any]] = []
    runs_dir = outdir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    for base in bases:
        for seed in seeds:
            run_dir = runs_dir / f"{base}_seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)

            print(f"[bench] base={base} seed={seed} -> {run_dir}")
            torch.manual_seed(int(seed))
            random.seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

            record: Dict[str, Any] = {
                "base": base,
                "seed": int(seed),
                "status": "ok",
            }

            try:
                vocab_size = int(tokenizer.vocab_size)
                factory = get_base(base)
                sine_cfg = train_cfg.get("sine_params", {}) or {}
                model = factory(
                    vocab_size=vocab_size,
                    d_model=int(train_cfg.get("d_model", 256)),
                    n_layers=int(train_cfg.get("n_layers", 4)),
                    n_heads=int(train_cfg.get("n_heads", 4)),
                    d_mlp=int(train_cfg.get("d_mlp", 1024)),
                    dropout=float(train_cfg.get("dropout", 0.0)),
                    positional_encoding=str(train_cfg.get("positional_encoding", "rope")),
                    mlp_activation=str(train_cfg.get("mlp_activation", "sine")),
                    sine=SineConfig(
                        amp_init=float(sine_cfg.get("amp_init", 1.0)),
                        freq_init=float(sine_cfg.get("freq_init", 1.0)),
                        damp_init=float(sine_cfg.get("damp_init", 0.01)),
                        trainable=bool(sine_cfg.get("trainable", True)),
                    ),
                    attn_impl=str(train_cfg.get("attn_impl", "auto")),
                )
                param_count = sum(p.numel() for p in model.parameters())
                record["param_count"] = int(param_count)

                train_ds = _build_stream_dataset(
                    data_cfg,
                    tokenizer,
                    split=str(data_cfg.get("train_split", "train")),
                    shuffle=bool(data_cfg.get("shuffle", True)),
                    seed=int(seed),
                )
                val_ds = _build_stream_dataset(
                    data_cfg,
                    tokenizer,
                    split=str(data_cfg.get("val_split", "validation")),
                    shuffle=False,
                    seed=int(seed),
                )

                batch_tokens = int(train_cfg.get("batch_tokens", 32768))
                grad_accum = int(train_cfg.get("grad_accum_steps", 1))
                seq_len = int(data_cfg.get("max_length", 512))
                batch_size = max(1, batch_tokens // seq_len)
                tokens_per_step = batch_size * seq_len * grad_accum
                tokens_target = int(train_cfg.get("tokens_target", 0) or 0)
                max_steps = int(train_cfg.get("max_steps", 300))
                if tokens_target > 0:
                    max_steps = max(1, tokens_target // max(1, tokens_per_step))

                tcfg = TrainConfig(
                    epochs=1,
                    batch_tokens=int(batch_tokens),
                    lr=float(train_cfg.get("lr", 2e-4)),
                    warmup_steps=int(train_cfg.get("warmup_steps", 200)),
                    weight_decay=float(train_cfg.get("weight_decay", 0.01)),
                    amp=str(train_cfg.get("amp", "bf16")),
                    grad_checkpoint=bool(train_cfg.get("grad_checkpoint", False)),
                    grad_accum_steps=int(grad_accum),
                    ddp=str(train_cfg.get("ddp", "off")),
                    fsdp="off",
                    steps_per_epoch=int(max_steps),
                    checkpoint_dir=str(run_dir / "checkpoints"),
                    log_interval_steps=int(train_cfg.get("log_interval_steps", 50)),
                    save_interval_steps=int(train_cfg.get("save_interval_steps", 500)),
                    dataloader_num_workers=0,
                    dataloader_prefetch_factor=2,
                    dataloader_persistent_workers=False,
                )

                trainer = Trainer(tcfg)
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                t0 = time.time()
                trainer.train(
                    model,
                    train_ds,
                    max_length=int(seq_len),
                    val_dataset=None,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.time() - t0

                steps_done = int(getattr(trainer.state, "step", 0))
                world_size = int(os.environ.get("WORLD_SIZE", "1"))
                total_tokens = steps_done * tokens_per_step * max(1, world_size)
                tokens_per_s = (total_tokens / elapsed) if elapsed > 0 else 0.0
                record.update(
                    {
                        "train_steps": steps_done,
                        "train_tokens": int(total_tokens),
                        "train_elapsed_s": round(elapsed, 4),
                        "train_tokens_per_s": round(tokens_per_s, 2),
                    }
                )

                peak_mem = None
                peak_reserved = None
                if torch.cuda.is_available():
                    try:
                        peak_mem = torch.cuda.max_memory_allocated() / float(1024**3)
                        peak_reserved = torch.cuda.max_memory_reserved() / float(1024**3)
                    except Exception:
                        peak_mem = None
                        peak_reserved = None
                record["peak_cuda_mem_gb"] = None if peak_mem is None else round(peak_mem, 4)
                record["peak_cuda_reserved_gb"] = (
                    None if peak_reserved is None else round(peak_reserved, 4)
                )

                eval_batch_tokens = int(eval_cfg.get("batch_tokens", batch_tokens))
                eval_metrics = _eval_model(
                    model,
                    val_ds,
                    max_tokens=int(eval_cfg.get("max_tokens", 200000)),
                    max_batches=int(eval_cfg.get("max_batches", 0)),
                    batch_tokens=int(eval_batch_tokens),
                    seq_len=int(seq_len),
                    amp_mode=str(train_cfg.get("amp", "bf16")).lower(),
                )
                record.update(eval_metrics)

                lm = psannLM(
                    base=base,
                    d_model=int(train_cfg.get("d_model", 256)),
                    n_layers=int(train_cfg.get("n_layers", 4)),
                    n_heads=int(train_cfg.get("n_heads", 4)),
                    d_mlp=int(train_cfg.get("d_mlp", 1024)),
                    vocab_size=vocab_size,
                    positional_encoding=str(train_cfg.get("positional_encoding", "rope")),
                    sine_params=sine_cfg,
                )
                lm._model = model  # reuse trained weights
                ckpt_path = run_dir / "final_model.pt"
                lm.save(str(ckpt_path))
                record["final_model_path"] = str(ckpt_path)

                if bench_cfg.get("with_lm_eval", False):
                    if not tokenizer_meta.get("model_path"):
                        record["lm_eval"] = {
                            "status": "skipped",
                            "reason": "tokenizer_model_path missing; lm-eval requires tokenizer parity",
                        }
                    else:
                        lm_eval = _run_lm_eval(
                            run_dir,
                            ckpt_path=str(ckpt_path),
                            tokenizer_meta=tokenizer_meta,
                            tasks=[str(t) for t in bench_cfg.get("lm_eval_tasks", [])],
                            limit=int(bench_cfg.get("lm_eval_limit", 256)),
                            num_fewshot=int(bench_cfg.get("lm_eval_num_fewshot", 0)),
                            device="cuda" if torch.cuda.is_available() else "cpu",
                        )
                        record["lm_eval"] = lm_eval

            except Exception as exc:
                record["status"] = "error"
                record["error"] = str(exc)

            results.append(record)
            _write_json(run_dir / "metrics.json", record)

            # Best-effort cleanup between runs
            try:
                del model
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = {
        "timestamp_utc": tag,
        "run_name": run_name,
        "bases": bases,
        "seeds": seeds,
        "system": _system_info(),
        "tokenizer": tokenizer_meta,
        "results": results,
    }
    _write_json(outdir / "summary.json", summary)

    # Write summary.csv
    csv_rows = [
        "base,seed,status,param_count,train_steps,train_tokens,train_tokens_per_s,train_elapsed_s,peak_cuda_mem_gb,val_loss,val_ppl,val_top1_acc"
    ]
    for row in results:
        csv_rows.append(
            ",".join(
                [
                    str(row.get("base", "")),
                    str(row.get("seed", "")),
                    str(row.get("status", "")),
                    str(row.get("param_count", "")),
                    str(row.get("train_steps", "")),
                    str(row.get("train_tokens", "")),
                    str(row.get("train_tokens_per_s", "")),
                    str(row.get("train_elapsed_s", "")),
                    str(row.get("peak_cuda_mem_gb", "")),
                    str(row.get("val_loss", "")),
                    str(row.get("val_ppl", "")),
                    str(row.get("val_top1_acc", "")),
                ]
            )
        )
    (outdir / "summary.csv").write_text("\n".join(csv_rows) + "\n", encoding="utf-8")

    # Leaderboard markdown
    ok_rows = [r for r in results if r.get("status") == "ok"]
    ok_rows.sort(key=lambda r: (r.get("val_ppl") is None, r.get("val_ppl", float("inf"))))
    lines = [
        "# Base Estimator Shootout",
        "",
        "| base | seed | val_ppl | val_loss | val_top1_acc | tokens/s | peak_cuda_mem_gb |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in ok_rows:
        lines.append(
            "| {base} | {seed} | {val_ppl} | {val_loss} | {val_top1_acc} | {train_tokens_per_s} | {peak_cuda_mem_gb} |".format(
                base=r.get("base"),
                seed=r.get("seed"),
                val_ppl=r.get("val_ppl"),
                val_loss=r.get("val_loss"),
                val_top1_acc=r.get("val_top1_acc"),
                train_tokens_per_s=r.get("train_tokens_per_s"),
                peak_cuda_mem_gb=r.get("peak_cuda_mem_gb"),
            )
        )
    (outdir / "leaderboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[bench] Wrote summary -> {outdir / 'summary.json'}")
    print(f"[bench] Leaderboard -> {outdir / 'leaderboard.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
