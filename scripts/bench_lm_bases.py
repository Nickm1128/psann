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
import copy
import json
import math
import os
import random
import re
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from itertools import product
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
            "save_run_logs": False,
            "lm_eval_tasks": ["lambada_openai", "hellaswag"],
            "lm_eval_limit": 256,
            "lm_eval_num_fewshot": 0,
        },
        "data": {
            "dataset": "iohadrubin/wikitext-103-raw-v1",
            "name": None,
            "data_files": None,
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
            "torch_compile": False,
            "torch_compile_mode": "default",
            "torch_compile_fullgraph": False,
            "torch_compile_dynamic": False,
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
        "sweep": {},
    }


class _TeeStream:
    def __init__(self, *streams: Any):
        self._streams = [s for s in streams if s is not None]

    def write(self, data: str) -> None:
        for s in self._streams:
            try:
                s.write(data)
            except Exception:
                pass
        if "\n" in data:
            self.flush()

    def flush(self) -> None:
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self) -> bool:  # pragma: no cover - passthrough helper
        for s in self._streams:
            fn = getattr(s, "isatty", None)
            if callable(fn):
                try:
                    if bool(fn()):
                        return True
                except Exception:
                    pass
        return False


@contextmanager
def _tee_run_logs(log_path: Path, *, enabled: bool) -> Iterable[None]:
    if not enabled:
        yield
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        orig_out = sys.stdout
        orig_err = sys.stderr
        sys.stdout = _TeeStream(orig_out, fh)  # type: ignore[assignment]
        sys.stderr = _TeeStream(orig_err, fh)  # type: ignore[assignment]
        try:
            yield
        finally:
            sys.stdout = orig_out  # type: ignore[assignment]
            sys.stderr = orig_err  # type: ignore[assignment]


def _coerce_betas(value: Any, default: tuple[float, float] = (0.9, 0.95)) -> tuple[float, float]:
    if value is None:
        return default
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        if len(parts) != 2:
            return default
        try:
            return (float(parts[0]), float(parts[1]))
        except Exception:
            return default
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return (float(value[0]), float(value[1]))
        except Exception:
            return default
    return default


def _coerce_ddp(value: Any, default: str = "off") -> str:
    # YAML 1.1 parses "on"/"off" as booleans; accept that and map to the expected strings.
    if value is None:
        return default
    if isinstance(value, bool):
        return "on" if value else "off"
    return str(value)


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
    data_files = data_cfg.get("data_files")
    if isinstance(data_files, str):
        df = data_files.strip()
        if "," in df:
            data_files = [s.strip() for s in df.split(",") if s.strip()]
        else:
            data_files = df
    if data_files:
        stream = load_dataset(
            data_cfg["dataset"],
            data_files=data_files,
            split=data_cfg.get("train_split", "train"),
            streaming=True,
            revision=data_cfg.get("revision"),
        )
    else:
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
        data_files=data_cfg.get("data_files"),
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


def _flatten_sweep_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a nested sweep spec into dotted-key -> values mappings."""

    out: Dict[str, Any] = {}

    def _walk(prefix: str, node: Any) -> None:
        if not isinstance(node, dict):
            if prefix:
                out[prefix] = node
            return
        for k, v in node.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                _walk(key, v)
            else:
                out[key] = v

    _walk("", spec)
    return out


def _set_by_dotted_path(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = [p for p in str(dotted).split(".") if p]
    if not parts:
        return
    cur: Dict[str, Any] = cfg
    for p in parts[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[p] = nxt
        cur = nxt
    cur[parts[-1]] = value


def _format_sweep_slug(overrides: Dict[str, Any]) -> str:
    if not overrides:
        return "default"
    abbrev = {
        "lr": "lr",
        "warmup_steps": "wu",
        "batch_tokens": "btok",
        "grad_accum_steps": "ga",
        "weight_decay": "wd",
        "amp": "amp",
        "grad_checkpoint": "gckpt",
        "d_model": "dm",
        "n_layers": "L",
        "n_heads": "H",
        "d_mlp": "mlp",
        "max_length": "T",
    }
    parts: List[str] = []
    for key in sorted(overrides.keys()):
        leaf = str(key).split(".")[-1]
        tag = abbrev.get(leaf, leaf)
        val = overrides[key]
        if isinstance(val, bool):
            sval = "t" if val else "f"
        elif isinstance(val, float):
            sval = f"{val:g}"
        else:
            sval = str(val)
        sval = sval.replace(" ", "")
        sval = re.sub(r"[^A-Za-z0-9._-]+", "", sval)
        parts.append(f"{tag}{sval}")
    slug = "_".join(parts)
    return (slug[:120] or "default") if slug else "default"


def _expand_sweep_configs(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return a list of sweep runs: [{id, slug, overrides, cfg}, ...]."""

    sweep = cfg.get("sweep") or {}
    if not sweep:
        return [
            {
                "id": 0,
                "slug": "default",
                "overrides": {},
                "cfg": cfg,
            }
        ]
    if not isinstance(sweep, dict):
        raise SystemExit("Config key 'sweep' must be a mapping.")

    flat = _flatten_sweep_spec(sweep)
    dims: List[tuple[str, List[Any]]] = []
    for key, values in sorted(flat.items(), key=lambda kv: str(kv[0])):
        if values is None:
            continue
        if isinstance(values, (list, tuple)):
            opts = list(values)
        else:
            opts = [values]
        if not opts:
            continue
        dims.append((str(key), opts))

    if not dims:
        return [
            {
                "id": 0,
                "slug": "default",
                "overrides": {},
                "cfg": cfg,
            }
        ]

    keys = [k for k, _ in dims]
    value_lists = [vals for _, vals in dims]
    expanded: List[Dict[str, Any]] = []
    for idx, combo in enumerate(product(*value_lists)):
        run_cfg = copy.deepcopy(cfg)
        run_cfg.pop("sweep", None)
        overrides: Dict[str, Any] = {}
        for key, val in zip(keys, combo):
            overrides[key] = val
            _set_by_dotted_path(run_cfg, key, val)
        expanded.append(
            {
                "id": idx,
                "slug": _format_sweep_slug(overrides),
                "overrides": overrides,
                "cfg": run_cfg,
            }
        )
    return expanded


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
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned sweep matrix and exit without running training.",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs that already have a metrics.json with status=ok.",
    )
    ap.add_argument(
        "--save-run-logs",
        action="store_true",
        help="Tee stdout/stderr to run_dir/stdout.log for each run.",
    )
    ap.add_argument(
        "--torch-compile",
        action="store_true",
        help="Enable torch.compile for training runs (single GPU only; skipped under DDP/FSDP).",
    )
    ap.add_argument(
        "--torch-compile-mode",
        type=str,
        default=None,
        choices=["default", "reduce-overhead", "max-autotune"],
        help="Optional torch.compile mode override.",
    )
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
    if args.save_run_logs:
        cfg["bench"]["save_run_logs"] = True
    if args.lm_eval_tasks:
        cfg["bench"]["lm_eval_tasks"] = [
            t.strip() for t in args.lm_eval_tasks.split(",") if t.strip()
        ]
    if args.lm_eval_limit is not None:
        cfg["bench"]["lm_eval_limit"] = int(args.lm_eval_limit)
    if args.seeds:
        cfg["bench"]["seeds"] = [int(s) for s in args.seeds.split(",") if s.strip()]
    if args.torch_compile:
        cfg["train"]["torch_compile"] = True
    if args.torch_compile_mode:
        cfg["train"]["torch_compile_mode"] = str(args.torch_compile_mode)

    bases = _parse_bases(args.bases, cfg)
    bench_cfg = cfg.get("bench", {})
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

    sweeps = _expand_sweep_configs(cfg)
    _write_json(
        outdir / "sweep_plan.json",
        {
            "count": len(sweeps),
            "bases": bases,
            "seeds": seeds,
            "sweeps": [
                {"id": int(s["id"]), "slug": str(s["slug"]), "overrides": s.get("overrides", {})}
                for s in sweeps
            ],
        },
    )

    if args.dry_run:
        print(f"[dry-run] outdir={outdir}")
        for s in sweeps:
            print(f"  sweep={int(s['id']):03d} slug={s['slug']} overrides={s.get('overrides', {})}")
            for base in bases:
                for seed in seeds:
                    print(f"    - base={base} seed={seed}")
        return 0

    results: List[Dict[str, Any]] = []
    tokenizers_by_sweep: Dict[str, Any] = {}
    runs_dir = outdir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    sweep_count = len(sweeps)
    for sweep in sweeps:
        sweep_id = int(sweep.get("id", 0))
        sweep_slug = str(sweep.get("slug", "default"))
        overrides = sweep.get("overrides", {}) or {}
        run_cfg: Dict[str, Any] = sweep["cfg"]

        bench_cfg_run = run_cfg.get("bench", {})
        data_cfg = run_cfg.get("data", {})
        train_cfg = run_cfg.get("train", {})
        eval_cfg = run_cfg.get("eval", {})

        sweep_root = runs_dir
        if sweep_count > 1:
            sweep_root = runs_dir / f"sweep{sweep_id:03d}_{sweep_slug}"
            sweep_root.mkdir(parents=True, exist_ok=True)
            (sweep_root / "config_resolved.yaml").write_text(
                yaml.safe_dump(run_cfg, sort_keys=False), encoding="utf-8"
            )

        tok_seed = int(seeds[0]) if seeds else 1337
        tokenizer, tokenizer_meta = _ensure_tokenizer(run_cfg, outdir=outdir, seed=tok_seed)
        tokenizers_by_sweep[str(sweep_id)] = tokenizer_meta
        tok_meta_path = (outdir / "tokenizer_meta.json") if sweep_count == 1 else (sweep_root / "tokenizer_meta.json")
        if sweep_count > 1 or not tok_meta_path.exists():
            _write_json(tok_meta_path, tokenizer_meta)

        for base in bases:
            for seed in seeds:
                run_dir = sweep_root / f"{base}_seed{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                metrics_path = run_dir / "metrics.json"
                if args.skip_existing and metrics_path.exists():
                    try:
                        cached = json.loads(metrics_path.read_text(encoding="utf-8"))
                        if isinstance(cached, dict) and cached.get("status") == "ok":
                            results.append(cached)
                            print(
                                f"[bench] skip existing base={base} seed={seed} sweep={sweep_id:03d}"
                            )
                            continue
                    except Exception:
                        pass

                print(f"[bench] base={base} seed={seed} sweep={sweep_id:03d} -> {run_dir}")
                torch.manual_seed(int(seed))
                random.seed(int(seed))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(seed))

                record: Dict[str, Any] = {
                    "base": base,
                    "seed": int(seed),
                    "sweep_id": sweep_id,
                    "sweep_slug": sweep_slug,
                    "sweep_overrides": overrides,
                    "status": "ok",
                }

                save_logs = bool(bench_cfg_run.get("save_run_logs", False))
                run_log_path = run_dir / "stdout.log"
                record["run_log_path"] = str(run_log_path) if save_logs else None

                with _tee_run_logs(run_log_path, enabled=save_logs):
                    if save_logs:
                        print(f"[bench] logging stdout/stderr -> {run_log_path}", flush=True)
                        print(
                            f"[bench] sweep={sweep_id:03d} slug={sweep_slug} base={base} seed={seed}",
                            flush=True,
                        )

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
                                amp_init_std=float(sine_cfg.get("amp_init_std", 0.0)),
                                freq_init=float(sine_cfg.get("freq_init", 1.0)),
                                freq_init_std=float(sine_cfg.get("freq_init_std", 0.0)),
                                damp_init=float(sine_cfg.get("damp_init", 0.01)),
                                damp_init_std=float(sine_cfg.get("damp_init_std", 0.0)),
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
                        record["train_config"] = {
                            "batch_tokens": int(batch_tokens),
                            "grad_accum_steps": int(grad_accum),
                            "lr": float(train_cfg.get("lr", 2e-4)),
                            "warmup_steps": int(train_cfg.get("warmup_steps", 200)),
                            "weight_decay": float(train_cfg.get("weight_decay", 0.01)),
                            "optimizer": str(train_cfg.get("optimizer", "adamw")),
                            "betas": list(_coerce_betas(train_cfg.get("betas"))),
                            "eps": float(train_cfg.get("eps", 1e-8)),
                            "label_smoothing": float(train_cfg.get("label_smoothing", 0.0)),
                            "grad_clip": float(train_cfg.get("grad_clip", 1.0)),
                            "amp": str(train_cfg.get("amp", "bf16")),
                            "grad_checkpoint": bool(train_cfg.get("grad_checkpoint", False)),
                            "log_gpu_mem": bool(train_cfg.get("log_gpu_mem", False)),
                            "ddp": _coerce_ddp(train_cfg.get("ddp"), default="off"),
                            "torch_compile": bool(train_cfg.get("torch_compile", False)),
                            "torch_compile_mode": str(train_cfg.get("torch_compile_mode", "default")),
                            "torch_compile_fullgraph": bool(train_cfg.get("torch_compile_fullgraph", False)),
                            "torch_compile_dynamic": bool(train_cfg.get("torch_compile_dynamic", False)),
                            "log_interval_steps": int(train_cfg.get("log_interval_steps", 50)),
                            "save_interval_steps": int(train_cfg.get("save_interval_steps", 500)),
                            "max_steps": int(max_steps),
                            "tokens_target": int(tokens_target),
                            "tokens_per_step": int(tokens_per_step),
                        }
                        record["data_config"] = {
                            "dataset": str(data_cfg.get("dataset")),
                            "name": data_cfg.get("name"),
                            "data_files": data_cfg.get("data_files"),
                            "train_split": str(data_cfg.get("train_split", "train")),
                            "val_split": str(data_cfg.get("val_split", "validation")),
                            "text_key": str(data_cfg.get("text_key", "text")),
                            "max_length": int(seq_len),
                            "shuffle": bool(data_cfg.get("shuffle", True)),
                            "shuffle_buffer": int(data_cfg.get("shuffle_buffer", 10000)),
                        }

                        tcfg = TrainConfig(
                            epochs=1,
                            batch_tokens=int(batch_tokens),
                            lr=float(train_cfg.get("lr", 2e-4)),
                            warmup_steps=int(train_cfg.get("warmup_steps", 200)),
                            weight_decay=float(train_cfg.get("weight_decay", 0.01)),
                            amp=str(train_cfg.get("amp", "bf16")),
                            optimizer=str(train_cfg.get("optimizer", "adamw")),
                            betas=_coerce_betas(train_cfg.get("betas")),
                            eps=float(train_cfg.get("eps", 1e-8)),
                            label_smoothing=float(train_cfg.get("label_smoothing", 0.0)),
                            grad_clip=float(train_cfg.get("grad_clip", 1.0)),
                            grad_checkpoint=bool(train_cfg.get("grad_checkpoint", False)),
                            log_gpu_mem=bool(train_cfg.get("log_gpu_mem", False)),
                            torch_compile=bool(train_cfg.get("torch_compile", False)),
                            torch_compile_mode=str(train_cfg.get("torch_compile_mode", "default")),
                            torch_compile_fullgraph=bool(train_cfg.get("torch_compile_fullgraph", False)),
                            torch_compile_dynamic=bool(train_cfg.get("torch_compile_dynamic", False)),
                            grad_accum_steps=int(grad_accum),
                            ddp=_coerce_ddp(train_cfg.get("ddp"), default="off"),
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
                        record["peak_cuda_mem_gb"] = (
                            None if peak_mem is None else round(peak_mem, 4)
                        )
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

                        if bench_cfg_run.get("with_lm_eval", False):
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
                                    tasks=[
                                        str(t) for t in bench_cfg_run.get("lm_eval_tasks", [])
                                    ],
                                    limit=int(bench_cfg_run.get("lm_eval_limit", 256)),
                                    num_fewshot=int(
                                        bench_cfg_run.get("lm_eval_num_fewshot", 0)
                                    ),
                                    device="cuda" if torch.cuda.is_available() else "cpu",
                                )
                                record["lm_eval"] = lm_eval

                    except Exception as exc:
                        record["status"] = "error"
                        record["error"] = str(exc)
                        print(f"[bench] ERROR: {exc}", file=sys.stderr, flush=True)

                results.append(record)
                _write_json(metrics_path, record)

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
        "sweeps": [
            {"id": int(s["id"]), "slug": str(s["slug"]), "overrides": s.get("overrides", {})}
            for s in sweeps
        ],
        "tokenizers_by_sweep": tokenizers_by_sweep,
        "results": results,
    }
    _write_json(outdir / "summary.json", summary)

    # Write summary.csv
    sweep_keys = sorted(
        {k for s in sweeps for k in (s.get("overrides", {}) or {}).keys()}
    )
    sweep_cols = [k.replace(".", "__") for k in sweep_keys]
    header = [
        "base",
        "seed",
        "sweep_id",
        "sweep_slug",
        *sweep_cols,
        "status",
        "param_count",
        "train_steps",
        "train_tokens",
        "train_tokens_per_s",
        "train_elapsed_s",
        "peak_cuda_mem_gb",
        "val_loss",
        "val_ppl",
        "val_top1_acc",
    ]
    csv_rows = [",".join(header)]
    for row in results:
        sweep_vals = row.get("sweep_overrides", {}) if isinstance(row, dict) else {}
        csv_rows.append(
            ",".join(
                [
                    str(row.get("base", "")),
                    str(row.get("seed", "")),
                    str(row.get("sweep_id", "")),
                    str(row.get("sweep_slug", "")),
                    *[str(sweep_vals.get(k, "")) for k in sweep_keys],
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
        "| sweep | base | seed | val_ppl | val_loss | val_top1_acc | tokens/s | peak_cuda_mem_gb |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in ok_rows:
        lines.append(
            "| {sweep} | {base} | {seed} | {val_ppl} | {val_loss} | {val_top1_acc} | {train_tokens_per_s} | {peak_cuda_mem_gb} |".format(
                sweep=r.get("sweep_slug", ""),
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
