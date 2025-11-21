"""Finalize BMRK-01 tiny-corpus benchmark artifacts.

Tasks:
- Optionally parse training log to metrics.csv and loss_curve.png (if matplotlib present)
- Compute validation loss/perplexity using the saved model and config
- Write metrics.json summarizing train/val metrics and run metadata

Usage:
  python scripts/finalize_bmrk01.py \
    --config examples/lm/configs/tiny_corpus_benchmark.yaml \
    --bench-dir reports/benchmarks/<timestamp> \
    --log reports/benchmarks/<timestamp>/tiny_benchmark.log \
    --out reports/benchmarks/<timestamp>/metrics.json \
    --plot
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml


def _parse_last_metrics(csv_path: Path) -> Dict[str, float]:
    last: Optional[Dict[str, float]] = None
    if not csv_path.exists():
        return {"train_loss": float("nan"), "train_ppl": float("nan")}
    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                last = {
                    "train_loss": float(row.get("loss", "nan")),
                    "train_ppl": float(row.get("ppl", "nan")),
                }
            except Exception:
                continue
    return last or {"train_loss": float("nan"), "train_ppl": float("nan")}


def _build_dataprep(cfg: Dict[str, Any]):
    from psann.lm import psannLMDataPrep

    data_cfg = cfg.get("data", {})
    sources = []
    if isinstance(data_cfg.get("sources"), list):
        for ent in data_cfg["sources"]:
            if isinstance(ent, dict) and "path" in ent:
                sources.append(str(ent["path"]))
            elif isinstance(ent, str):
                sources.append(ent)
    dp = psannLMDataPrep(
        sources,
        tokenizer=str(data_cfg.get("tokenizer", "auto")),
        tokenizer_model_path=(
            str(data_cfg.get("tokenizer_model_path"))
            if data_cfg.get("tokenizer_model_path") is not None
            else None
        ),
        max_length=int(data_cfg.get("max_length", 512)),
        pack_sequences=bool(data_cfg.get("pack_sequences", True)),
        val_split=(
            float(data_cfg.get("val_split", 0.02)) if data_cfg.get("val_split") is not None else 0.0
        ),
        seed=int(data_cfg.get("seed", 1337)),
    )
    return dp


def _evaluate_validation(cfg: Dict[str, Any], bench_dir: Path, dp) -> Dict[str, float]:
    from psann.lm import psannLM
    from psann.lm.data.dataset import collate_batch

    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})

    ckpt_dir = Path(train_cfg.get("checkpoint_dir", "runs/lm/bmrk_tiny"))
    ckpt_path = ckpt_dir / "final_model.pt"
    if not ckpt_path.exists():
        # Try under bench dir if model saved there
        alt = bench_dir / "final_model.pt"
        ckpt_path = alt if alt.exists() else ckpt_path

    lm = psannLM.load(str(ckpt_path))
    vocab = int((model_cfg.get("vocab_size") or dp.vocab_size))
    model = lm._ensure_model(vocab)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # Batch size selection similar to Trainer
    batch_tokens = int(train_cfg.get("batch_tokens", 32768))
    max_length = int(cfg.get("data", {}).get("max_length", 512))
    batch_size = max(1, batch_tokens // max_length)

    # AMP selection mirroring config
    amp_mode = str(train_cfg.get("amp", "bf16")).lower()
    use_amp = device.type == "cuda" and amp_mode in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if amp_mode == "bf16" else torch.float16
    autocast_ctx = torch.amp.autocast("cuda", dtype=amp_dtype) if use_amp else torch.no_grad()

    val_ds = dp.val_dataset
    if val_ds is None:
        return {"val_loss": float("nan"), "val_ppl": float("nan")}

    import math
    from torch.nn import functional as F
    from torch.utils.data import DataLoader

    # Fixed-length examples; simple stack collation
    dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in dl:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            with autocast_ctx:
                logits = model(input_ids)
                B, T, V = logits.shape
                loss = F.cross_entropy(logits.view(B * T, V), labels.view(B * T), reduction="sum")
            total_loss += float(loss.detach().cpu().item())
            total_tokens += int(B * T)
    if total_tokens == 0:
        return {"val_loss": float("nan"), "val_ppl": float("nan")}
    val_loss = total_loss / float(total_tokens)
    val_ppl = math.exp(val_loss)
    return {"val_loss": round(val_loss, 6), "val_ppl": round(val_ppl, 6)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--bench-dir", type=str, required=True)
    ap.add_argument("--log", type=str, default=None, help="Path to training log (tee output)")
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Path to metrics.json (default: <bench-dir>/metrics.json)",
    )
    ap.add_argument(
        "--plot", action="store_true", help="Generate loss_curve.png if matplotlib available"
    )
    args = ap.parse_args()

    bench_dir = Path(args.bench_dir)
    bench_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = bench_dir / "metrics.csv"
    metrics_json = Path(args.out) if args.out else (bench_dir / "metrics.json")

    # Optional: parse log to metrics.csv (+plot)
    if args.log:
        parse_cmd = [
            "python",
            "scripts/parse_trainer_log.py",
            "--log",
            str(args.log),
            "--out",
            str(bench_dir),
        ]
        if args.plot:
            parse_cmd.append("--plot")
        try:
            import subprocess

            subprocess.run(parse_cmd, check=False)
        except Exception:
            pass

    train_summary = _parse_last_metrics(metrics_csv)

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    dp = _build_dataprep(cfg)
    val_summary = _evaluate_validation(cfg, bench_dir, dp)

    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})

    payload: Dict[str, Any] = {
        "train_loss_final": train_summary.get("train_loss"),
        "train_ppl_final": train_summary.get("train_ppl"),
        "val_loss": val_summary.get("val_loss"),
        "val_ppl": val_summary.get("val_ppl"),
        "corpus_path": [
            ent.get("path") if isinstance(ent, dict) else ent
            for ent in (data_cfg.get("sources") or [])
        ],
        "model": {
            "base": model_cfg.get("base"),
            "d_model": model_cfg.get("d_model"),
            "n_layers": model_cfg.get("n_layers"),
            "n_heads": model_cfg.get("n_heads"),
            "vocab_size": dp.vocab_size,
            "rope": model_cfg.get("rope", True),
        },
        "train": {
            "batch_tokens": train_cfg.get("batch_tokens"),
            "amp": train_cfg.get("amp", "bf16"),
            "grad_checkpoint": bool(train_cfg.get("grad_checkpoint", False)),
            "epochs": train_cfg.get("epochs", 1),
        },
    }
    metrics_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
