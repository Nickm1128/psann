"""
Unified GPU validation and benchmarking runner for PSANN-LM.

This script executes the GPU work block in one go and stores
results in a timestamped reports directory that is easy to commit.

Outputs written to: reports/gpu/<UTC_yyyymmdd_HHMMSS>/

Contents:
- summary.json: structured results and metrics
- system.json: system information (Torch/CUDA/GPU)
- stdout.log: human-readable log stream
- checkpoints/ (optional): saved model(s)

Usage (from repo root):
  python scripts/run_gpu_validation.py --out reports/gpu

If not installed as editable, either install `-e .[lm]` or set PYTHONPATH=.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import torch

try:
    from psann.lm import psannLM, psannLMDataPrep
except Exception as e:  # pragma: no cover - runner convenience
    print("Failed to import psann.lm — ensure PYTHONPATH=.<repo root> or install -e .", file=sys.stderr)
    raise


def _now_utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def system_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version.split(" ")[0],
        "platform": sys.platform,
        "torch": torch.__version__,
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "bf16_supported": getattr(torch.cuda, "is_bf16_supported", lambda: False)(),
    }
    if torch.cuda.is_available():
        gpus: List[Dict[str, Any]] = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append(
                {
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                    "major": props.major,
                    "minor": props.minor,
                }
            )
        info["gpus"] = gpus
    return info


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def gpu_01_forward_backward() -> Dict[str, Any]:
    texts = ["hello world", "goodnight moon", "abc def ghi", "lorem ipsum"]
    # Use a small max_length to ensure at least one chunk from tiny texts
    dp = psannLMDataPrep(texts, tokenizer="simple", max_length=16, pack_sequences=True, val_split=0.0)
    lm = psannLM(base="waveresnet", d_model=128, n_layers=2, n_heads=4, vocab_size=dp.vocab_size, rope=True)
    t0 = time.time()
    lm.fit(dp, epochs=1, batch_tokens=4096, lr=3e-4)
    dt = time.time() - t0
    return {
        "status": "ok",
        "elapsed_s": round(dt, 4),
        "vocab_size": dp.vocab_size,
        "model": {
            "base": lm.base,
            "d_model": lm.d_model,
            "n_layers": lm.n_layers,
            "n_heads": lm.n_heads,
        },
    }


def gpu_02_amp_parity() -> Dict[str, Any]:
    # single-step loss compare between fp32 and autocast (fp16 or bf16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        return {"status": "skipped", "reason": "cuda not available"}

    texts = ["a b c d e f g", "h i j k l m", "n o p q r s"]
    dp = psannLMDataPrep(texts, tokenizer="simple", max_length=64, pack_sequences=True, val_split=0.0)
    lm = psannLM(base="respsann", d_model=128, n_layers=2, n_heads=4, vocab_size=dp.vocab_size, rope=True)

    # Build a single batch tensor
    tok = dp.tokenizer  # type: ignore[attr-defined]
    seq = torch.tensor([[tok.bos_id] + tok.encode("hello world", add_specials=False) + [tok.eos_id]], dtype=torch.long, device=device)
    model = lm._ensure_model(dp.vocab_size).to(device)
    criterion = torch.nn.CrossEntropyLoss()

    # FP32
    model.zero_grad(set_to_none=True)
    logits = model(seq)
    B, T, V = logits.shape
    loss32 = criterion(logits.view(B * T, V), seq.view(B * T)).detach().float().item()

    # AMP (prefer bf16 if supported)
    amp_dtype = torch.bfloat16 if getattr(torch.cuda, "is_bf16_supported", lambda: False)() else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    model.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(dtype=amp_dtype):
        logits_amp = model(seq)
        loss_amp = criterion(logits_amp.view(B * T, V), seq.view(B * T))
    scaler.scale(loss_amp).backward()
    scaler.step(torch.optim.SGD(model.parameters(), lr=0.0))  # no-op step
    scaler.update()
    loss_amp_val = float(loss_amp.detach().float().item())

    rel = abs(loss_amp_val - loss32) / max(1e-8, abs(loss32))
    return {
        "status": "ok",
        "fp32_loss": round(loss32, 6),
        "amp_loss": round(loss_amp_val, 6),
        "amp_dtype": str(amp_dtype),
        "rel_diff": round(rel, 6),
    }


def gpu_03_throughput() -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        return {"status": "skipped", "reason": "cuda not available"}

    def bench(base: str) -> Dict[str, Any]:
        vocab, T, B, steps = 32000, 256, 4, 20
        lm = psannLM(base=base, d_model=256, n_layers=4, n_heads=4, vocab_size=vocab, rope=True)
        model = lm._ensure_model(vocab).to(device).eval()
        torch.cuda.synchronize()
        x = torch.randint(0, vocab, (B, T), device=device, dtype=torch.long)
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(x)
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            for _ in range(steps):
                _ = model(x)
        torch.cuda.synchronize()
        dt = time.time() - t0
        tokens = B * T * steps
        return {
            "tokens": tokens,
            "elapsed_s": round(dt, 4),
            "tokens_per_s": round(tokens / dt, 2),
        }

    return {
        "status": "ok",
        "respsann": bench("respsann"),
        "waveresnet": bench("waveresnet"),
    }


def gpu_04_checkpointing() -> Dict[str, Any]:
    """Smoke-test training with gradient checkpointing enabled.

    Runs a tiny single-epoch fit with grad checkpointing=True to ensure
    the path is wired end-to-end. Returns basic timing metadata.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        return {"status": "skipped", "reason": "cuda not available"}

    texts = [
        "hello world",
        "goodnight moon",
        "abc def ghi",
        "lorem ipsum",
        "pack my box with five dozen liquor jugs",
    ]
    dp = psannLMDataPrep(texts, tokenizer="simple", max_length=64, pack_sequences=True, val_split=0.0)
    lm = psannLM(base="waveresnet", d_model=128, n_layers=2, n_heads=4, vocab_size=dp.vocab_size, rope=True)
    t0 = time.time()
    # Enable gradient checkpointing via Trainer config
    lm.fit(dp, epochs=1, batch_tokens=4096, lr=3e-4, grad_checkpoint=True)
    dt = time.time() - t0
    return {
        "status": "ok",
        "grad_checkpoint": True,
        "elapsed_s": round(dt, 4),
        "vocab_size": dp.vocab_size,
        "model": {
            "base": lm.base,
            "d_model": lm.d_model,
            "n_layers": lm.n_layers,
            "n_heads": lm.n_heads,
        },
    }


def gpu_05_ddp() -> Dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return {"status": "skipped", "reason": "requires >=2 CUDA devices"}
    return {"status": "skipped", "reason": "DDP integration not implemented"}


def gpu_06_zerofsdp() -> Dict[str, Any]:
    return {"status": "skipped", "reason": "DeepSpeed/FSDP hooks not implemented"}


def gpu_07_generation_smoke() -> Dict[str, Any]:
    texts = ["pack my box with five dozen liquor jugs", "sphinx of black quartz judge my vow"]
    dp = psannLMDataPrep(texts, tokenizer="simple", max_length=64, pack_sequences=True, val_split=0.0)
    lm = psannLM(base="waveresnet", d_model=128, n_layers=2, n_heads=4, vocab_size=dp.vocab_size, rope=True)
    lm.fit(dp, epochs=1, batch_tokens=4096, lr=3e-4)
    out = lm.generate("Once upon a time", max_new_tokens=24, top_p=0.9, temperature=0.9)
    return {
        "status": "ok",
        "sample": out,
        "length": len(out),
    }


def gpu_08_save_load(outdir: Path) -> Dict[str, Any]:
    texts = ["hello world", "goodnight moon", "abc def ghi", "lorem ipsum"]
    # Use a small max_length to ensure dataset has samples for tiny texts
    dp = psannLMDataPrep(texts, tokenizer="simple", max_length=16, pack_sequences=True, val_split=0.0)
    lm = psannLM(base="respsann", d_model=128, n_layers=2, n_heads=4, vocab_size=dp.vocab_size, rope=True)
    lm.fit(dp, epochs=1, batch_tokens=4096, lr=3e-4)

    ckpt_dir = outdir / "checkpoints"
    _ensure_dir(ckpt_dir)
    ckpt_path = ckpt_dir / "lm.pt"
    lm.save(str(ckpt_path))
    loaded = psannLM.load(str(ckpt_path))

    # Attach tokenizer for convenience
    loaded._tokenizer = dp.tokenizer  # type: ignore[attr-defined]

    # Compare parameter tensors equality
    same = True
    if lm._model is None or loaded._model is None:
        same = False
    else:
        for (n1, p1), (n2, p2) in zip(lm._model.state_dict().items(), loaded._model.state_dict().items()):
            if n1 != n2:
                same = False
                break
            if not torch.allclose(p1.detach().cpu(), p2.detach().cpu()):
                same = False
                break

    # Seed determinism on same device
    torch.manual_seed(123)
    out1 = lm.generate("The quick brown fox", max_new_tokens=12, top_p=0.9, temperature=0.8)
    torch.manual_seed(123)
    out2 = loaded.generate("The quick brown fox", max_new_tokens=12, top_p=0.9, temperature=0.8)

    return {
        "status": "ok",
        "params_equal": bool(same),
        "gen_equal": out1 == out2,
        "ckpt_path": str(ckpt_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="reports/gpu", help="Base directory for reports")
    args = ap.parse_args()

    base = Path(args.out).resolve()
    tag = _now_utc_tag()
    outdir = base / tag
    _ensure_dir(outdir)

    # tee stdout
    log_fp = (outdir / "stdout.log").open("w", encoding="utf-8")

    def _log(msg: str) -> None:
        print(msg)
        print(msg, file=log_fp)
        log_fp.flush()

    _log(f"[INFO] Writing GPU report to: {outdir}")
    _log("[INFO] Collecting system info …")
    sysinfo = system_info()
    _write_json(outdir / "system.json", sysinfo)
    _log(json.dumps(sysinfo, indent=2))

    summary: Dict[str, Any] = {"timestamp_utc": tag, "system": sysinfo, "results": {}}

    steps = [
        ("GPU-01", gpu_01_forward_backward),
        ("GPU-02", gpu_02_amp_parity),
        ("GPU-03", gpu_03_throughput),
        ("GPU-04", gpu_04_checkpointing),
        ("GPU-05", gpu_05_ddp),
        ("GPU-06", gpu_06_zerofsdp),
        ("GPU-07", gpu_07_generation_smoke),
    ]

    for name, fn in steps:
        _log(f"[RUN] {name}")
        try:
            res = fn()
            summary["results"][name] = res
            _log(f"[OK ] {name}: {res}")
        except Exception as e:  # pragma: no cover - runtime safety
            summary["results"][name] = {"status": "error", "error": str(e)}
            _log(f"[ERR] {name}: {e}")
        _write_json(outdir / "summary.json", summary)

    # GPU-08 depends on output dir
    _log("[RUN] GPU-08")
    try:
        res8 = gpu_08_save_load(outdir)
        summary["results"]["GPU-08"] = res8
        _log(f"[OK ] GPU-08: {res8}")
    except Exception as e:  # pragma: no cover
        summary["results"]["GPU-08"] = {"status": "error", "error": str(e)}
        _log(f"[ERR] GPU-08: {e}")
    _write_json(outdir / "summary.json", summary)

    _log("[DONE] GPU validation complete.")
    log_fp.close()


if __name__ == "__main__":
    main()
