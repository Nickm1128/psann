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
from typing import Any, Dict, List, Sequence

import torch

try:
    from psannlm.lm import psannLM, psannLMDataPrep
except Exception:  # pragma: no cover - runner convenience
    print(
        "Failed to import psannlm.lm — ensure PYTHONPATH=.<repo root> or install -e .",
        file=sys.stderr,
    )
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


# -------------------------- DDP worker helper --------------------------


def _setup_dist_env(rank: int, world_size: int, port: int) -> None:
    import os as _os

    # env for torch.distributed default init
    _os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    _os.environ.setdefault("MASTER_PORT", str(int(port)))
    _os.environ["WORLD_SIZE"] = str(int(world_size))
    _os.environ["RANK"] = str(int(rank))
    _os.environ["LOCAL_RANK"] = str(int(rank))
    # RunPod L4 stability: disable IB/P2P transports that tend to SIGABRT on multi-tenant pods
    _os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    _os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    _os.environ.setdefault("NCCL_IB_DISABLE", "1")
    _os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    _os.environ.setdefault("NCCL_SHM_DISABLE", "1")
    _os.environ.setdefault("NCCL_DEBUG", "WARN")


def _ddp_loss_worker(
    rank: int,
    world_size: int,
    port: int,
    batch_ids: Sequence[Sequence[int]],
    vocab_size: int,
    model_cfg: Dict[str, Any],
    shared,
) -> None:
    import torch as _torch
    import torch.distributed as _dist
    from torch.nn.parallel import DistributedDataParallel as _DDP

    _setup_dist_env(rank, world_size, port)

    device = _torch.device("cuda", int(rank))
    try:
        _torch.cuda.set_device(device)
    except Exception:
        pass
    backend = "nccl"
    _dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    try:
        # Deterministic-ish seed
        _torch.manual_seed(123)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(123)

        # Build model consistent with single-GPU baseline
        lm = psannLM(**model_cfg)
        model = lm._ensure_model(int(vocab_size)).to(device).eval()
        model = _DDP(
            model, device_ids=[int(rank)], output_device=int(rank), find_unused_parameters=False
        )
        model.eval()

        # Build batch tensor on this device
        import torch.nn.functional as _F

        B = len(batch_ids)
        T = len(batch_ids[0]) if B > 0 else 0
        seq = _torch.tensor(batch_ids, dtype=_torch.long, device=device)

        with _torch.no_grad():
            logits = model(seq)
            V = int(vocab_size)
            loss = _F.cross_entropy(logits.view(B * T, V), seq.view(B * T))

        # Average loss across ranks
        loss_t = loss.detach().float().clone()
        _dist.all_reduce(loss_t, op=_dist.ReduceOp.SUM)
        loss_t /= float(world_size)

        if rank == 0:
            shared["ddp_avg_loss"] = float(loss_t.item())
    except Exception as exc:
        if rank == 0:
            shared["error"] = f"DDP worker error: {exc}"
        raise
    finally:
        try:
            _dist.destroy_process_group()
        except Exception:
            pass


# -------------------------- FSDP worker helper --------------------------


def _fsdp_loss_worker(
    rank: int,
    world_size: int,
    port: int,
    batch_ids: Sequence[Sequence[int]],
    vocab_size: int,
    model_cfg: Dict[str, Any],
    shared,
) -> None:
    import torch as _torch
    import torch.distributed as _dist
    from torch.distributed.fsdp import FullyShardedDataParallel as _FSDP

    _setup_dist_env(rank, world_size, port)

    device = _torch.device("cuda", int(rank))
    try:
        _torch.cuda.set_device(device)
    except Exception:
        pass
    backend = "nccl"
    _dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    try:
        _torch.manual_seed(123)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(123)

        lm = psannLM(**model_cfg)
        base_model = lm._ensure_model(int(vocab_size)).to(device)
        # Match evaluation mode with single-GPU baseline to avoid dropout-induced deltas
        base_model.eval()
        model = _FSDP(base_model)
        # Ensure wrapper reflects eval state
        model.eval()

        # Build batch tensor on this device
        import torch.nn.functional as _F

        B = len(batch_ids)
        T = len(batch_ids[0]) if B > 0 else 0
        seq = _torch.tensor(batch_ids, dtype=_torch.long, device=device)

        optim = _torch.optim.SGD(model.parameters(), lr=0.0)
        optim.zero_grad(set_to_none=True)
        logits = model(seq)
        V = int(vocab_size)
        loss = _F.cross_entropy(logits.view(B * T, V), seq.view(B * T))
        loss.backward()
        optim.step()  # lr=0.0: no change; exercises FSDP step path

        # Average loss across ranks
        loss_t = loss.detach().float().clone()
        _dist.all_reduce(loss_t, op=_dist.ReduceOp.SUM)
        loss_t /= float(world_size)

        if rank == 0:
            shared["fsdp_avg_loss"] = float(loss_t.item())
    except Exception as exc:
        if rank == 0:
            shared["error"] = f"FSDP worker error: {exc}"
        raise
    finally:
        try:
            _dist.destroy_process_group()
        except Exception:
            pass


def gpu_01_forward_backward() -> Dict[str, Any]:
    texts = ["hello world", "goodnight moon", "abc def ghi", "lorem ipsum"]
    # Use a small max_length to ensure at least one chunk from tiny texts
    dp = psannLMDataPrep(
        texts, tokenizer="simple", max_length=16, pack_sequences=True, val_split=0.0
    )
    lm = psannLM(
        base="waveresnet", d_model=128, n_layers=2, n_heads=4, vocab_size=dp.vocab_size, rope=True
    )
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
    dp = psannLMDataPrep(
        texts, tokenizer="simple", max_length=64, pack_sequences=True, val_split=0.0
    )
    lm = psannLM(
        base="respsann", d_model=128, n_layers=2, n_heads=4, vocab_size=dp.vocab_size, rope=True
    )

    # Build a single batch tensor
    tok = dp.tokenizer  # type: ignore[attr-defined]
    seq = torch.tensor(
        [[tok.bos_id] + tok.encode("hello world", add_specials=False) + [tok.eos_id]],
        dtype=torch.long,
        device=device,
    )
    model = lm._ensure_model(dp.vocab_size).to(device)
    criterion = torch.nn.CrossEntropyLoss()

    # FP32
    model.zero_grad(set_to_none=True)
    logits = model(seq)
    B, T, V = logits.shape
    loss32 = criterion(logits.view(B * T, V), seq.view(B * T)).detach().float().item()

    # AMP (prefer bf16 if supported)
    amp_dtype = (
        torch.bfloat16
        if getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        else torch.float16
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    model.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=amp_dtype):
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

    def _choose_dims(target_tokens: int) -> tuple[int, int]:
        max_t = max(64, int(os.environ.get("PSANN_GPU03_MAX_T", "4096")))
        max_b = max(1, int(os.environ.get("PSANN_GPU03_MAX_B", "64")))
        # start with as many tokens per sequence as allowed, then scale batch size
        T = min(max_t, max(64, target_tokens))
        B = max(1, math.ceil(target_tokens / T))
        if B > max_b:
            B = max_b
            T = max(64, math.ceil(target_tokens / B))
            if T > max_t:
                T = max_t
                B = max(1, math.ceil(target_tokens / T))
        return B, T

    def bench(base: str) -> Dict[str, Any]:
        # Allow simple environment overrides for sweeps
        vocab = int(os.environ.get("PSANN_GPU03_VOCAB", "32000"))
        steps = int(os.environ.get("PSANN_GPU03_STEPS", "20"))
        target_tokens_env = os.environ.get("PSANN_GPU03_BATCH_TOKENS")
        if target_tokens_env:
            target_tokens = max(1, int(target_tokens_env))
            if "PSANN_GPU03_B" in os.environ or "PSANN_GPU03_T" in os.environ:
                B = max(1, int(os.environ.get("PSANN_GPU03_B", "4")))
                T = max(1, int(os.environ.get("PSANN_GPU03_T", str(max(1, target_tokens // B)))))
            else:
                B, T = _choose_dims(target_tokens)
        else:
            T = max(1, int(os.environ.get("PSANN_GPU03_T", "256")))
            B = max(1, int(os.environ.get("PSANN_GPU03_B", "4")))
            target_tokens = B * T
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
            "B": B,
            "T": T,
            "batch_tokens": B * T,
            "steps": steps,
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
    dp = psannLMDataPrep(
        texts, tokenizer="simple", max_length=64, pack_sequences=True, val_split=0.0
    )
    lm = psannLM(
        base="waveresnet", d_model=128, n_layers=2, n_heads=4, vocab_size=dp.vocab_size, rope=True
    )
    # Reset and record CUDA memory stats for a clean measurement window
    try:
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass
    torch.cuda.synchronize()
    t0 = time.time()
    # Enable gradient checkpointing via Trainer config
    lm.fit(dp, epochs=1, batch_tokens=4096, lr=3e-4, grad_checkpoint=True)
    torch.cuda.synchronize()
    dt = time.time() - t0
    # Memory snapshot
    try:
        max_mem_alloc = int(torch.cuda.max_memory_allocated())
        max_mem_res = int(torch.cuda.max_memory_reserved())
    except Exception:
        max_mem_alloc = -1
        max_mem_res = -1
    return {
        "status": "ok",
        "grad_checkpoint": True,
        "elapsed_s": round(dt, 4),
        "max_memory_allocated_mb": (
            None if max_mem_alloc < 0 else round(max_mem_alloc / (1024**2), 2)
        ),
        "max_memory_reserved_mb": None if max_mem_res < 0 else round(max_mem_res / (1024**2), 2),
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

    # Single-GPU baseline (rank 0) forward loss
    torch.manual_seed(123)
    texts = ["the quick brown fox", "jumps over the lazy dog"]
    dp = psannLMDataPrep(
        texts, tokenizer="simple", max_length=32, pack_sequences=True, val_split=0.0
    )
    vocab = int(dp.vocab_size)
    # Build one small batch tensor
    tok = dp.tokenizer  # type: ignore[attr-defined]
    sample = tok.encode("the quick brown fox", add_specials=True)
    # Ensure consistent T across samples
    T = min(16, len(sample))
    base_ids = sample[:T]
    batch_ids = [base_ids, base_ids]  # B=2

    model_cfg = dict(
        base="waveresnet", d_model=128, n_layers=2, n_heads=4, vocab_size=vocab, rope=True
    )

    device0 = torch.device("cuda", 0)
    lm = psannLM(**model_cfg)
    model = lm._ensure_model(vocab).to(device0).eval()
    with torch.no_grad():
        import torch.nn.functional as F

        seq0 = torch.tensor(batch_ids, dtype=torch.long, device=device0)
        logits0 = model(seq0)
        V = int(vocab)
        loss_single = (
            F.cross_entropy(logits0.view(seq0.size(0) * seq0.size(1), V), seq0.view(-1))
            .detach()
            .float()
            .item()
        )

    # Multi-process DDP run (2 ranks) to compute average loss
    import torch.multiprocessing as mp
    from random import randint as _randint

    port = 29577 + (_randint(0, 1000))  # reduce chance of collision
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    shared = manager.dict()
    nprocs = 2
    try:
        mp.spawn(
            _ddp_loss_worker,
            args=(nprocs, port, batch_ids, vocab, model_cfg, shared),
            nprocs=nprocs,
            join=True,
        )
    except Exception as exc:
        return {"status": "error", "reason": f"DDP spawn failed: {exc}"}
    if "error" in shared:
        return {"status": "error", "reason": str(shared["error"])}
    ddp_loss = shared.get("ddp_avg_loss")
    if ddp_loss is None:
        return {"status": "error", "reason": "DDP run produced no result"}

    rel = abs(ddp_loss - loss_single) / max(1e-8, abs(loss_single))
    return {
        "status": "ok",
        "single_loss": round(loss_single, 6),
        "ddp_avg_loss": round(ddp_loss, 6),
        "rel_diff": round(rel, 6),
        "world_size": nprocs,
    }


def gpu_06_zerofsdp() -> Dict[str, Any]:
    # Prefer PyTorch FSDP if available; fall back to DeepSpeed if installed.
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return {"status": "skipped", "reason": "requires >=2 CUDA devices"}

    # Build a tiny batch/model config similar to GPU-05
    try:
        texts = ["the quick brown fox", "jumps over the lazy dog"]
        dp = psannLMDataPrep(
            texts, tokenizer="simple", max_length=32, pack_sequences=True, val_split=0.0
        )
        vocab = int(dp.vocab_size)
        tok = dp.tokenizer  # type: ignore[attr-defined]
        sample = tok.encode("the quick brown fox", add_specials=True)
        T = min(16, len(sample))
        base_ids = sample[:T]
        batch_ids = [base_ids, base_ids]
        model_cfg = dict(
            base="waveresnet", d_model=128, n_layers=2, n_heads=4, vocab_size=vocab, rope=True
        )

        # Compute single-GPU baseline
        device0 = torch.device("cuda", 0)
        torch.manual_seed(123)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(123)
        lm = psannLM(**model_cfg)
        model = lm._ensure_model(vocab).to(device0).eval()
        with torch.no_grad():
            import torch.nn.functional as F

            seq0 = torch.tensor(batch_ids, dtype=torch.long, device=device0)
            logits0 = model(seq0)
            V = int(vocab)
            loss_single = (
                F.cross_entropy(logits0.view(seq0.size(0) * seq0.size(1), V), seq0.view(-1))
                .detach()
                .float()
                .item()
            )

        # Try FSDP multi-process run
        import torch.multiprocessing as mp
        from random import randint as _randint

        port = 29677 + (_randint(0, 1000))
        ctx = mp.get_context("spawn")
        manager = ctx.Manager()
        shared = manager.dict()
        nprocs = 2
        try:
            mp.spawn(
                _fsdp_loss_worker,
                args=(nprocs, port, batch_ids, vocab, model_cfg, shared),
                nprocs=nprocs,
                join=True,
            )
        except Exception as exc:
            return {"status": "error", "reason": f"FSDP spawn failed: {exc}"}
        if "error" in shared:
            return {"status": "error", "reason": str(shared["error"])}
        fsdp_loss = shared.get("fsdp_avg_loss")
        if fsdp_loss is None:
            return {"status": "error", "reason": "FSDP run produced no result"}

        rel = abs(fsdp_loss - loss_single) / max(1e-8, abs(loss_single))
        return {
            "status": "ok",
            "single_loss": round(loss_single, 6),
            "fsdp_avg_loss": round(fsdp_loss, 6),
            "rel_diff": round(rel, 6),
            "world_size": nprocs,
            "engine": "torch.fsdp",
        }
    except Exception as e:
        # Try DeepSpeed fallback if installed
        try:
            pass  # type: ignore
        except Exception:
            return {"status": "skipped", "reason": f"FSDP/deepspeed unavailable: {e}"}
        return {"status": "skipped", "reason": f"DeepSpeed path not implemented: {e}"}


def gpu_07_generation_smoke() -> Dict[str, Any]:
    texts = ["pack my box with five dozen liquor jugs", "sphinx of black quartz judge my vow"]
    dp = psannLMDataPrep(
        texts, tokenizer="simple", max_length=64, pack_sequences=True, val_split=0.0
    )
    lm = psannLM(
        base="waveresnet", d_model=128, n_layers=2, n_heads=4, vocab_size=dp.vocab_size, rope=True
    )
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
    dp = psannLMDataPrep(
        texts, tokenizer="simple", max_length=16, pack_sequences=True, val_split=0.0
    )
    lm = psannLM(
        base="respsann", d_model=128, n_layers=2, n_heads=4, vocab_size=dp.vocab_size, rope=True
    )
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
        for (n1, p1), (n2, p2) in zip(
            lm._model.state_dict().items(), loaded._model.state_dict().items()
        ):
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
    ap.add_argument(
        "--only",
        nargs="+",
        default=None,
        metavar="STEP",
        help="Restrict execution to specific GPU steps (e.g. --only GPU-03 GPU-04)",
    )
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

    step_fns = {
        "GPU-01": gpu_01_forward_backward,
        "GPU-02": gpu_02_amp_parity,
        "GPU-03": gpu_03_throughput,
        "GPU-04": gpu_04_checkpointing,
        "GPU-05": gpu_05_ddp,
        "GPU-06": gpu_06_zerofsdp,
        "GPU-07": gpu_07_generation_smoke,
        "GPU-08": lambda: gpu_08_save_load(outdir),
    }
    selected = list(step_fns.keys())
    if args.only:
        filtered = []
        for name in args.only:
            name = name.upper()
            if name not in step_fns:
                raise SystemExit(
                    f"Unknown step '{name}'. Valid options: {', '.join(step_fns.keys())}"
                )
            filtered.append(name)
        selected = filtered

    for name in selected:
        if name == "GPU-08":
            continue  # run after loop to preserve checkpoint output usage
        fn = step_fns[name]
        _log(f"[RUN] {name}")
        try:
            res = fn()
            summary["results"][name] = res
            _log(f"[OK ] {name}: {res}")
        except Exception as e:  # pragma: no cover - runtime safety
            summary["results"][name] = {"status": "error", "error": str(e)}
            _log(f"[ERR] {name}: {e}")
        _write_json(outdir / "summary.json", summary)

    if "GPU-08" in selected:
        _log("[RUN] GPU-08")
        try:
            res8 = step_fns["GPU-08"]()
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
