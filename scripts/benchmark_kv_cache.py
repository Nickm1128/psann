"""Benchmark the PyTorch-only KV-cache fast path used by psannLM.generate_batch.

This script compares the batched KV-cache path against a naive per-sample
generation loop (which reprocesses the entire prompt every token) to quantify
the speed-up we get without a custom C++/CUDA kernel.

Example:
    python scripts/benchmark_kv_cache.py --batch-size 8 --prompt-length 96 \
        --max-new-tokens 64 --out reports/kv_cache/<ts>/metrics.json
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

from psannlm.lm import psannLM, psannLMDataPrep


@contextmanager
def _force_device(mode: str):
    """Temporarily force torch.cuda.is_available() to return False when needed."""

    if mode != "cpu":
        yield
        return
    original = torch.cuda.is_available

    def _always_false() -> bool:
        return False

    torch.cuda.is_available = _always_false  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.cuda.is_available = original  # type: ignore[assignment]


def _build_training_texts() -> List[str]:
    base = "abcdefghijklmnopqrstuvwxyz"
    extra = base.upper() + "0123456789.,;:!?-" + "/\\ \n"
    samples = [
        (base + extra) * 4,
        "psann kv cache benchmarking corpus for tokenizer coverage",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
    ]
    return samples


def _make_prompts(batch_size: int, prompt_length: int) -> List[str]:
    prompt_length = max(prompt_length, 24)
    stem = "kv-cache prompt "
    filler_len = max(0, prompt_length - len(stem))
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    prompts: List[str] = []
    for i in range(batch_size):
        filler_char = alphabet[i % len(alphabet)]
        filler = (filler_char * filler_len) or filler_char
        prompts.append(stem + filler)
    return prompts


@dataclass
class BenchmarkConfig:
    batch_size: int
    prompt_length: int
    max_new_tokens: int
    base: str
    d_model: int
    n_layers: int
    n_heads: int
    tokenizer: str
    positional_encoding: str
    device_mode: str


def _run_generate_batch(
    model: psannLM, prompts: Sequence[str], cfg: BenchmarkConfig
) -> Tuple[float, List[str]]:
    start = time.perf_counter()
    outputs = model.generate_batch(
        prompts,
        max_new_tokens=cfg.max_new_tokens,
        top_k=None,
        top_p=1.0,
        temperature=1.0,
        repetition_penalty=None,
    )
    elapsed = time.perf_counter() - start
    return elapsed, outputs


def _run_generate_naive(model: psannLM, prompts: Sequence[str], cfg: BenchmarkConfig) -> float:
    start = time.perf_counter()
    for prompt in prompts:
        model.generate(
            prompt,
            max_new_tokens=cfg.max_new_tokens,
            top_k=None,
            top_p=1.0,
            temperature=1.0,
            repetition_penalty=None,
        )
    return time.perf_counter() - start


def run_benchmark(cfg: BenchmarkConfig) -> Dict[str, Any]:
    texts = _build_training_texts()
    dp = psannLMDataPrep(
        texts,
        tokenizer=cfg.tokenizer,
        max_length=max(cfg.prompt_length + cfg.max_new_tokens + 8, 32),
        pack_sequences=True,
        val_split=0.0,
    )
    model = psannLM(
        base=cfg.base,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        vocab_size=dp.vocab_size,
        positional_encoding=cfg.positional_encoding,
    )
    model._tokenizer = dp.tokenizer  # type: ignore[attr-defined]
    _ = model._ensure_model(dp.vocab_size)  # type: ignore[attr-defined]

    prompts = _make_prompts(cfg.batch_size, cfg.prompt_length)

    # Warm-up (no timing) to populate caches/graph
    model.generate_batch(prompts, max_new_tokens=4, top_p=1.0, top_k=None)

    fast_elapsed, outputs = _run_generate_batch(model, prompts, cfg)
    naive_elapsed = _run_generate_naive(model, prompts, cfg)

    total_tokens = cfg.batch_size * cfg.max_new_tokens
    fast_tps = total_tokens / fast_elapsed
    naive_tps = total_tokens / naive_elapsed

    return {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "config": asdict(cfg),
        "batch_tokens": total_tokens,
        "fast_path": {
            "latency_s": fast_elapsed,
            "tokens_per_s": fast_tps,
            "per_token_ms": (fast_elapsed / total_tokens) * 1000.0,
            "sample_output_preview": outputs[0] if outputs else "",
        },
        "naive_path": {
            "latency_s": naive_elapsed,
            "tokens_per_s": naive_tps,
            "per_token_ms": (naive_elapsed / total_tokens) * 1000.0,
        },
        "speedup_vs_naive": naive_elapsed / fast_elapsed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark psannLM KV-cache fast path.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--prompt-length", type=int, default=96)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--base", type=str, default="waveresnet", choices=["waveresnet", "respsann"]
    )
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--tokenizer", type=str, default="simple")
    parser.add_argument(
        "--positional-encoding", type=str, default="rope", choices=["rope", "alibi", "sinusoidal"]
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu"])
    parser.add_argument(
        "--out", type=Path, default=None, help="Optional JSON path for benchmark metrics."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = BenchmarkConfig(
        batch_size=args.batch_size,
        prompt_length=args.prompt_length,
        max_new_tokens=args.max_new_tokens,
        base=args.base,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        tokenizer=args.tokenizer,
        positional_encoding=args.positional_encoding,
        device_mode=args.device,
    )

    out_path = args.out
    if out_path is None:
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_dir = Path("reports") / "kv_cache" / ts
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "metrics.json"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    with _force_device(cfg.device_mode):
        metrics = run_benchmark(cfg)

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"\nSaved benchmark metrics to {out_path}")


if __name__ == "__main__":
    main()
