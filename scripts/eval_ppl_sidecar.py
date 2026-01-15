#!/usr/bin/env python3
"""
Sidecar perplexity eval for PSANN-LM checkpoints.

Example:
  PYTHONPATH=src python scripts/eval_ppl_sidecar.py \
    --ckpt runs/lm/300m_en/ckpt_step004000.pt \
    --tokenizer-dir runs/tokenizer_300m_shuffle_v4 \
    --dataset allenai/c4 --split validation --text-key text \
    --max-batches 128 --seq-len 2048
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset  # type: ignore

# Local imports (repo root on PYTHONPATH)
from psannlm.lm.data.tokenizer import Tokenizer, TokenizerConfig
from psannlm.lm.models.sine import SineConfig
from psannlm.lm.models.registry import get_base


def _infer_dims(state_dict: dict) -> Tuple[int, int, int, int]:
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
    tok.fit([])  # loads from serialized model
    return tok


def batch_iterator(
    ds,
    tokenizer: Tokenizer,
    text_key: str,
    seq_len: int,
    max_batches: int,
    *,
    add_specials: bool,
) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    buffer: List[int] = []
    batches = 0
    for row in ds:
        text = str(row.get(text_key, "")).strip()
        if not text:
            continue
        try:
            ids = tokenizer.encode(text, add_specials=bool(add_specials))
        except Exception:
            continue
        if not ids:
            continue
        buffer.extend(int(t) for t in ids)
        while len(buffer) >= seq_len + 1:
            chunk = buffer[: seq_len + 1]
            input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
            labels = torch.tensor(chunk[1:], dtype=torch.long)
            yield input_ids, labels
            del buffer[:seq_len]
            batches += 1
            if 0 < max_batches <= batches:
                return


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute perplexity on a held-out split.")
    p.add_argument("--ckpt", required=True, help="Path to trainer checkpoint (.pt)")
    p.add_argument("--tokenizer-dir", required=True, help="Directory with tokenizer.json")
    p.add_argument("--dataset", default="allenai/c4", help="HF dataset id")
    p.add_argument(
        "--name",
        default=None,
        help="HF dataset config/name (e.g., 'en' for allenai/c4).",
    )
    p.add_argument(
        "--data-files",
        default=None,
        help="Optional local data files (comma-separated) for a JSON/Text dataset. "
        "When provided, --dataset is passed to load_dataset with data_files and split='train'.",
    )
    p.add_argument("--split", default="validation", help="HF split (e.g., validation)")
    p.add_argument("--text-key", default="text", help="Column containing text")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--max-batches", type=int, default=128, help="Limit batches for a quick eval")
    p.add_argument(
        "--add-specials",
        action="store_true",
        help="Add BOS/EOS tokens per document before packing (does NOT match streaming training).",
    )
    p.add_argument(
        "--attn-impl",
        type=str,
        default="sdpa",
        choices=["math", "sdpa", "auto"],
        help="Attention implementation for inference",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    vocab_size, d_model, d_mlp, n_layers = _infer_dims(state_dict)
    n_heads = max(1, d_model // 64)
    if d_model % n_heads != 0 or (d_model // n_heads) % 2 != 0:
        raise SystemExit(
            f"Choose an n_heads that divides d_model evenly with even head_dim "
            f"(inferred d_model={d_model}, n_heads={n_heads})."
        )

    tokenizer = _load_tokenizer(Path(args.tokenizer_dir))

    factory = get_base("waveresnet")
    model = factory(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_mlp=d_mlp,
        dropout=0.0,
        positional_encoding="rope",
        mlp_activation="sine",
        sine=SineConfig(),
        attn_impl=args.attn_impl,
    )
    model.load_state_dict(state_dict)
    model.to(device).eval()

    if args.data_files:
        data_files = [s.strip() for s in str(args.data_files).split(",") if s.strip()]
        ds = load_dataset(args.dataset, data_files=data_files, split="train", streaming=True)
    else:
        ds = load_dataset(
            args.dataset,
            name=args.name,
            split=args.split,
            streaming=True,
        )
    iterator = batch_iterator(
        ds,
        tokenizer,
        args.text_key,
        int(args.seq_len),
        int(args.max_batches),
        add_specials=bool(args.add_specials),
    )

    total_loss = 0.0
    total_tokens = 0
    use_amp = device.type == "cuda"
    with torch.no_grad():
        for input_ids, labels in iterator:
            input_ids = input_ids.unsqueeze(0).to(device)
            labels = labels.unsqueeze(0).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(input_ids)
                B, T, V = logits.shape
                loss = F.cross_entropy(logits.view(B * T, V), labels.view(B * T), reduction="sum")
            total_loss += float(loss.item())
            total_tokens += int(labels.numel())

    if total_tokens == 0:
        print("[eval] No tokens processed; check dataset/text-key/filters.")
        return
    ppl = math.exp(total_loss / total_tokens)
    print(f"[eval] tokens={total_tokens} loss={total_loss/total_tokens:.4f} ppl={ppl:.3f}")


if __name__ == "__main__":
    main()
