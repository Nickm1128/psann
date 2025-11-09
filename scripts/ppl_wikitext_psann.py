#!/usr/bin/env python
"""Compute perplexity on WikiText using PSANN-LM directly (fallback when lm-eval rolling API is finicky).

Example:
  python scripts/ppl_wikitext_psann.py \
    --ckpt runs/lm/wrn_cpu_local/final_model.pt \
    --tokenizer-backend sentencepiece \
    --tokenizer-model-path examples/lm/tokenizer/sample_texts.model \
    --hub-id iohadrubin/wikitext-103-raw-v1 --split validation \
    --limit 1000 --device cuda
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from datasets import load_dataset

# Ensure repository root on path
import sys
from pathlib import Path as _Path
_here = _Path(__file__).resolve()
_root = _here.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from psann_adapter import PSANNLM  # reuse adapter for scoring


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=str)
    p.add_argument("--device", default="auto", type=str)
    p.add_argument("--tokenizer-backend", default="auto", type=str)
    p.add_argument("--tokenizer-model-path", default=None, type=str)
    p.add_argument("--hub-id", default="iohadrubin/wikitext-103-raw-v1", type=str)
    p.add_argument("--split", default="validation", type=str)
    p.add_argument("--text-field", default="text", type=str)
    p.add_argument("--limit", default=1000, type=int)
    args = p.parse_args()

    lm = PSANNLM(
        ckpt=args.ckpt,
        device=args.device,
        tokenizer_backend=args.tokenizer_backend,
        tokenizer_model_path=args.tokenizer_model_path,
        max_ctx=2048,
        max_batch_size=4,
    )

    ds = load_dataset(args.hub_id, split=args.split)
    total_lp = 0.0
    total_toks = 0

    # build requests for loglikelihood_rolling-like scoring
    def _score_text(s: str) -> tuple[float, int]:
        ids = lm.tok.encode(s, add_specials=False)
        seq = [lm.tok.bos_id] + ids + [lm.tok.eos_id]
        lp_sum = 0.0
        n_tok = 0
        pos = 0
        while pos < len(seq) - 1:
            end = min(len(seq), pos + lm.max_ctx)
            chunk = seq[pos:end]
            inp = torch.tensor([chunk], dtype=torch.long, device=lm.device)
            with torch.no_grad():
                logits = lm.model(inp)
                logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)
                targets = inp[:, 1:]
                lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                lp_sum += float(lp.sum().item())
                n_tok += int(targets.numel())
            pos = end - 1
        return lp_sum, n_tok

    n = int(args.limit)
    for i, row in enumerate(ds):
        if i >= n:
            break
        text = str(row.get(args.text_field) or "").strip()
        if not text:
            continue
        lp, nt = _score_text(text)
        total_lp += lp
        total_toks += nt

    if total_toks == 0:
        print("No tokens scored; check dataset/text field.")
        return 1
    avg_nll = -total_lp / total_toks
    ppl = math.exp(avg_nll)
    print(f"WikiText ppl over {total_toks} tokens (limit={n} docs): {ppl:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

