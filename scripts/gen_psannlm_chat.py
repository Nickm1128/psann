#!/usr/bin/env python
"""Load a trained PSANN-LM chat checkpoint and generate text.

Usage (examples):
  python scripts/gen_psannlm_chat.py \
    --ckpt runs/chat_runs/20251108_222544/psannlm_chat_final.pt \
    --tok  runs/chat_runs/20251108_222544/tokenizer_final \
    --prompt "User: Summarize PSANN-LM.\nAssistant:"

  python scripts/gen_psannlm_chat.py \
    --ckpt <path/to/ckpt.pt> --tok <path/to/tokenizer_dir> \
    --prompts-file prompts.txt --max-new-tokens 256 --temperature 0.7 --top-p 0.9
"""

from __future__ import annotations

import argparse
import os
from typing import List

import torch
from transformers import AutoTokenizer

from psann.lm import psannLM


def _log(msg: str) -> None:
    print(f"[gen] {msg}", flush=True)


class TokAdapter:
    def __init__(self, tok) -> None:
        self.t = tok

    @property
    def pad_id(self) -> int:
        return int(self.t.pad_token_id or 0)

    @property
    def bos_id(self) -> int:
        return int(self.t.bos_token_id or self.pad_id)

    @property
    def eos_id(self) -> int:
        return int(self.t.eos_token_id or self.pad_id)

    @property
    def vocab_size(self) -> int:
        return int(self.t.vocab_size)

    def encode(self, text: str, add_specials: bool = True) -> List[int]:
        ids = self.t.encode(text, add_special_tokens=False)
        return ([self.bos_id] + ids) if add_specials else ids

    def decode(self, ids, skip_specials: bool = True) -> str:
        return self.t.decode(ids, skip_special_tokens=skip_specials)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate text from a PSANN-LM chat checkpoint")
    p.add_argument("--ckpt", required=True, help="Path to psannLM checkpoint (.pt)")
    p.add_argument("--tok", required=True, help="Path to HF tokenizer directory")
    p.add_argument("--prompt", default=None, help="Single prompt string to generate from")
    p.add_argument("--prompts-file", default=None, help="Optional file with one prompt per line")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))

    _log(f"loading checkpoint: {args.ckpt}")
    lm = psannLM.load(args.ckpt)

    _log(f"loading tokenizer: {args.tok}")
    hf_tok = AutoTokenizer.from_pretrained(args.tok)
    if hf_tok.pad_token is None:
        hf_tok.pad_token = hf_tok.eos_token or hf_tok.unk_token
    lm._tokenizer = TokAdapter(hf_tok)

    prompts: List[str] = []
    if args.prompts_file:
        with open(args.prompts_file, "r", encoding="utf-8") as fh:
            for line in fh:
                s = line.strip()
                if s:
                    prompts.append(s)
    if args.prompt:
        prompts.append(args.prompt)
    if not prompts:
        prompts = [
            "User: Summarize the benefits of PSANN-LM in two sentences.\nAssistant:",
            "User: Write a haiku about transformers and sine activations.\nAssistant:",
        ]

    _log(f"device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    for p in prompts:
        out = lm.generate(
            p,
            max_new_tokens=int(args.max_new_tokens),
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=float(args.temperature),
        )
        print("\n[prompt]\n" + p)
        print("[output]\n" + out)
    _log("done")


if __name__ == "__main__":
    main()

