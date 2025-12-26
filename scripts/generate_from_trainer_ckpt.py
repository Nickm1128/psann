#!/usr/bin/env python3
"""Generate text from a PSANN-LM *trainer* checkpoint (.pt).

This works with the checkpoints produced by pretraining (`psannlm.train`) and
SFT (`psannlm.sft`), which store the model state dict under the `"model"` key.

Examples:
  # SFT checkpoint (chat-style prompt)
  python3 scripts/generate_from_trainer_ckpt.py \\
    --ckpt runs/lm/300m_en_sft_300m_sft_oasst1/final.pt \\
    --tokenizer-dir runs/tokenizer_300m_shuffle_v4 \\
    --add-bos \\
    --prompt "User: Summarize PSANN-LM in one paragraph.\\nAssistant:"

  # Pretraining checkpoint (usually no BOS)
  python3 scripts/generate_from_trainer_ckpt.py \\
    --ckpt runs/lm/300m_en/ckpt_step057000.pt \\
    --tokenizer-dir runs/tokenizer_300m_shuffle_v4 \\
    --prompt "The future of"
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
import sys
from typing import Iterable, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from psann.lm.infer.generate import sample_next_token  # noqa: E402
from psann.lm.data.tokenizer import Tokenizer, TokenizerConfig  # noqa: E402
from psann.lm.models.registry import get_base  # noqa: E402
from psann.lm.models.sine import SineConfig  # noqa: E402


def _infer_dims(state_dict: dict) -> Tuple[int, int, int, int]:
    vocab_size, d_model = state_dict["embed.weight"].shape
    d_mlp = state_dict["blocks.0.mlp.fc1.weight"].shape[0]
    layers = [int(k.split(".")[1]) for k in state_dict if k.startswith("blocks.")]
    n_layers = max(layers) + 1 if layers else 0
    return int(vocab_size), int(d_model), int(d_mlp), int(n_layers)


def _load_state_dict(ckpt_path: Path) -> dict:
    payload = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"]
    if isinstance(payload, dict):
        return payload
    raise TypeError(f"Unsupported checkpoint format at {str(ckpt_path)!r}")


def _load_tokenizer(tokenizer_dir: Path) -> Tokenizer:
    cfg = TokenizerConfig(
        backend="tokenizers",
        model_path=str(tokenizer_dir / "tokenizer.json"),
        special_tokens_map_path=str(tokenizer_dir / "special_tokens_map.json"),
        hf_passthrough_ids=True,
    )
    tok = Tokenizer(cfg)
    tok.fit([])  # loads from tokenizer.json
    return tok


def _default_prompts() -> List[str]:
    return [
        "User: Summarize PSANN-LM in one paragraph.\nAssistant:",
        "User: Write a haiku about GPUs.\nAssistant:",
        "User: Explain what supervised fine-tuning (SFT) does.\nAssistant:",
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate text from a trainer checkpoint (.pt).")
    p.add_argument("--ckpt", required=True, help="Path to trainer checkpoint (.pt).")
    p.add_argument("--tokenizer-dir", required=True, help="Tokenizer directory (tokenizer.json + special_tokens_map.json).")
    p.add_argument("--prompt", action="append", help="Prompt to generate from (can be passed multiple times).")
    p.add_argument("--prompts-file", type=str, default=None, help="Optional file with one prompt per line.")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--min-new-tokens", type=int, default=0, help="Do not stop on EOS before this many tokens.")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help=">1.0 discourages repeating tokens (HuggingFace-style penalty).",
    )
    p.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=0,
        help="If >0, bans repeating any n-gram of this size (often 3 or 4).",
    )
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])

    # Model construction knobs (inferred from ckpt where possible)
    p.add_argument("--base", type=str, default="waveresnet")
    p.add_argument("--pos-enc", type=str, default="rope")
    p.add_argument("--n-heads", type=int, default=None, help="Override head count if needed.")
    p.add_argument("--attn-impl", type=str, default="sdpa", choices=["math", "sdpa", "auto"])

    # Token handling
    p.add_argument("--add-bos", action="store_true", help="Prepend BOS to the prompt (recommended for SFT).")
    p.add_argument("--stop-at-eos", action="store_true", help="Stop when EOS is generated (default).")
    p.add_argument("--no-stop-at-eos", dest="stop_at_eos", action="store_false")
    p.set_defaults(stop_at_eos=True)
    p.add_argument(
        "--pretty-decode",
        dest="pretty_decode",
        action="store_true",
        help="Apply lightweight whitespace/punctuation cleanup to decoded text (default).",
    )
    p.add_argument(
        "--raw-decode",
        dest="pretty_decode",
        action="store_false",
        help="Disable whitespace cleanup; print tokenizer.decode() output as-is.",
    )
    p.set_defaults(pretty_decode=True)
    return p.parse_args()


def _read_prompts(args: argparse.Namespace) -> List[str]:
    prompts: List[str] = []
    if args.prompts_file:
        for line in Path(args.prompts_file).read_text(encoding="utf-8").splitlines():
            s = line.strip("\n").replace("\\n", "\n")
            if s.strip():
                prompts.append(s)
    if args.prompt:
        prompts.extend([str(p).replace("\\n", "\n") for p in args.prompt if str(p).strip()])
    if not prompts:
        prompts = _default_prompts()
    return prompts


def _pretty_detok(text: str) -> str:
    # Punctuation spacing
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([(\[{])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]}])", r"\1", text)
    text = re.sub(r"\s+'", "'", text)

    # Merge common suffix splits from whitespace-joined BPE pieces.
    # This is best-effort; it improves readability for outputs like "Supervis ed" -> "Supervised".
    suffixes = (
        "s",
        "es",
        "ed",
        "er",
        "est",
        "ly",
        "ing",
        "ion",
        "ions",
        "tion",
        "tions",
        "ity",
        "ities",
        "ment",
        "ments",
        "ness",
        "less",
        "able",
        "ible",
        "ful",
        "ive",
        "ize",
        "ized",
        "izing",
        "al",
        "ally",
    )
    suf_re = "|".join(re.escape(s) for s in suffixes)
    text = re.sub(rf"\b([A-Za-z]{{3,}})\s+({suf_re})\b", r"\1\2", text)
    # Acronym stitch: "V RAM" -> "VRAM", "S FT" -> "SFT"
    text = re.sub(r"\b([A-Z])\s+([A-Z]{2,6})\b", r"\1\2", text)
    return text


def _apply_repetition_penalty(
    logits: torch.Tensor,
    *,
    token_ids: List[int],
    penalty: float,
) -> None:
    if penalty is None or penalty <= 1.0:
        return
    if not token_ids:
        return
    # HuggingFace-style repetition penalty: scale logits for seen tokens.
    # https://arxiv.org/abs/1909.05858 (heuristic popularized by CTRL)
    ids = set(int(t) for t in token_ids)
    for tid in ids:
        val = logits[0, tid]
        logits[0, tid] = val * float(penalty) if val < 0 else val / float(penalty)


def _get_banned_tokens_no_repeat_ngram(tokens: List[int], n: int) -> List[int]:
    if n <= 0:
        return []
    if len(tokens) < n:
        return []
    prefix = tokens[-(n - 1) :] if n > 1 else []
    banned: List[int] = []
    # Find all occurrences of prefix and ban the following token.
    # This is O(L*n) and fine for interactive generation sizes.
    limit = len(tokens) - n + 1
    for i in range(limit):
        if n == 1:
            banned.append(tokens[i])
            continue
        if tokens[i : i + n - 1] == prefix:
            banned.append(tokens[i + n - 1])
    return banned


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))

    device = torch.device("cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu")

    ckpt_path = Path(args.ckpt)
    state_dict = _load_state_dict(ckpt_path)
    vocab_size, d_model, d_mlp, n_layers = _infer_dims(state_dict)
    n_heads = int(args.n_heads) if args.n_heads else max(1, d_model // 64)
    if d_model % n_heads != 0 or (d_model // n_heads) % 2 != 0:
        raise SystemExit(
            f"Choose an --n-heads that divides d_model evenly with an even head_dim "
            f"(got d_model={d_model}, n_heads={n_heads})."
        )

    tokenizer = _load_tokenizer(Path(args.tokenizer_dir))
    if int(tokenizer.vocab_size) != int(vocab_size):
        raise SystemExit(
            f"Tokenizer vocab_size={tokenizer.vocab_size} does not match checkpoint vocab_size={vocab_size}. "
            f"Double-check --tokenizer-dir."
        )

    factory = get_base(str(args.base))
    model = factory(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_mlp=d_mlp,
        dropout=0.0,
        positional_encoding=str(args.pos_enc),
        mlp_activation="sine",
        sine=SineConfig(),
        attn_impl=str(args.attn_impl),
    )
    model.load_state_dict(state_dict)
    model.to(device).eval()

    bos_id = int(getattr(tokenizer, "bos_id", 1))
    eos_id = int(getattr(tokenizer, "eos_id", 2))

    print(
        f"[gen] device={device} ckpt={ckpt_path} vocab_size={vocab_size} d_model={d_model} "
        f"n_layers={n_layers} n_heads={n_heads} d_mlp={d_mlp} add_bos={bool(args.add_bos)}",
        flush=True,
    )

    prompts: Iterable[str] = _read_prompts(args)
    for prompt in prompts:
        prompt_ids = tokenizer.encode(prompt, add_specials=False)
        if args.add_bos:
            prompt_ids = [bos_id] + [int(t) for t in prompt_ids]
        context = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        generated: List[int] = []
        for _ in range(int(args.max_new_tokens)):
            with torch.no_grad():
                logits = model(context)
                next_logits = logits[:, -1, :]
                # Sampling guardrails to reduce repetition loops
                if args.repetition_penalty and float(args.repetition_penalty) > 1.0:
                    _apply_repetition_penalty(
                        next_logits,
                        token_ids=prompt_ids + generated,
                        penalty=float(args.repetition_penalty),
                    )
                if args.no_repeat_ngram_size and int(args.no_repeat_ngram_size) > 0:
                    banned = _get_banned_tokens_no_repeat_ngram(
                        prompt_ids + generated,
                        int(args.no_repeat_ngram_size),
                    )
                    if banned:
                        next_logits[0, torch.tensor(banned, device=next_logits.device)] = float("-inf")
                next_id = sample_next_token(
                    next_logits,
                    temperature=float(args.temperature),
                    top_k=args.top_k,
                    top_p=float(args.top_p),
                )
            nid = int(next_id.item())
            generated.append(nid)
            context = torch.cat([context, next_id.view(1, 1)], dim=1)
            if args.stop_at_eos and nid == eos_id and len(generated) >= int(args.min_new_tokens):
                break

        out = tokenizer.decode(generated, skip_specials=True)
        if args.pretty_decode:
            out = _pretty_detok(out)
        print("\n[prompt]\n" + prompt)
        print("[output]\n" + out)


if __name__ == "__main__":
    main()
