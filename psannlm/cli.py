#!/usr/bin/env python3
"""Unified CLI for PSANN-LM training, evaluation, and generation."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset  # type: ignore

from psannlm.lm.data.tokenizer import Tokenizer, TokenizerConfig
from psannlm.lm.infer.generate import sample_next_token
from psannlm.lm.models.registry import get_base
from psannlm.lm.models.sine import SineConfig
from psannlm.train import main as train_main


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
    tok.fit([])
    return tok


def _eval_batches(
    ds,
    tokenizer: Tokenizer,
    *,
    text_key: str,
    seq_len: int,
    max_batches: int,
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


def _pretty_detok(text: str) -> str:
    for src, dst in ((" ,", ","), (" .", "."), (" !", "!"), (" ?", "?"), (" :", ":")):
        text = text.replace(src, dst)
    return text


def _generate(
    *,
    ckpt: Path,
    tokenizer_dir: Path,
    prompts: List[str],
    base: str,
    pos_enc: str,
    n_heads: Optional[int],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: Optional[int],
    add_bos: bool,
    stop_at_eos: bool,
    device: torch.device,
) -> None:
    state_dict = _load_state_dict(ckpt)
    vocab_size, d_model, d_mlp, n_layers = _infer_dims(state_dict)
    n_heads = int(n_heads) if n_heads else max(1, d_model // 64)
    if d_model % n_heads != 0 or (d_model // n_heads) % 2 != 0:
        raise SystemExit(
            f"Choose --n-heads that divides d_model evenly with even head_dim "
            f"(d_model={d_model}, n_heads={n_heads})."
        )

    tokenizer = _load_tokenizer(tokenizer_dir)
    if int(tokenizer.vocab_size) != int(vocab_size):
        raise SystemExit(
            f"Tokenizer vocab_size={tokenizer.vocab_size} does not match checkpoint vocab_size={vocab_size}."
        )

    factory = get_base(str(base))
    model = factory(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_mlp=d_mlp,
        dropout=0.0,
        positional_encoding=str(pos_enc),
        mlp_activation="sine",
        sine=SineConfig(),
        attn_impl="sdpa",
    )
    model.load_state_dict(state_dict)
    model.to(device).eval()

    bos_id = int(getattr(tokenizer, "bos_id", 1))
    eos_id = int(getattr(tokenizer, "eos_id", 2))

    for prompt in prompts:
        prompt_ids = tokenizer.encode(prompt, add_specials=False)
        if add_bos:
            prompt_ids = [bos_id] + [int(t) for t in prompt_ids]
        context = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        generated: List[int] = []
        for _ in range(int(max_new_tokens)):
            with torch.no_grad():
                logits = model(context)
                next_id = sample_next_token(
                    logits[:, -1, :],
                    temperature=float(temperature),
                    top_k=top_k,
                    top_p=float(top_p),
                )
            nid = int(next_id.item())
            generated.append(nid)
            context = torch.cat([context, next_id.view(1, 1)], dim=1)
            if stop_at_eos and nid == eos_id:
                break
        out = tokenizer.decode(generated, skip_specials=True)
        out = _pretty_detok(out)
        print("\n[prompt]\n" + prompt)
        print("[output]\n" + out)


def _eval(
    *,
    ckpt: Path,
    tokenizer_dir: Path,
    dataset: str,
    name: Optional[str],
    data_files: Optional[str],
    split: str,
    text_key: str,
    seq_len: int,
    max_batches: int,
    add_specials: bool,
    attn_impl: str,
    base: str,
    pos_enc: str,
    n_heads: Optional[int],
    device: torch.device,
) -> None:
    state_dict = _load_state_dict(ckpt)
    vocab_size, d_model, d_mlp, n_layers = _infer_dims(state_dict)
    n_heads = int(n_heads) if n_heads else max(1, d_model // 64)
    if d_model % n_heads != 0 or (d_model // n_heads) % 2 != 0:
        raise SystemExit(
            f"Choose --n-heads that divides d_model evenly with even head_dim "
            f"(d_model={d_model}, n_heads={n_heads})."
        )

    tokenizer = _load_tokenizer(tokenizer_dir)
    factory = get_base(str(base))
    model = factory(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_mlp=d_mlp,
        dropout=0.0,
        positional_encoding=str(pos_enc),
        mlp_activation="sine",
        sine=SineConfig(),
        attn_impl=str(attn_impl),
    )
    model.load_state_dict(state_dict)
    model.to(device).eval()

    if data_files:
        files = [s.strip() for s in str(data_files).split(",") if s.strip()]
        ds = load_dataset(dataset, data_files=files, split="train", streaming=True)
    else:
        ds = load_dataset(dataset, name=name, split=split, streaming=True)

    iterator = _eval_batches(
        ds,
        tokenizer,
        text_key=str(text_key),
        seq_len=int(seq_len),
        max_batches=int(max_batches),
        add_specials=bool(add_specials),
    )

    total_loss = 0.0
    total_tokens = 0
    use_amp = device.type == "cuda"
    with torch.no_grad():
        for input_ids, labels in iterator:
            input_ids = input_ids.unsqueeze(0).to(device)
            labels = labels.unsqueeze(0).to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits = model(input_ids)
                bsz, tsz, vocab = logits.shape
                loss = F.cross_entropy(logits.view(bsz * tsz, vocab), labels.view(bsz * tsz), reduction="sum")
            total_loss += float(loss.item())
            total_tokens += int(labels.numel())

    if total_tokens == 0:
        print("[eval] No tokens processed; check dataset/text-key/filters.")
        return
    ppl = math.exp(total_loss / total_tokens)
    print(f"[eval] tokens={total_tokens} loss={total_loss/total_tokens:.4f} ppl={ppl:.3f}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PSANN-LM unified CLI")
    sub = p.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train or resume a PSANN-LM run")
    train_p.add_argument("args", nargs=argparse.REMAINDER)

    resume_p = sub.add_parser("resume", help="Resume training (alias of train)")
    resume_p.add_argument("args", nargs=argparse.REMAINDER)

    eval_p = sub.add_parser("eval", help="Evaluate perplexity on a dataset")
    eval_p.add_argument("--ckpt", required=True)
    eval_p.add_argument("--tokenizer-dir", required=True)
    eval_p.add_argument("--dataset", default="allenai/c4")
    eval_p.add_argument("--name", default=None)
    eval_p.add_argument("--data-files", default=None)
    eval_p.add_argument("--split", default="validation")
    eval_p.add_argument("--text-key", default="text")
    eval_p.add_argument("--seq-len", type=int, default=2048)
    eval_p.add_argument("--max-batches", type=int, default=128)
    eval_p.add_argument("--add-specials", action="store_true")
    eval_p.add_argument("--attn-impl", type=str, default="sdpa", choices=["math", "sdpa", "auto"])
    eval_p.add_argument("--base", type=str, default="waveresnet")
    eval_p.add_argument("--pos-enc", type=str, default="rope")
    eval_p.add_argument("--n-heads", type=int, default=None)
    eval_p.add_argument("--device", type=str, default="auto")

    gen_p = sub.add_parser("generate", help="Generate text from a checkpoint")
    gen_p.add_argument("--ckpt", required=True)
    gen_p.add_argument("--tokenizer-dir", required=True)
    gen_p.add_argument("--prompt", action="append", default=None)
    gen_p.add_argument("--prompts-file", type=str, default=None)
    gen_p.add_argument("--max-new-tokens", type=int, default=256)
    gen_p.add_argument("--temperature", type=float, default=0.7)
    gen_p.add_argument("--top-p", type=float, default=0.9)
    gen_p.add_argument("--top-k", type=int, default=None)
    gen_p.add_argument("--add-bos", action="store_true")
    gen_p.add_argument("--stop-at-eos", action="store_true", default=True)
    gen_p.add_argument("--no-stop-at-eos", dest="stop_at_eos", action="store_false")
    gen_p.add_argument("--base", type=str, default="waveresnet")
    gen_p.add_argument("--pos-enc", type=str, default="rope")
    gen_p.add_argument("--n-heads", type=int, default=None)
    gen_p.add_argument("--device", type=str, default="auto")

    return p


def _read_prompts(prompt_list: Optional[List[str]], prompts_file: Optional[str]) -> List[str]:
    prompts: List[str] = []
    if prompts_file:
        for line in Path(prompts_file).read_text(encoding="utf-8").splitlines():
            s = line.strip("\n").replace("\\n", "\n")
            if s.strip():
                prompts.append(s)
    if prompt_list:
        prompts.extend([str(p).replace("\\n", "\n") for p in prompt_list if str(p).strip()])
    if not prompts:
        prompts = ["The future of PSANN-LM is"]
    return prompts


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in {"train", "resume"}:
        if not args.args:
            raise SystemExit("Provide training flags after the subcommand.")
        if args.command == "resume" and "--resume-ckpt" not in args.args:
            raise SystemExit("resume requires --resume-ckpt.")
        return int(train_main(list(args.args)) or 0)

    if args.command == "eval":
        device = torch.device(
            "cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu"
        )
        _eval(
            ckpt=Path(args.ckpt),
            tokenizer_dir=Path(args.tokenizer_dir),
            dataset=str(args.dataset),
            name=args.name,
            data_files=args.data_files,
            split=str(args.split),
            text_key=str(args.text_key),
            seq_len=int(args.seq_len),
            max_batches=int(args.max_batches),
            add_specials=bool(args.add_specials),
            attn_impl=str(args.attn_impl),
            base=str(args.base),
            pos_enc=str(args.pos_enc),
            n_heads=args.n_heads,
            device=device,
        )
        return 0

    if args.command == "generate":
        device = torch.device(
            "cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu"
        )
        prompts = _read_prompts(args.prompt, args.prompts_file)
        _generate(
            ckpt=Path(args.ckpt),
            tokenizer_dir=Path(args.tokenizer_dir),
            prompts=prompts,
            base=str(args.base),
            pos_enc=str(args.pos_enc),
            n_heads=args.n_heads,
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            top_k=args.top_k,
            add_bos=bool(args.add_bos),
            stop_at_eos=bool(args.stop_at_eos),
            device=device,
        )
        return 0

    raise SystemExit(f"Unknown command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
