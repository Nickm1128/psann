#!/usr/bin/env python3
"""Supervised fine-tuning (SFT) entrypoint for PSANN-LM.

This script fine-tunes a pretrained PSANN-LM checkpoint on prompt/response
pairs. It masks loss on the prompt tokens (only the assistant response is
trained), and saves standard trainer checkpoints compatible with the existing
`psannlm.train` + `scripts/eval_ppl_sidecar.py` tooling.

Example (OpenAssistant/oasst1):
  PYTHONPATH=src python3 -m psannlm.sft \
    --init-ckpt runs/lm/300m_en/ckpt_step078000.pt \
    --tokenizer-dir runs/tokenizer_300m_shuffle_v4 \
    --sft-source oasst1 \
    --checkpoint-dir runs/lm/300m_en_sft_oasst1 \
    --seq-len 2048 --batch-tokens 65536 --grad-accum-steps 2 \
    --lr 5e-5 --warmup-steps 200 --max-steps 2000

Example (local JSONL with {"prompt": "...", "response": "..."}):
  PYTHONPATH=src python3 -m psannlm.sft \
    --init-ckpt runs/lm/300m_en/ckpt_step078000.pt \
    --tokenizer-dir runs/tokenizer_300m_shuffle_v4 \
    --sft-source pairs --dataset json --data-files sft_data.jsonl \
    --prompt-key prompt --response-key response \
    --checkpoint-dir runs/lm/300m_en_sft_local \
    --seq-len 2048 --batch-tokens 65536 --grad-accum-steps 2 \
    --lr 5e-5 --warmup-steps 200 --max-steps 2000
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch

from psannlm.lm.config import TrainConfig
from psannlm.lm.data.dataset import build_text_filter
from psannlm.lm.data.tokenizer import Tokenizer, TokenizerConfig
from psannlm.lm.models.registry import get_base
from psannlm.lm.models.sine import SineConfig
from psannlm.lm.train.trainer import Trainer

from .data_loader import build_stream_loader


def _require_hf_datasets():
    # Avoid local `./datasets/` namespace package shadowing when HF isn't installed.
    import importlib

    mod = importlib.import_module("datasets")
    if not hasattr(mod, "load_dataset"):
        raise SystemExit(
            "HuggingFace 'datasets' is not installed (or is being shadowed). "
            "Install it inside your training environment: pip install datasets"
        )
    return mod.load_dataset  # type: ignore[return-value]


def _infer_dims(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int, int, int]:
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
    tok.fit([])  # load from tokenizer.json
    return tok


def _load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"]
    if isinstance(payload, dict):
        # Some older checkpoints store raw state_dict at top-level
        return payload  # type: ignore[return-value]
    raise TypeError(f"Unsupported checkpoint format at {path!r}")


def _format_pair(
    prompt: str,
    response: str,
    *,
    template: str,
) -> Tuple[str, str]:
    p = str(prompt).strip()
    r = str(response).strip()
    if template == "chat":
        return f"User: {p}\nAssistant:", f" {r}"
    if template == "alpaca":
        return f"### Instruction:\n{p}\n\n### Response:\n", f"{r}"
    raise ValueError(f"Unknown template: {template}")


def _make_langdetect_filter(
    languages: Optional[Sequence[str]],
    *,
    lang_threshold: float,
) -> Optional[Callable[[str], bool]]:
    langs = [s.lower() for s in (languages or []) if s]
    if not langs:
        return None
    return build_text_filter(ascii_only=False, languages=langs, lang_threshold=lang_threshold)


def _build_oasst1_pairs(
    *,
    dataset: str,
    split: str,
    max_pairs: int,
    seed: int,
    languages: Optional[Sequence[str]] = None,
    lang_threshold: float = 0.85,
    ascii_only: bool = False,
) -> List[Tuple[str, str]]:
    load_dataset = _require_hf_datasets()
    ds = load_dataset(dataset, split=split)
    message_by_id: Dict[str, Dict[str, Any]] = {}
    for row in ds:
        mid = row.get("message_id")
        if mid:
            message_by_id[str(mid)] = row

    languages_norm = [s.lower() for s in (languages or []) if s]
    langdetect_filter = _make_langdetect_filter(languages_norm, lang_threshold=lang_threshold)

    def _passes(text: str, row_lang: Optional[str]) -> bool:
        if ascii_only and not text.isascii():
            return False
        if not languages_norm:
            return True
        if row_lang:
            return str(row_lang).lower() in languages_norm
        if langdetect_filter is None:
            return True
        return bool(langdetect_filter(text))

    assistant_rows: List[Dict[str, Any]] = []
    for row in ds:
        if str(row.get("role", "")).lower() != "assistant":
            continue
        parent_id = row.get("parent_id")
        if not parent_id or str(parent_id) not in message_by_id:
            continue
        assistant_rows.append(row)

    rng = random.Random(int(seed))
    rng.shuffle(assistant_rows)

    pairs: List[Tuple[str, str]] = []
    for row in assistant_rows:
        parent = message_by_id[str(row.get("parent_id"))]
        user_text = str(parent.get("text", "")).strip()
        assistant_text = str(row.get("text", "")).strip()
        if not user_text or not assistant_text:
            continue
        parent_lang = parent.get("lang")
        assistant_lang = row.get("lang")
        if not _passes(user_text, parent_lang):
            continue
        if not _passes(assistant_text, assistant_lang):
            continue
        pairs.append((user_text, assistant_text))
        if max_pairs > 0 and len(pairs) >= int(max_pairs):
            break
    return pairs


def _iter_hf_pairs_forever(
    *,
    dataset: str,
    name: Optional[str],
    split: str,
    data_files: Optional[str],
    prompt_key: str,
    response_key: str,
    ascii_only: bool,
    languages: Optional[Sequence[str]],
    lang_threshold: float,
    seed: int,
    shuffle: bool,
    shuffle_buffer: int,
) -> Iterator[Tuple[str, str]]:
    load_dataset = _require_hf_datasets()
    langdetect_filter = _make_langdetect_filter(languages, lang_threshold=lang_threshold)
    epoch = 0
    while True:
        epoch += 1
        if data_files:
            files = [s.strip() for s in str(data_files).split(",") if s.strip()]
            ds = load_dataset(dataset, data_files=files, split="train", streaming=True)
        else:
            ds = load_dataset(dataset, name=name, split=split, streaming=True)
        if shuffle:
            try:
                ds = ds.shuffle(seed=int(seed) + epoch, buffer_size=int(shuffle_buffer))
            except Exception:
                pass
        for row in ds:
            try:
                prompt = str(row.get(prompt_key, "")).strip()
                response = str(row.get(response_key, "")).strip()
            except Exception:
                continue
            if not prompt or not response:
                continue
            if ascii_only and (not prompt.isascii() or not response.isascii()):
                continue
            if langdetect_filter is not None:
                if not langdetect_filter(prompt) or not langdetect_filter(response):
                    continue
            yield prompt, response


def _shard_pairs(
    pairs: Sequence[Tuple[str, str]],
    *,
    worker_id: Optional[int],
    num_workers: Optional[int],
    rank: int,
    world_size: int,
) -> List[Tuple[str, str]]:
    if not pairs:
        return []
    out = []
    for i, pr in enumerate(pairs):
        if world_size > 1 and (i % world_size) != rank:
            continue
        if num_workers and num_workers > 1 and worker_id is not None and (i % num_workers) != worker_id:
            continue
        out.append(pr)
    return out


def streamed_sft_iterator(
    *,
    pair_iter: Callable[[], Iterator[Tuple[str, str]]],
    tokenizer: Tokenizer,
    seq_len: int,
    template: str,
    add_bos: bool,
    add_eos: bool,
    pack_buffer_tokens: int,
) -> Iterator[Dict[str, torch.Tensor]]:
    bos_id = int(getattr(tokenizer, "bos_id", 1))
    eos_id = int(getattr(tokenizer, "eos_id", 2))
    max_len = int(seq_len) + 1

    token_buf: List[int] = []
    mask_buf: List[bool] = []
    pack_cap = max(int(pack_buffer_tokens), max_len * 4)

    for prompt, response in pair_iter():
        p_txt, r_txt = _format_pair(prompt, response, template=template)
        try:
            prompt_ids = tokenizer.encode(p_txt, add_specials=False)
            resp_ids = tokenizer.encode(r_txt, add_specials=False)
        except Exception:
            continue
        if not prompt_ids or not resp_ids:
            continue

        tokens: List[int] = []
        mask: List[bool] = []
        if add_bos:
            tokens.append(bos_id)
            mask.append(False)
        tokens.extend(int(t) for t in prompt_ids)
        mask.extend([False] * len(prompt_ids))
        tokens.extend(int(t) for t in resp_ids)
        mask.extend([True] * len(resp_ids))
        if add_eos:
            tokens.append(eos_id)
            mask.append(True)

        if len(tokens) > max_len:
            # Prefer truncating from the start of the prompt, then from the end of the response.
            overflow = len(tokens) - max_len
            # Calculate prompt span (after optional BOS)
            prompt_start = 1 if add_bos else 0
            prompt_end = prompt_start + len(prompt_ids)
            prompt_len = max(0, prompt_end - prompt_start)
            drop_from_prompt = min(prompt_len, overflow)
            if drop_from_prompt > 0:
                del tokens[prompt_start : prompt_start + drop_from_prompt]
                del mask[prompt_start : prompt_start + drop_from_prompt]
                overflow -= drop_from_prompt
            if overflow > 0:
                tokens = tokens[:-overflow]
                mask = mask[:-overflow]
            if len(tokens) < 2:
                continue

        token_buf.extend(tokens)
        mask_buf.extend(mask)

        while len(token_buf) >= max_len:
            chunk_tokens = token_buf[:max_len]
            chunk_mask = mask_buf[:max_len]
            input_ids = torch.tensor(chunk_tokens[:-1], dtype=torch.long)
            label_tokens = chunk_tokens[1:]
            label_mask = chunk_mask[1:]
            labels = torch.tensor(
                [t if m else -100 for t, m in zip(label_tokens, label_mask)],
                dtype=torch.long,
            )
            yield {"input_ids": input_ids, "labels": labels}
            del token_buf[:seq_len]
            del mask_buf[:seq_len]

        if len(token_buf) > pack_cap:
            drop = len(token_buf) - pack_cap
            del token_buf[:drop]
            del mask_buf[:drop]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SFT fine-tune a pretrained PSANN-LM checkpoint.")

    # Checkpoints/tokenizer
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--init-ckpt", type=str, help="Pretrained trainer checkpoint to initialize from.")
    g.add_argument(
        "--resume-ckpt",
        type=str,
        help="Resume an existing SFT run checkpoint (loads model+optimizer).",
    )
    p.add_argument("--tokenizer-dir", type=str, required=True)

    # Data
    p.add_argument(
        "--sft-source",
        type=str,
        default="oasst1",
        choices=["oasst1", "pairs"],
        help="'oasst1' builds user/assistant pairs from OpenAssistant/oasst1; 'pairs' uses explicit prompt/response columns.",
    )
    p.add_argument("--dataset", type=str, default="OpenAssistant/oasst1")
    p.add_argument("--name", type=str, default=None, help="HF dataset config/name (optional).")
    p.add_argument("--split", type=str, default="train")
    p.add_argument(
        "--data-files",
        type=str,
        default=None,
        help="Local data files (comma-separated) for a JSON/Text dataset (pairs mode).",
    )
    p.add_argument("--prompt-key", type=str, default="prompt", help="Column for the prompt (pairs).")
    p.add_argument(
        "--response-key", type=str, default="response", help="Column for the response (pairs)."
    )
    p.add_argument("--max-pairs", type=int, default=0, help="Cap number of pairs (0 = all).")
    p.add_argument(
        "--lang",
        action="append",
        default=None,
        help="Optional language filter (e.g., --lang en).",
    )
    p.add_argument(
        "--lang-threshold",
        type=float,
        default=0.85,
        help="Language detection threshold when langdetect is used.",
    )
    p.add_argument("--ascii-only", action="store_true", help="Keep only ASCII text.")
    p.add_argument("--shuffle", action="store_true", help="Shuffle pairs in streaming mode.")
    p.add_argument("--shuffle-buffer", type=int, default=10_000)

    # Formatting
    p.add_argument("--template", type=str, default="chat", choices=["chat", "alpaca"])
    p.add_argument("--add-bos", action="store_true", help="Prepend BOS token per example.")
    p.add_argument("--add-eos", action="store_true", help="Append EOS token after each response.")

    # Model
    p.add_argument("--base", type=str, default="waveresnet")
    p.add_argument("--n-heads", type=int, default=None)
    p.add_argument("--pos-enc", type=str, default="rope", choices=["rope", "alibi", "sinusoidal"])
    p.add_argument("--attn-impl", type=str, default="sdpa", choices=["math", "sdpa", "auto"])

    # Training
    p.add_argument("--seq-len", type=int, default=2048)
    # Safer defaults for long-context SFT: smaller micro-batch, same tokens/step.
    p.add_argument("--batch-tokens", type=int, default=32_768)
    p.add_argument("--grad-accum-steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adamw8bit", "adafactor"])
    p.add_argument("--betas", type=str, default="0.9,0.95")
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--amp", type=str, default="bf16", choices=["bf16", "fp16", "fp32", "none"])
    p.add_argument("--grad-clip", type=float, default=1.0)
    # Default to gradient checkpointing to avoid OOM at long sequence lengths.
    p.add_argument(
        "--grad-checkpoint",
        dest="grad_checkpoint",
        action="store_true",
        help="Enable gradient checkpointing (default).",
    )
    p.add_argument(
        "--no-grad-checkpoint",
        dest="grad_checkpoint",
        action="store_false",
        help="Disable gradient checkpointing (can OOM at long seq_len).",
    )
    p.set_defaults(grad_checkpoint=True)
    p.add_argument(
        "--torch-compile",
        action="store_true",
        help="Enable torch.compile for the model (single GPU only; skipped under DDP/FSDP).",
    )
    p.add_argument(
        "--torch-compile-mode",
        type=str,
        default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode.",
    )
    p.add_argument(
        "--torch-compile-fullgraph",
        action="store_true",
        help="Pass fullgraph=True to torch.compile (can be more brittle).",
    )
    p.add_argument(
        "--torch-compile-dynamic",
        action="store_true",
        help="Pass dynamic=True to torch.compile (for dynamic shapes; can reduce performance).",
    )
    p.add_argument("--log-interval-steps", type=int, default=25)
    p.add_argument("--save-interval-steps", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--tokens-target", type=int, default=0, help="Optional token budget to derive max-steps.")
    p.add_argument("--pack-buffer-tokens", type=int, default=2_000_000)
    p.add_argument("--num-workers", type=int, default=0)

    # Output
    p.add_argument("--checkpoint-dir", type=str, required=True)

    # Misc
    p.add_argument("--seed", type=int, default=1337)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    print(
        f"[env] torch={torch.__version__} cuda_available={torch.cuda.is_available()}",
        flush=True,
    )
    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            total_gb = props.total_memory / float(1024**3)
            print(
                f"[gpu] device={props.name} capability={props.major}.{props.minor} "
                f"total_mem_gb={total_gb:.2f}",
                flush=True,
            )
        except Exception:
            pass

    seq_len = int(args.seq_len)
    micro_batch = max(1, int(args.batch_tokens) // seq_len)
    tokens_per_step = micro_batch * seq_len * max(1, int(args.grad_accum_steps))
    max_steps = int(args.max_steps)
    tokens_target = int(args.tokens_target)
    if tokens_target > 0 and max_steps <= 0:
        max_steps = max(1, tokens_target // max(1, tokens_per_step))
    if max_steps <= 0:
        raise SystemExit("--max-steps must be set (or provide --tokens-target).")

    # Load tokenizer
    tokenizer_dir = Path(args.tokenizer_dir)
    tokenizer = _load_tokenizer(tokenizer_dir)

    # Load checkpoint state dict to infer model dims
    init_path = str(args.resume_ckpt or args.init_ckpt)
    state_dict = _load_state_dict(init_path)
    vocab_size, d_model, d_mlp, n_layers = _infer_dims(state_dict)
    if int(tokenizer.vocab_size) != int(vocab_size):
        raise SystemExit(
            f"Tokenizer vocab_size={tokenizer.vocab_size} does not match checkpoint vocab_size={vocab_size}. "
            f"Double-check --tokenizer-dir."
        )
    n_heads = int(args.n_heads) if args.n_heads else max(1, d_model // 64)
    if d_model % n_heads != 0 or (d_model // n_heads) % 2 != 0:
        raise SystemExit(
            f"Choose an --n-heads that divides d_model evenly with even head_dim "
            f"(got d_model={d_model}, n_heads={n_heads})."
        )

    # Model
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

    # Data stream
    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    random.seed(int(args.seed) + rank)

    add_bos = bool(args.add_bos)
    add_eos = bool(args.add_eos)

    if args.sft_source == "oasst1":
        pairs = _build_oasst1_pairs(
            dataset=str(args.dataset),
            split=str(args.split),
            max_pairs=int(args.max_pairs),
            seed=int(args.seed),
            languages=args.lang,
            lang_threshold=float(args.lang_threshold),
            ascii_only=bool(args.ascii_only),
        )

        def _pair_iter(worker_info: Optional[Any] = None) -> Iterator[Tuple[str, str]]:
            worker_id = getattr(worker_info, "id", None) if worker_info else None
            num_workers = getattr(worker_info, "num_workers", None) if worker_info else None
            local_pairs = _shard_pairs(
                pairs,
                worker_id=worker_id,
                num_workers=num_workers,
                rank=rank,
                world_size=world_size,
            )
            if not local_pairs:
                raise RuntimeError("No SFT pairs after filtering/sharding.")
            rng = random.Random(int(args.seed) + 17 * (worker_id or 0) + 31 * rank)
            while True:
                rng.shuffle(local_pairs)
                for pr in local_pairs:
                    yield pr

        pair_iter_fn: Callable[[], Iterator[Tuple[str, str]]] = lambda: _pair_iter(None)
    else:
        # pairs mode: use HF datasets (streaming) or local JSONL
        def _pair_iter(worker_info: Optional[Any] = None) -> Iterator[Tuple[str, str]]:
            worker_id = getattr(worker_info, "id", None) if worker_info else None
            num_workers = getattr(worker_info, "num_workers", None) if worker_info else None
            # Adjust seed for each worker/rank so shuffle is de-correlated
            effective_seed = int(args.seed) + 31 * rank + 101 * (worker_id or 0)
            it = _iter_hf_pairs_forever(
                dataset=str(args.dataset),
                name=(str(args.name) if args.name else None),
                split=str(args.split),
                data_files=(str(args.data_files) if args.data_files else None),
                prompt_key=str(args.prompt_key),
                response_key=str(args.response_key),
                ascii_only=bool(args.ascii_only),
                languages=args.lang,
                lang_threshold=float(args.lang_threshold),
                seed=effective_seed,
                shuffle=bool(args.shuffle),
                shuffle_buffer=int(args.shuffle_buffer),
            )
            # Best-effort sharding on the pair stream by (index % shards)
            i = 0
            for pr in it:
                if world_size > 1 and (i % world_size) != rank:
                    i += 1
                    continue
                if num_workers and num_workers > 1 and worker_id is not None and (i % num_workers) != worker_id:
                    i += 1
                    continue
                i += 1
                yield pr

        pair_iter_fn = lambda: _pair_iter(None)

    def _iterator(worker_info: Optional[Any] = None):
        def _pairs_for_worker():
            return _pair_iter(worker_info)  # type: ignore[misc]

        return streamed_sft_iterator(
            pair_iter=_pairs_for_worker,
            tokenizer=tokenizer,
            seq_len=seq_len,
            template=str(args.template),
            add_bos=add_bos,
            add_eos=add_eos,
            pack_buffer_tokens=int(args.pack_buffer_tokens),
        )

    stream_loader = build_stream_loader(_iterator, batch_size=micro_batch, num_workers=int(args.num_workers))
    dataset = stream_loader.dataset  # type: ignore[assignment]

    # Training config
    try:
        b0, b1 = [float(x.strip()) for x in str(args.betas).split(",", 1)]
        betas = (b0, b1)
    except Exception:
        betas = (0.9, 0.95)

    print(
        f"[sft] seq_len={seq_len} batch_tokens={args.batch_tokens} micro_batch={micro_batch} "
        f"grad_accum={args.grad_accum_steps} tokens_per_step={tokens_per_step} "
        f"grad_checkpoint={bool(args.grad_checkpoint)}",
        flush=True,
    )
    tcfg = TrainConfig(
        epochs=1,
        batch_tokens=int(args.batch_tokens),
        lr=float(args.lr),
        warmup_steps=int(args.warmup_steps),
        weight_decay=float(args.weight_decay),
        optimizer=str(args.optimizer),
        betas=betas,
        eps=float(args.eps),
        amp=str(args.amp),
        grad_clip=float(args.grad_clip),
        grad_accum_steps=int(args.grad_accum_steps),
        ddp=str(getattr(args, "ddp", "auto")),
        fsdp=str(getattr(args, "fsdp", "off")),
        steps_per_epoch=int(max_steps),
        checkpoint_dir=str(args.checkpoint_dir),
        log_interval_steps=int(args.log_interval_steps),
        save_interval_steps=int(args.save_interval_steps),
        grad_checkpoint=bool(args.grad_checkpoint),
        torch_compile=bool(getattr(args, "torch_compile", False)),
        torch_compile_mode=str(getattr(args, "torch_compile_mode", "default")),
        torch_compile_fullgraph=bool(getattr(args, "torch_compile_fullgraph", False)),
        torch_compile_dynamic=bool(getattr(args, "torch_compile_dynamic", False)),
        dataloader_num_workers=int(args.num_workers),
    )

    trainer = Trainer(tcfg)

    # Initialize weights (and optionally resume optimizer state)
    if args.resume_ckpt:
        trainer.train(
            model,
            dataset,
            max_length=seq_len,
            val_dataset=None,
            data_loader=stream_loader,
            resume_checkpoint=str(args.resume_ckpt),
        )
        return 0

    model.load_state_dict(state_dict)
    trainer.train(
        model,
        dataset,
        max_length=seq_len,
        val_dataset=None,
        data_loader=stream_loader,
        resume_checkpoint=None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
