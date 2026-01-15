"""lm-eval adapter for PSANN-LM checkpoints.

Usage example:

  lm_eval --model custom \
    --model_args \
      custom_model_class=psann_adapter.PSANNLM,\
      ckpt=runs/lm/wrn_cpu_local/final_model.pt,\
      device=cuda,\
      tokenizer_backend=sentencepiece,\
      tokenizer_model_path=examples/lm/tokenizer/sample_texts.model \
    --tasks wikitext,lambada_openai,hellaswag,piqa,winogrande \
    --limit 1500 --num_fewshot 0 --batch_size auto \
    --output_path eval_out/smoke_quick.json

Notes:
- For meaningful perplexity/accuracy, the tokenizer MUST match training.
  Pass a `tokenizer_model_path` (SentencePiece .model or HF tokenizers .json)
  that was used during training. If omitted, results may be invalid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import logging

import math
import torch
import torch.nn.functional as F

from lm_eval.api.model import LM

LOGGER = logging.getLogger(__name__)


@dataclass
class _TokShim:
    impl: object
    pad_id: int
    bos_id: int
    eos_id: int
    unk_id: int

    def encode(self, s: str, *, add_specials: bool = True) -> List[int]:
        return list(self.impl.encode(s, add_specials=add_specials))

    def decode(self, ids: List[int], *, skip_specials: bool = True) -> str:
        return str(self.impl.decode(ids, skip_specials=skip_specials))


class PSANNLM(LM):
    """Custom model wrapper for EleutherAI lm-eval.

    Args (via --model_args):
      - ckpt: path to psannLM.save(".../final_model.pt") checkpoint
      - device: "cuda" or "cpu" (defaults to auto)
      - tokenizer_backend: one of {auto,sentencepiece,tokenizers,simple}
      - tokenizer_model_path: path to SPM .model or HF tokenizers .json (RECOMMENDED)
      - max_batch_size: cap for request batching (default: 8)
      - max_ctx: max context tokens including BOS (default: 2048)
    """

    def __init__(
        self,
        *,
        ckpt: str | None = None,
        hf_repo: str | None = None,
        hf_filename: str | None = None,
        hf_revision: str | None = None,
        hf_token: str | None = None,
        device: str | None = None,
        tokenizer_backend: str = "auto",
        tokenizer_model_path: Optional[str] = None,
        tokenizer_special_map_path: Optional[str] = None,
        max_batch_size: int = 8,
        max_ctx: int = 2048,
        batch_size: object | None = None,
    ) -> None:
        super().__init__()

        # Resolve device
        if device is None or str(device).lower() == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(str(device))

        # Lazy import from local package path
        import sys
        from pathlib import Path
        root = Path(__file__).resolve().parent
        src_dir = root / "src"
        if src_dir.exists():
            sys.path.insert(0, str(src_dir))

        # Resolve checkpoint (local path or download from Hugging Face Hub)
        if ckpt is None and hf_repo is None:
            raise ValueError("Provide either 'ckpt' or 'hf_repo' to locate the model artifact.")

        if ckpt is None and hf_repo is not None:
            from huggingface_hub import hf_hub_download
            tried: list[str] = []
            candidates = [
                hf_filename or "final_model.pt",
                "model.pt",
                "checkpoint.pt",
                "psann_final.pt",
                "pytorch_model.bin",  # fallback if user pushed raw torch state
            ]
            local_ckpt: str | None = None
            last_err: Exception | None = None
            for name in candidates:
                try:
                    path = hf_hub_download(repo_id=hf_repo, filename=name, revision=hf_revision, token=hf_token)
                    local_ckpt = path
                    break
                except Exception as e:
                    tried.append(name)
                    last_err = e
            if local_ckpt is None:
                raise RuntimeError(
                    f"Could not download a checkpoint from '{hf_repo}'. Tried: {tried}.\n"
                    f"If your file has a different name, pass hf_filename=<name>."
                ) from last_err
            ckpt = local_ckpt

        assert ckpt is not None

        # Load PSANN-LM model
        from psannlm.lm.api import psannLM  # type: ignore
        inst = psannLM.load(ckpt)
        self.model = inst._ensure_model(int(inst.vocab_size or 32000))
        self.model.eval().to(self.device)

        # Build tokenizer facade; MUST match training to be meaningful
        from psannlm.lm.data.tokenizer import Tokenizer, TokenizerConfig  # type: ignore

        cfg = TokenizerConfig(
            backend=str(tokenizer_backend or "auto"),
            model_path=str(tokenizer_model_path) if tokenizer_model_path else None,
            special_tokens_map_path=str(tokenizer_special_map_path) if tokenizer_special_map_path else None,
            hf_passthrough_ids=(str(tokenizer_backend or "auto").lower() == "tokenizers"),
        )
        tok = Tokenizer(cfg)
        # Ensure backend is fully ready (SP/HF loaders stick model in fit())
        # For loaders, the actual input text is irrelevant; they read from model_path.
        try:
            tok.fit([""])
        except Exception:
            # Some backends may already be materialized; ignore.
            pass
        self.tok = _TokShim(tok, tok.pad_id, tok.bos_id, tok.eos_id, tok.unk_id)

        self.vocab_size = int(getattr(self.model, "lm_head").weight.shape[0])
        self._id_range_warned = False
        # lm-eval may pass a `batch_size` top-level arg into the model ctor.
        # We don't need it beyond shaping; keep a sensible cap.
        try:
            if isinstance(batch_size, int):
                max_batch_size = int(batch_size)
        except Exception:
            pass
        self.max_batch_size = int(max(1, max_batch_size))
        self.max_ctx = int(max(8, max_ctx))

    # --------- Helper encoding/forward utilities ---------
    def _encode_pair(self, context: str, continuation: str) -> Tuple[torch.Tensor, int]:
        # Encode without injecting specials; we add BOS/EOS explicitly
        ctx_ids = self.tok.encode(context, add_specials=False)
        cont_ids = self.tok.encode(continuation, add_specials=False)

        # Assemble full sequence = [BOS] + ctx + cont + [EOS]
        full: List[int] = [self.tok.bos_id] + ctx_ids + cont_ids + [self.tok.eos_id]

        # Truncate from the left to fit context window
        if len(full) > self.max_ctx:
            # Keep at least BOS + cont + EOS
            keep = max(self.max_ctx, len(cont_ids) + 2)
            full = full[-keep:]
        input_ids = torch.tensor([full], dtype=torch.long, device=self.device)
        # Continuation target range in next-token targets
        # input: [BOS, ctx..., cont..., EOS]
        # targets align to input[:, 1:], so the first cont token is at index ctx_len-1
        cont_len = len(cont_ids)
        return input_ids, cont_len

    def _sanitize_token_ids(self, ids: torch.Tensor) -> torch.Tensor:
        """Map any out-of-range token ids to a safe value (UNK by default)."""
        vocab_cap = int(getattr(self, "vocab_size", 0) or 0)
        if ids.dtype != torch.long:
            ids = ids.long()
        if vocab_cap <= 0:
            return ids
        mask = (ids < 0) | (ids >= vocab_cap)
        needs_fix = bool(mask.any().item()) if mask.numel() else False
        if not needs_fix:
            return ids
        if not self._id_range_warned:
            unk = getattr(self.tok, "unk_id", None)
            LOGGER.warning(
                "Tokenizer ids exceed model vocab size (%d); remapping to unk_id=%s.",
                vocab_cap,
                unk,
            )
            self._id_range_warned = True
        safe = ids.clamp(0, vocab_cap - 1)
        unk_id = getattr(self.tok, "unk_id", None)
        if unk_id is not None:
            unk_val = int(unk_id)
            if 0 <= unk_val < vocab_cap:
                safe = safe.masked_fill(mask, unk_val)
        return safe

    @torch.no_grad()
    def _score_continuation(self, input_ids: torch.Tensor, cont_len: int) -> Tuple[float, bool]:
        input_ids = self._sanitize_token_ids(input_ids)
        logits = self.model(input_ids)
        logprobs = F.log_softmax(logits[:, :-1, :], dim=-1)  # (1, T-1, V)
        targets = input_ids[:, 1:]  # (1, T-1)
        T = int(input_ids.shape[1])
        # Continuation occupies the last cont_len positions before EOS
        cont_start_full = max(0, T - 1 - int(cont_len))
        # Map to target indices (targets align to positions 1..T-1)
        start = max(0, cont_start_full - 1)
        end = max(start, start + int(cont_len))
        if end > targets.shape[1]:
            end = targets.shape[1]
        # Gather logprobs over continuation region
        lp_slice = logprobs[:, start:end, :]
        tgt_slice = targets[:, start:end]
        # Safety: clamp any out-of-range target ids
        V = lp_slice.shape[-1]
        if V <= 0:
            return 0.0, False
        tgt_slice = torch.clamp(tgt_slice, 0, V - 1)
        # Sum token log-probabilities
        lp = lp_slice.gather(-1, tgt_slice.unsqueeze(-1)).squeeze(-1)
        logprob_sum = float(lp.sum().item())
        # Greedy check
        greedy_ids = lp_slice.argmax(dim=-1)
        isgreedy = bool(torch.equal(greedy_ids, tgt_slice))
        return logprob_sum, isgreedy

    # ---------------- LM API methods ----------------
    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        out: List[Tuple[float, bool]] = []
        for req in requests:
            ctx, cont = req.args  # lm-eval wraps as Instance; .args -> (context, continuation)
            ids, cont_len = self._encode_pair(ctx, cont)
            lp, greedy = self._score_continuation(ids, cont_len)
            out.append((lp, greedy))
        return out

    def loglikelihood_rolling(self, requests) -> List[float]:
        # Compute full-string loglikelihood per request with long-context rolling
        # lm-eval expects flat float log-likelihoods per document.
        results: List[float] = []
        for req in requests:
            (text,) = req.args
            # Tokenize without specials; we insert BOS/EOS and roll windows
            ids = self.tok.encode(text, add_specials=False)
            total_lp = 0.0
            # Build [BOS] + ids + [EOS]
            seq = [self.tok.bos_id] + ids + [self.tok.eos_id]
            # Slide over seq with window size <= max_ctx
            pos = 0
            while pos < len(seq) - 1:
                end = min(len(seq), pos + self.max_ctx)
                chunk = seq[pos:end]
                # We always score all next-tokens in this chunk except the first input token
                inp = torch.tensor([chunk], dtype=torch.long, device=self.device)
                inp = self._sanitize_token_ids(inp)
                # Guard embedding range
                logits = self.model(inp)
                logprobs = F.log_softmax(logits[:, :-1, :], dim=-1)
                targets = inp[:, 1:]
                V_out = int(logits.size(-1))
                targets = torch.clamp(targets, 0, max(0, V_out - 1))
                lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).sum().item()
                total_lp += float(lp)
                # Move window to overlap by 1 token to maximize context, per lm-eval spec
                pos = end - 1
            results.append(total_lp)
        return results

    @torch.no_grad()
    def generate_until(self, requests) -> List[str]:
        # Minimal greedy generator; sufficient for tasks that might call it.
        outs: List[str] = []
        max_new = 128
        for req in requests:
            ctx, until = req.args  # until: List[str]
            until = list(until) if until is not None else []
            # Start with [BOS] + ctx
            ctx_ids = self.tok.encode(ctx, add_specials=False)
            seq = [self.tok.bos_id] + ctx_ids
            generated: List[int] = []
            for _ in range(max_new):
                inp = torch.tensor([seq], dtype=torch.long, device=self.device)
                inp = self._sanitize_token_ids(inp)
                logits = self.model(inp)
                next_id = int(logits[0, -1].argmax(dim=-1).item())
                seq.append(next_id)
                generated.append(next_id)
                text = self.tok.decode(generated, skip_specials=True)
                if any(s in text for s in until):
                    break
                # stop on EOS
                if next_id == self.tok.eos_id:
                    break
            outs.append(self.tok.decode(generated, skip_specials=True))
        return outs
