"""Public API for PSANN-LM.

This file defines the user-facing classes:

- `psannLM`:      model wrapper with fit() and generate() entry points
- `psannLMDataPrep`: lightweight data preparation wrapper

These are minimal stubs to enable imports and examples while
the underlying components (tokenizer, datasets, trainer, inference)
are implemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence

import torch
from torch import nn

from .config import TrainConfig, normalize_positional_encoding
from .models.registry import get_base
from .data.tokenizer import Tokenizer, TokenizerConfig
from .data.dataset import LMDataset
from .train.trainer import Trainer
from .models.sine import SineConfig


@dataclass
class SineParams:
    """Parameters controlling the parametric sine used in MLP blocks.

    This is a convenience container for the public API. Internal modules
    may represent and constrain these differently.
    """

    amp_init: float = 1.0
    freq_init: float = 1.0
    damp_init: float = 0.01
    trainable: bool = True


class psannLMDataPrep:
    """Lightweight data preparation wrapper for PSANN-LM.

    Parameters
    ----------
    texts:
        Iterable of raw text strings to prepare for language modeling.
    tokenizer:
        Tokenizer backend identifier. Use "auto" to select the default
        policy (sentencepiece -> tokenizers -> simple char fallback).
    max_length:
        Maximum sequence length for tokenized chunks.
    pack_sequences:
        If True, enables sequence packing for higher throughput.
    val_split:
        Optional fraction for validation split (0.0-1.0).
    seed:
        Random seed used for data shuffling/splitting.
    """

    def __init__(
        self,
        texts: Iterable[str],
        *,
        tokenizer: str = "auto",
        tokenizer_model_path: Optional[str] = None,
        tokenizer_special_map_path: Optional[str] = None,
        max_length: int = 1024,
        pack_sequences: bool = True,
        val_split: Optional[float] = None,
        seed: int = 1337,
    ) -> None:
        # Accept list of raw strings or file paths. If all entries are file paths that
        # exist, load them as text sources (one document per line).
        items = list(texts)
        if items and all(isinstance(t, str) for t in items):
            import os

            if all(os.path.exists(t) for t in items):
                loaded: list[str] = []
                for p in items:
                    try:
                        with open(p, "r", encoding="utf-8") as fh:
                            loaded.extend([ln.rstrip("\n") for ln in fh.readlines() if ln.strip()])
                    except Exception:
                        # Fallback: treat as a raw text if file cannot be read
                        loaded.append(p)
                self._texts = loaded
            else:
                self._texts = items
        else:
            self._texts = items
        self._tokenizer_backend = tokenizer
        self._tokenizer_model_path = tokenizer_model_path
        self.max_length = max_length
        self.pack_sequences = pack_sequences
        self.val_split = val_split
        self.seed = seed

        # Placeholder attributes until tokenizer/dataset wiring lands.
        # Build tokenizer and cached dataset lazily
        # Prefer passthrough ids for HF tokenizers to ensure parity with eval path
        _backend = str(tokenizer or "auto").lower()
        self._tokenizer = Tokenizer(
            TokenizerConfig(
                backend=_backend,
                model_path=tokenizer_model_path,
                special_tokens_map_path=tokenizer_special_map_path,
                hf_passthrough_ids=(_backend == "tokenizers"),
            )
        )
        self._tokenizer.fit(self._texts)
        self._vocab_size: int = self._tokenizer.vocab_size
        # Optional train/val split
        self._train_texts = self._texts
        self._val_texts: Optional[list[str]] = None
        if val_split is not None and float(val_split) > 0.0 and len(self._texts) > 1:
            import random as _random

            vs = float(val_split)
            n = len(self._texts)
            val_n = max(1, int(n * vs))
            val_n = min(n - 1, val_n)
            idxs = list(range(n))
            rng = _random.Random(int(seed))
            rng.shuffle(idxs)
            self._train_texts = [self._texts[i] for i in idxs[val_n:]]
            self._val_texts = [self._texts[i] for i in idxs[:val_n]]

        self._dataset: Optional[LMDataset] = None
        self._val_dataset: Optional[LMDataset] = None

    @property
    def vocab_size(self) -> int:
        """Vocabulary size for the prepared dataset."""
        return int(self._vocab_size)

    @property
    def tokenizer(self) -> Tokenizer:
        return self._tokenizer

    @property
    def dataset(self) -> LMDataset:
        if self._dataset is None:
            from .data.dataset import PackingConfig

            cfg = PackingConfig(max_length=self.max_length, pack_sequences=self.pack_sequences)
            self._dataset = LMDataset(self._train_texts, self._tokenizer, cfg)
        return self._dataset

    @property
    def pad_id(self) -> int:
        return int(self._tokenizer.pad_id)

    @property
    def val_dataset(self) -> Optional[LMDataset]:
        if self._val_texts is None:
            return None
        if self._val_dataset is None:
            from .data.dataset import PackingConfig

            cfg = PackingConfig(max_length=self.max_length, pack_sequences=self.pack_sequences)
            self._val_dataset = LMDataset(self._val_texts, self._tokenizer, cfg)
        return self._val_dataset

    @property
    def tokenizer_backend(self) -> str:
        """Resolved tokenizer backend after auto-detection."""
        return self._tokenizer.backend_name

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._texts)


class psannLM:
    """High-level language model wrapper for PSANN-LM.

    Parameters map 1:1 with the public spec and are persisted for save/load:

    - base: `"waveresnet"` (default), `"respsann"`, or `"sgrpsann"`
    - d_model / n_layers / n_heads / d_mlp: transformer dimensions
    - vocab_size: optional override (defaults to psannLMDataPrep vocab)
    - positional_encoding: `"rope"`, `"alibi"`, or `"sinusoidal"`
    - sine_params: dict or :class:`SineParams` controlling the sine MLPs

    Use :meth:`fit` to attach trainer/data state, :meth:`generate` /
    :meth:`generate_batch` for inference, and :meth:`save` / :meth:`load`
    for checkpointing.
    """

    def __init__(
        self,
        *,
        base: str = "waveresnet",
        d_model: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        d_mlp: Optional[int] = None,
        vocab_size: Optional[int] = None,
        sine_params: Optional[Dict[str, float]] | SineParams = None,
        rope: bool = True,
        positional_encoding: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.base = base
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_mlp = d_mlp
        self.vocab_size = vocab_size
        enc_value = positional_encoding
        if enc_value is None:
            enc_value = "rope" if rope else "sinusoidal"
        self.positional_encoding = normalize_positional_encoding(enc_value)
        self.rope = self.positional_encoding == "rope"  # backwards compatibility flag
        self.config_overrides = dict(kwargs)

        if isinstance(sine_params, dict) or sine_params is None:
            d = sine_params or {}
            self.sine = SineParams(
                amp_init=float(d.get("amp_init", 1.0)),
                freq_init=float(d.get("freq_init", 1.0)),
                damp_init=float(d.get("damp_init", 0.01)),
                trainable=bool(d.get("trainable", True)),
            )
        else:
            self.sine = sine_params

        # Internal placeholders; real modules will be attached later.
        self._model: Optional[nn.Module] = None
        self._trainer: Optional[Trainer] = None
        self._tokenizer: Optional[Tokenizer] = None

    # ----------------------- Internal helpers ----------------------
    def _ensure_model(self, vocab_size: int) -> nn.Module:
        if self._model is not None:
            return self._model
        d_mlp = self.d_mlp if self.d_mlp is not None else 4 * self.d_model
        factory = get_base(self.base)
        sine_cfg = SineConfig(
            amp_init=float(self.sine.amp_init),
            freq_init=float(self.sine.freq_init),
            damp_init=float(self.sine.damp_init),
            trainable=bool(self.sine.trainable),
        )
        self._model = factory(
            vocab_size=vocab_size,
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            d_mlp=d_mlp,
            dropout=float(self.config_overrides.get("dropout", 0.0)),
            positional_encoding=self.positional_encoding,
            mlp_activation=str(self.config_overrides.get("mlp_activation", "sine")),
            sine=sine_cfg,
        )
        return self._model

    # ------------------------- Training API -------------------------
    def fit(
        self,
        train_data: psannLMDataPrep,
        *,
        val_data: Optional[psannLMDataPrep] = None,
        epochs: int = 1,
        batch_tokens: Optional[int] = None,
        lr: Optional[float] = None,
        amp: Optional[str] = None,
        ddp: Optional[str] = None,
        **kwargs: Any,
    ) -> "psannLM":
        """Train the language model on the prepared dataset.

        Parameters
        ----------
        train_data:
            psannLMDataPrep instance providing tokenizer + dataset.
        val_data:
            Optional psannLMDataPrep for validation splits (defaults to the
            built-in val split on ``train_data`` when available).
        epochs, batch_tokens, lr, amp, ddp:
            Passed through to :class:`TrainConfig`.
        **kwargs:
            Trainer overrides such as ``grad_checkpoint``.

        Returns
        -------
        self:
            Enables chaining (`psannLM(...).fit(...).generate(...)`).
        """

        if not isinstance(train_data, psannLMDataPrep):
            raise TypeError("train_data must be psannLMDataPrep")
        vocab = int(train_data.vocab_size if self.vocab_size is None else self.vocab_size)
        model = self._ensure_model(vocab)
        self._trainer = self._trainer or Trainer(
            TrainConfig(
                epochs=int(epochs),
                batch_tokens=int(batch_tokens or 131072),
                lr=float(lr or 2e-4),
                amp=str(amp or "bf16"),
                ddp=str(ddp or "auto"),
                grad_checkpoint=bool(kwargs.get("grad_checkpoint", False)),
            )
        )
        # Update existing trainer config if present and overrides provided
        if self._trainer is not None and ("grad_checkpoint" in kwargs):
            try:
                self._trainer.cfg.grad_checkpoint = bool(kwargs.get("grad_checkpoint"))  # type: ignore[attr-defined]
            except Exception:
                pass
        max_length = int(train_data.max_length)
        self._tokenizer = train_data.tokenizer
        val_ds = None
        if val_data is not None:
            try:
                val_ds = val_data.dataset  # type: ignore[attr-defined]
            except Exception:
                val_ds = None
        elif hasattr(train_data, "val_dataset"):
            val_ds = train_data.val_dataset  # type: ignore[attr-defined]
        self._trainer.train(model, train_data.dataset, max_length=max_length, val_dataset=val_ds)
        return self

    # ------------------------ Inference API ------------------------
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 128,
        top_k: Optional[int] = None,
        top_p: Optional[float] = 0.9,
        temperature: float = 1.0,
        repetition_penalty: Optional[float] = None,
        **kwargs: Any,
    ) -> str:
        """Generate text from one prompt using top-k/top-p sampling."""

        if self._model is None:
            _ = self._ensure_model(int(self.vocab_size or 32000))
        assert self._model is not None
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not available. Call fit() first to attach a tokenizer.")

        from .infer.generate import sample_next_token

        self._model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(device)

        # Encode prompt
        input_ids = self._tokenizer.encode(prompt, add_specials=True)
        context = torch.tensor([input_ids], dtype=torch.long, device=device)

        generated: list[int] = []
        eos_id = int(self._tokenizer.eos_id)
        for _ in range(int(max_new_tokens)):
            with torch.no_grad():
                logits = self._model(context)  # (1,T,V)
                next_logits = logits[:, -1, :]
                # repetition penalty (basic): down-weight seen tokens
                if repetition_penalty and repetition_penalty > 1.0:
                    if generated:
                        idxs = torch.tensor(generated, dtype=torch.long, device=next_logits.device)
                        next_logits.scatter_add_(
                            -1,
                            idxs.view(1, -1),
                            torch.full(
                                (1, len(generated)),
                                -abs(float(repetition_penalty)),
                                device=next_logits.device,
                            ),
                        )
                next_id = sample_next_token(
                    next_logits,
                    temperature=float(temperature),
                    top_k=top_k,
                    top_p=top_p,
                )
            nid = int(next_id.item())
            generated.append(nid)
            context = torch.cat([context, next_id.view(1, 1)], dim=1)
            if nid == eos_id:
                break

        # Decode only the newly generated portion (skip specials)
        out = self._tokenizer.decode(generated, skip_specials=True)
        return out

    def generate_batch(
        self,
        prompts: Sequence[str],
        *,
        max_new_tokens: int = 128,
        top_k: Optional[int] = None,
        top_p: Optional[float] = 0.9,
        temperature: float = 1.0,
        repetition_penalty: Optional[float] = None,
    ) -> list[str]:
        """Generate text for multiple prompts, reusing KV-cache state.

        Prompts with identical lengths share a fast batched path; mixed
        lengths are bucketed automatically to avoid attention masks.
        """
        if self._model is None:
            _ = self._ensure_model(int(self.vocab_size or 32000))
        assert self._model is not None
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not available. Call fit() first to attach a tokenizer.")

        from .infer.generate import sample_next_token

        self._model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(device)

        # Encode prompts
        enc = [self._tokenizer.encode(p, add_specials=True) for p in prompts]
        lengths = [len(e) for e in enc]
        B = len(enc)
        same_len = len(set(lengths)) == 1
        eos_id = int(self._tokenizer.eos_id)
        if not same_len:
            # Bucket by prompt length to avoid padding/masks while still batching
            buckets: dict[int, list[tuple[int, list[int]]]] = {}
            for i, (l, ids) in enumerate(zip(lengths, enc)):
                buckets.setdefault(int(l), []).append((i, ids))
            outputs: list[str] = [""] * B
            for T0, items in sorted(buckets.items(), key=lambda kv: -kv[0]):
                idxs = [i for (i, _) in items]
                enc_batch = [ids for (_, ids) in items]
                context = torch.tensor(enc_batch, dtype=torch.long, device=device)
                with torch.no_grad():
                    logits, past_kvs = self._model(context, use_cache=True)  # type: ignore[call-arg]
                generated_bucket: list[list[int]] = [[] for _ in range(len(items))]
                for _ in range(int(max_new_tokens)):
                    with torch.no_grad():
                        next_logits = logits[:, -1, :]
                        if repetition_penalty and repetition_penalty > 1.0:
                            for b in range(len(items)):
                                if generated_bucket[b]:
                                    idxs_rep = torch.tensor(
                                        generated_bucket[b],
                                        dtype=torch.long,
                                        device=next_logits.device,
                                    )
                                    next_logits[b : b + 1].scatter_add_(
                                        -1,
                                        idxs_rep.view(1, -1),
                                        torch.full(
                                            (1, len(generated_bucket[b])),
                                            -abs(float(repetition_penalty)),
                                            device=next_logits.device,
                                        ),
                                    )
                        next_ids = sample_next_token(
                            next_logits,
                            temperature=float(temperature),
                            top_k=top_k,
                            top_p=top_p,
                        )
                    for b in range(len(items)):
                        nid = int(next_ids[b].item())
                        generated_bucket[b].append(nid)
                    step_tokens = next_ids.view(len(items), 1)
                    with torch.no_grad():
                        logits, past_kvs = self._model(step_tokens, use_cache=True, past_kvs=past_kvs)  # type: ignore[call-arg]
                    if all(g and g[-1] == eos_id for g in generated_bucket):
                        break
                # Decode and scatter back to original indices
                for b, iorig in enumerate(idxs):
                    outputs[iorig] = self._tokenizer.decode(generated_bucket[b], skip_specials=True)
            return outputs

        # Equal-length fast path with KV cache
        context = torch.tensor(enc, dtype=torch.long, device=device)
        with torch.no_grad():
            logits, past_kvs = self._model(context, use_cache=True)  # type: ignore[call-arg]
        generated: list[list[int]] = [[] for _ in range(B)]

        for _ in range(int(max_new_tokens)):
            with torch.no_grad():
                # Sample next token for each batch item from last logits
                next_logits = logits[:, -1, :]
                # Simple per-batch repetition penalty (optional)
                if repetition_penalty and repetition_penalty > 1.0:
                    for b in range(B):
                        if generated[b]:
                            idxs = torch.tensor(
                                generated[b], dtype=torch.long, device=next_logits.device
                            )
                            next_logits[b : b + 1].scatter_add_(
                                -1,
                                idxs.view(1, -1),
                                torch.full(
                                    (1, len(generated[b])),
                                    -abs(float(repetition_penalty)),
                                    device=next_logits.device,
                                ),
                            )
                next_ids = sample_next_token(
                    next_logits,
                    temperature=float(temperature),
                    top_k=top_k,
                    top_p=top_p,
                )  # (B,)
            # Append and step with cache
            for b in range(B):
                nid = int(next_ids[b].item())
                generated[b].append(nid)
            step_tokens = next_ids.view(B, 1)
            with torch.no_grad():
                logits, past_kvs = self._model(step_tokens, use_cache=True, past_kvs=past_kvs)  # type: ignore[call-arg]
            # Early stop if all hit EOS
            if all(g and g[-1] == eos_id for g in generated):
                break

        return [self._tokenizer.decode(g, skip_specials=True) for g in generated]

    # ------------------------ Persistence API ----------------------
    def save(self, path: str) -> None:
        """Serialize model config/state (including sine + positional encoding)."""
        model = self._model or self._ensure_model(int(self.vocab_size or 32000))
        # Record the device where the model currently resides, to help
        # loaders place the model consistently when CUDA is available.
        try:
            dev = str(next(model.parameters()).device)
        except StopIteration:
            dev = "cpu"
        payload = {
            "config": {
                "base": self.base,
                "d_model": self.d_model,
                "n_layers": self.n_layers,
                "n_heads": self.n_heads,
                "d_mlp": self.d_mlp,
                "vocab_size": self.vocab_size,
                "rope": self.rope,
                "positional_encoding": self.positional_encoding,
                "sine": {
                    "amp_init": self.sine.amp_init,
                    "freq_init": self.sine.freq_init,
                    "damp_init": self.sine.damp_init,
                    "trainable": self.sine.trainable,
                },
                "overrides": self.config_overrides,
            },
            "state_dict": model.state_dict(),
            "device": dev,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str) -> "psannLM":
        """Load a psannLM checkpoint produced by :meth:`save`."""
        payload = torch.load(path, map_location="cpu")
        cfg = payload.get("config", {})
        inst = cls(
            base=cfg.get("base", "waveresnet"),
            d_model=cfg.get("d_model", 512),
            n_layers=cfg.get("n_layers", 8),
            n_heads=cfg.get("n_heads", 8),
            d_mlp=cfg.get("d_mlp"),
            vocab_size=cfg.get("vocab_size"),
            sine_params=cfg.get("sine"),
            rope=cfg.get("rope", True),
            positional_encoding=cfg.get("positional_encoding"),
            **cfg.get("overrides", {}),
        )
        model = inst._ensure_model(int(inst.vocab_size or 32000))
        model.load_state_dict(payload["state_dict"])  # type: ignore[index]
        # If original checkpoint reports CUDA and it's available, place model on CUDA
        saved_dev = str(payload.get("device", "cpu")).lower()
        if saved_dev.startswith("cuda") and torch.cuda.is_available():
            model.to(torch.device("cuda"))
        return inst
