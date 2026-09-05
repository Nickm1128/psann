"""Canonical high-level language modeling, data preparation and 0.x adapters."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Optional, Sequence

import torch
from torch import nn

from .config import TrainConfig, normalize_train_config
from ..architectures import LMConfig, build_lm_model, normalize_lm_config
from ..architectures.compat import compatibility_warning, legacy_lm_config
from .data.dataset import LMDataset
from .data.tokenizer import Tokenizer, TokenizerConfig
from .train.trainer import LMTrainer


@dataclass
class SineParams:
    """Parameters controlling the parametric sine used in MLP blocks.

    This is a convenience container for the public API. Internal modules
    may represent and constrain these differently.
    """

    amp_init: float = 1.0
    amp_init_std: float = 0.0
    freq_init: float = 1.0
    freq_init_std: float = 0.0
    damp_init: float = 0.01
    damp_init_std: float = 0.0
    trainable: bool = True


class PSANNLMDataPrep:
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


class PSANNLM:
    """A config-first language model with immutable construction and training policies.

    Pass an LMConfig or strict tagged mapping. Vocabulary may be resolved by fit().
    A supplied device selects execution; omission retains automatic CPU/CUDA selection.
    Use TrainConfig for fitting, attach_tokenizer() for standalone loaded models, and
    save()/load() for portable versioned model artifacts.
    """

    def __init__(
        self,
        *,
        config: LMConfig | Mapping[str, Any],
        device: str | torch.device | None = None,
        **flat: Any,
    ) -> None:
        self.config = normalize_lm_config(config, **flat)
        self._device = torch.device(device) if device is not None else None
        self._model: nn.Module | None = None
        self._trainer: LMTrainer | None = None
        self._tokenizer: Tokenizer | None = None

    @property
    def base(self) -> str:
        architecture = self.config.architecture
        return {
            "transformer": "transformer",
            "residual": "sgrpsann" if architecture.spectral is not None else "respsann",
            "wave": "waveresnet",
            "geometric-sparse": "geosparse",
        }[architecture.kind]

    @property
    def vocab_size(self) -> int | None:
        return self.config.vocab_size

    @property
    def d_model(self) -> int:
        return self.config.d_model

    @property
    def n_layers(self) -> int:
        return self.config.n_layers

    @property
    def n_heads(self) -> int:
        return self.config.n_heads

    @property
    def d_mlp(self) -> int | None:
        return self.config.d_mlp

    @property
    def positional_encoding(self) -> str:
        return self.config.positional_encoding

    @property
    def rope(self) -> bool:
        return self.positional_encoding == "rope"

    def _ensure_model(self, vocab_size: int) -> nn.Module:
        if self.config.vocab_size is not None and self.config.vocab_size != vocab_size:
            raise ValueError("vocab_size conflicts with config.vocab_size.")
        if self._model is None:
            config = replace(self.config, vocab_size=vocab_size)
            self._model = build_lm_model(config).model
            self.config = normalize_lm_config(config, for_build=True)
            if self._device is not None:
                self._model.to(self._device)
        elif self.config.vocab_size != vocab_size:
            raise ValueError("vocab_size conflicts with the constructed model.")
        return self._model

    def attach_tokenizer(self, tokenizer: Tokenizer) -> PSANNLM:
        """Attach an already fitted tokenizer without fitting or changing model state."""
        if not isinstance(tokenizer, Tokenizer):
            raise TypeError("tokenizer must be a fitted Tokenizer.")
        self._tokenizer = tokenizer
        return self

    def fit(
        self,
        train_data: PSANNLMDataPrep,
        *,
        train: TrainConfig | Mapping[str, Any] | None = None,
        val_data: PSANNLMDataPrep | None = None,
        resume_checkpoint: str | None = None,
        **flat: Any,
    ) -> PSANNLM:
        """Fit using TrainConfig; flat 0.x fit values remain a warning adapter."""
        if not isinstance(train_data, PSANNLMDataPrep):
            raise TypeError("train_data must be PSANNLMDataPrep.")
        if train is not None:
            training = normalize_train_config(train)
            if flat:
                # Validate duplicates with the same strict policy boundary as all
                # other canonical training input (True must not compare as 1).
                from dataclasses import fields

                values = {field.name: getattr(training, field.name) for field in fields(training)}
                for name in flat:
                    if name not in values:
                        raise ValueError(f"flat.{name} conflicts with train.{name}.")
                duplicated = normalize_train_config(dict(values, **flat))
                for name in flat:
                    if getattr(training, name) != getattr(duplicated, name):
                        raise ValueError(f"flat.{name} conflicts with train.{name}.")
            if flat:
                compatibility_warning(
                    "Flat fit arguments are deprecated; matching values were normalized once."
                )
        elif flat:
            active = {"epochs", "batch_tokens", "lr", "amp", "ddp", "grad_checkpoint"}
            ignored = sorted(set(flat) - active)
            compatibility_warning(
                "Flat fit arguments are deprecated; use train=TrainConfig. "
                + ("Inactive legacy arguments ignored: " + ", ".join(ignored) if ignored else "")
            )
            if self._trainer is not None:
                training = (
                    replace(self._trainer.cfg, grad_checkpoint=bool(flat["grad_checkpoint"]))
                    if "grad_checkpoint" in flat
                    else self._trainer.cfg
                )
            else:
                values = dict(epochs=1, batch_tokens=131072, lr=2e-4, amp="bf16", ddp="auto")
                values.update(
                    {
                        name: value
                        for name, value in flat.items()
                        if name in active and value is not None
                    }
                )
                training = normalize_train_config(values)
        else:
            training = TrainConfig()
        vocab = train_data.vocab_size if self.vocab_size is None else self.vocab_size
        model = self._ensure_model(vocab)
        self.attach_tokenizer(train_data.tokenizer)
        if self._trainer is None:
            self._trainer = LMTrainer(training)
        else:
            self._trainer.cfg = training
        val_ds = val_data.dataset if val_data is not None else train_data.val_dataset
        self._trainer.train(
            model,
            train_data.dataset,
            max_length=train_data.max_length,
            val_dataset=val_ds,
            resume_checkpoint=resume_checkpoint,
            device=self._device,
        )
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
        device = self._device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        device = self._device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
                            repetition_ids = torch.tensor(
                                generated[b], dtype=torch.long, device=next_logits.device
                            )
                            next_logits[b : b + 1].scatter_add_(
                                -1,
                                repetition_ids.view(1, -1),
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

    def save(self, path: str) -> None:
        """Save strict schema-v1 config, model state, device and fitted tokenizer."""
        from ..persistence import model_payload

        if self.vocab_size is None:
            raise ValueError(
                "vocab_size must be resolved before save; fit or supply LMConfig.vocab_size."
            )
        model = self._model or self._ensure_model(self.vocab_size)
        torch.save(model_payload(model, self._tokenizer), path)

    @classmethod
    def load(cls, path: str, *, map_location: str | torch.device | None = None) -> PSANNLM:
        """Load a model artifact, including 0.x files; trainer files use load_lm_checkpoint."""
        from ..persistence import load_lm_checkpoint

        loaded = load_lm_checkpoint(path, map_location=map_location, require_model=True)
        inst = cls(config=loaded.config, device=next(loaded.model.parameters()).device)
        inst._model = loaded.model
        inst._tokenizer = loaded.tokenizer
        return inst


class psannLM(PSANNLM):
    """Deprecated 0.x flat constructor; prefer PSANNLM(config=...)."""

    def __init__(
        self,
        *,
        config: LMConfig | Mapping[str, Any] | None = None,
        device: str | torch.device | None = None,
        **flat: Any,
    ) -> None:
        if config is None:
            base = flat.pop("base", "waveresnet")
            # The high-level adapter retains the exact historically forwarded subset.
            normalized = legacy_lm_config(base, flat, high_level=True, warn=True)
        else:
            normalized = normalize_lm_config(config, **flat)
            if not flat:
                compatibility_warning("psannLM is deprecated; use PSANNLM(config=...).")
        super().__init__(config=normalized, device=device)

    def fit(self, train_data: PSANNLMDataPrep, **kwargs: Any) -> psannLM:
        if "train" not in kwargs and not any(
            key in kwargs
            for key in ("epochs", "batch_tokens", "lr", "amp", "ddp", "grad_checkpoint")
        ):
            kwargs["epochs"] = 1
        super().fit(train_data, **kwargs)
        return self

    def save(self, path: str) -> None:
        if self.vocab_size is None:
            self._ensure_model(32000)
        super().save(path)


class psannLMDataPrep(PSANNLMDataPrep):
    """Deprecated 0.x spelling of PSANNLMDataPrep."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        compatibility_warning("psannLMDataPrep is deprecated; use PSANNLMDataPrep.")
        super().__init__(*args, **kwargs)
