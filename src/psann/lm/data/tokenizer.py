"""Tokenizer plugin interface for PSANN-LM.

Provides a small, dependency-free "simple" backend (char-level) as a
fallback when `backend="auto"`. Adapters for `sentencepiece` and
`tokenizers` are provided and auto-selected when the packages are available.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence


logger = logging.getLogger(__name__)


@dataclass
class TokenizerConfig:
    backend: str = "auto"  # "auto" | "simple" | "sentencepiece" | "tokenizers"
    vocab_size: int = 32000  # upper bound for learned vocab (where applicable)
    model_path: Optional[str] = None  # load prebuilt model if provided
    special_tokens_map_path: Optional[str] = None  # optional HF special tokens map path
    hf_passthrough_ids: bool = False  # if True, expose raw HF ids (no +4 remap)
    min_frequency: int = 2  # for BPE tokenizers
    # SentencePiece options
    sp_model_type: str = "unigram"  # "unigram" | "bpe"
    sp_character_coverage: float = 1.0
    sp_input_sentence_size: int = 0  # 0 = all
    sp_shuffle_input_sentence: bool = False


class SimpleCharTokenizer:
    """A tiny char-level tokenizer with special tokens.

    - pad: 0, bos: 1, eos: 2, unk: 3
    - chars start from 4
    - fit() builds vocabulary from provided texts
    """

    PAD = 0
    BOS = 1
    EOS = 2
    UNK = 3

    def __init__(self) -> None:
        self.stoi: Dict[str, int] = {}
        self.itos: List[str] = []
        self._fitted = False

    @property
    def pad_id(self) -> int:
        return self.PAD

    @property
    def bos_id(self) -> int:
        return self.BOS

    @property
    def eos_id(self) -> int:
        return self.EOS

    @property
    def unk_id(self) -> int:
        return self.UNK

    @property
    def vocab_size(self) -> int:
        # 0..3 reserved + len(chars)
        return 4 + len(self.itos)

    def fit(self, texts: Iterable[str]) -> None:
        charset = []
        seen = set()
        for t in texts:
            for ch in t:
                if ch not in seen:
                    seen.add(ch)
                    charset.append(ch)
        self.itos = charset
        self.stoi = {ch: 4 + i for i, ch in enumerate(self.itos)}
        self._fitted = True

    def encode(self, text: str, add_specials: bool = True) -> List[int]:
        if not self._fitted:
            raise RuntimeError("SimpleCharTokenizer must be fit() before encode().")
        ids = [self.stoi.get(ch, self.UNK) for ch in text]
        if add_specials:
            ids = [self.BOS] + ids + [self.EOS]
        return ids

    def decode(self, ids: Sequence[int], skip_specials: bool = True) -> str:
        out = []
        for i in ids:
            if skip_specials and i in (self.PAD, self.BOS, self.EOS):
                continue
            if i >= 4:
                idx = i - 4
                if 0 <= idx < len(self.itos):
                    out.append(self.itos[idx])
                else:
                    out.append("?")
            else:
                out.append("?")
        return "".join(out)


class Tokenizer:
    """Tokenizer facade with pluggable backends.

    Backends:
      - "simple" (default fallback for "auto"): small char-level tokenizer
      - "sentencepiece" (preferred when installed)
      - "tokenizers" (Hugging Face tokenizers)
    """

    _AUTO_ORDER = ("sentencepiece", "tokenizers")

    def __init__(self, cfg: TokenizerConfig = TokenizerConfig()) -> None:
        self.cfg = cfg
        backend = (cfg.backend or "auto").lower()
        if backend == "auto":
            impl, resolved = self._select_auto_backend(cfg)
        else:
            impl = self._instantiate_backend(backend, cfg)
            resolved = backend
        self._impl = impl
        self._selected_backend = resolved

    def _instantiate_backend(self, backend: str, cfg: TokenizerConfig):
        if backend == "simple":
            return SimpleCharTokenizer()
        if backend == "sentencepiece":
            return _make_sentencepiece_tokenizer(cfg)
        if backend == "tokenizers":
            return _make_hf_tokenizers(cfg)
        raise NotImplementedError(f"Tokenizer backend '{backend}' is not available yet")

    def _select_auto_backend(self, cfg: TokenizerConfig):
        errors: Dict[str, str] = {}
        for candidate in self._AUTO_ORDER:
            try:
                impl = self._instantiate_backend(candidate, cfg)
                logger.info("Tokenizer(auto): selected '%s' backend", candidate)
                return impl, candidate
            except ImportError as exc:
                errors[candidate] = str(exc)
        if errors:
            details = "; ".join(f"{name}: {msg}" for name, msg in errors.items())
            logger.warning(
                "Tokenizer(auto): sentencepiece/tokenizers unavailable (%s); "
                "falling back to simple char-level tokenizer",
                details,
            )
        return SimpleCharTokenizer(), "simple"

    @property
    def vocab_size(self) -> int:
        return self._impl.vocab_size

    @property
    def pad_id(self) -> int:
        return self._impl.pad_id

    @property
    def bos_id(self) -> int:
        return self._impl.bos_id

    @property
    def eos_id(self) -> int:
        return self._impl.eos_id

    @property
    def unk_id(self) -> int:
        return self._impl.unk_id

    @property
    def backend_name(self) -> str:
        """Resolved backend name (auto selection included)."""
        return self._selected_backend

    def fit(self, texts: Iterable[str]) -> None:
        self._impl.fit(texts)

    def encode(self, text: str, add_specials: bool = True) -> List[int]:
        return self._impl.encode(text, add_specials=add_specials)

    def decode(self, ids: Sequence[int], skip_specials: bool = True) -> str:
        return self._impl.decode(ids, skip_specials=skip_specials)


# ----------------------- SentencePiece backend -----------------------

def _make_sentencepiece_tokenizer(cfg: TokenizerConfig):
    try:
        import sentencepiece as spm  # type: ignore
    except Exception as e:
        raise ImportError(
            "Tokenizer backend 'sentencepiece' requires the 'sentencepiece' package.\n"
            "Install with: pip install 'psann[lm]' or 'sentencepiece'"
        ) from e

    class SentencePieceTokenizer:
        PAD = 0
        BOS = 1
        EOS = 2
        UNK = 3

        def __init__(self, cfg: TokenizerConfig) -> None:
            self.cfg = cfg
            self.sp: Optional[spm.SentencePieceProcessor] = None

        @property
        def pad_id(self) -> int:
            return self.PAD

        @property
        def bos_id(self) -> int:
            return self.BOS

        @property
        def eos_id(self) -> int:
            return self.EOS

        @property
        def unk_id(self) -> int:
            return self.UNK

        @property
        def vocab_size(self) -> int:
            if self.sp is None:
                return int(self.cfg.vocab_size)
            return int(self.sp.get_piece_size())

        def fit(self, texts: Iterable[str]) -> None:
            # Load prebuilt model if provided
            if self.cfg.model_path:
                sp = spm.SentencePieceProcessor()
                sp.load(self.cfg.model_path)
                self.sp = sp
                return
            from tempfile import NamedTemporaryFile
            import os

            with NamedTemporaryFile("w", delete=False, encoding="utf-8") as fh:
                for t in texts:
                    if t and t.strip():
                        fh.write(t.replace("\n", " ") + "\n")
                corpus_path = fh.name

            mp = NamedTemporaryFile(delete=False)
            model_prefix = mp.name
            mp.close()

            try:
                sp_args = dict(
                    input=corpus_path,
                    model_prefix=model_prefix,
                    vocab_size=int(self.cfg.vocab_size),
                    model_type=str(self.cfg.sp_model_type),
                    character_coverage=float(self.cfg.sp_character_coverage),
                    bos_id=self.BOS,
                    eos_id=self.EOS,
                    unk_id=self.UNK,
                    pad_id=self.PAD,
                    hard_vocab_limit=False,
                    shuffle_input_sentence=bool(self.cfg.sp_shuffle_input_sentence),
                )
                iss = int(self.cfg.sp_input_sentence_size)
                if iss > 0:
                    sp_args["input_sentence_size"] = iss
                spm.SentencePieceTrainer.Train(**sp_args)
                model_path = model_prefix + ".model"
                sp = spm.SentencePieceProcessor()
                sp.load(model_path)
                self.sp = sp
            finally:
                for ext in ("", ".model", ".vocab"):
                    try:
                        os.remove(model_prefix + ext)
                    except Exception:
                        pass
                try:
                    os.remove(corpus_path)
                except Exception:
                    pass

        def encode(self, text: str, add_specials: bool = True) -> List[int]:
            if self.sp is None:
                raise RuntimeError("SentencePieceTokenizer must be fit() before encode().")
            ids = list(self.sp.encode(text, out_type=int))
            if add_specials:
                ids = [self.BOS] + ids + [self.EOS]
            return ids

        def decode(self, ids: Sequence[int], skip_specials: bool = True) -> str:
            if self.sp is None:
                raise RuntimeError("SentencePieceTokenizer must be fit() before decode().")
            if skip_specials:
                ids = [i for i in ids if i not in (self.PAD, self.BOS, self.EOS)]
            return str(self.sp.decode(ids))

    return SentencePieceTokenizer(cfg)


# --------------------- HuggingFace tokenizers backend ---------------------

def _make_hf_tokenizers(cfg: TokenizerConfig):
    try:
        from tokenizers import Tokenizer as HFTokenizer  # type: ignore
        from tokenizers import models, trainers, pre_tokenizers, normalizers
    except Exception as e:
        raise ImportError(
            "Tokenizer backend 'tokenizers' requires the 'tokenizers' package.\n"
            "Install with: pip install 'psann[lm]' or 'tokenizers'"
        ) from e

    class HFTokenizersWrapper:
        PAD = 0
        BOS = 1
        EOS = 2
        UNK = 3

        def __init__(self, cfg: TokenizerConfig) -> None:
            self.cfg = cfg
            self.tk: Optional[HFTokenizer] = None
            self._ids: Dict[str, int] = {}
            self._src_special_ids: Dict[str, int] = {}

        @property
        def pad_id(self) -> int:
            return self.PAD

        @property
        def bos_id(self) -> int:
            return self.BOS

        @property
        def eos_id(self) -> int:
            return self.EOS

        @property
        def unk_id(self) -> int:
            return self.UNK

        @property
        def vocab_size(self) -> int:
            if self.tk is None:
                return int(self.cfg.vocab_size)
            if self.cfg.hf_passthrough_ids:
                return int(self.tk.get_vocab_size())
            # Reserve 0..3 for fixed specials; shift others by +4
            return 4 + int(self.tk.get_vocab_size())

        def _normalize_special_entry(self, value) -> Optional[str]:
            if isinstance(value, str):
                return value
            if isinstance(value, dict):
                return value.get("content") or value.get("token")
            return None

        def _load_special_token_strings(self) -> Dict[str, str]:
            mapping: Dict[str, str] = {}
            path = self.cfg.special_tokens_map_path
            if not path:
                return mapping
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    raw = json.load(fh)
                for key in ("pad_token", "bos_token", "eos_token", "unk_token"):
                    if key in raw:
                        normalized = self._normalize_special_entry(raw[key])
                        if normalized:
                            mapping[key] = normalized
            except Exception as exc:  # pragma: no cover - best-effort
                logger.warning("Failed to read special tokens map '%s': %s", path, exc)
            return mapping

        def _resolve_special_id(
            self,
            tk: HFTokenizer,
            key: str,
            provided: Dict[str, str],
            fallback_tokens: List[str],
            default: int,
        ) -> int:
            candidates: List[Optional[str]] = []
            if key in provided:
                candidates.append(provided[key])
            candidates.extend(fallback_tokens)
            for token in candidates:
                if not token:
                    continue
                tid = tk.token_to_id(token)
                if tid is not None:
                    return int(tid)
            return int(default)

        def _configure_special_ids(self, tk: HFTokenizer) -> None:
            provided = self._load_special_token_strings()
            pad_src = self._resolve_special_id(tk, "pad_token", provided, ["[PAD]", "<pad>", "<PAD>", "pad"], self.PAD)
            bos_src = self._resolve_special_id(
                tk, "bos_token", provided, ["[BOS]", "<s>", "<BOS>", "bos", "<bos>"], self.BOS
            )
            eos_src = self._resolve_special_id(
                tk, "eos_token", provided, ["[EOS]", "</s>", "<EOS>", "eos", "<eos>"], self.EOS
            )
            unk_src = self._resolve_special_id(tk, "unk_token", provided, ["[UNK]", "<unk>", "<UNK>", "unk"], self.UNK)
            self._ids = {
                "[PAD]": pad_src,
                "[BOS]": bos_src,
                "[EOS]": eos_src,
                "[UNK]": unk_src,
            }
            self._src_special_ids = {
                "pad": pad_src,
                "bos": bos_src,
                "eos": eos_src,
                "unk": unk_src,
            }

        def fit(self, texts: Iterable[str]) -> None:
            # Load from JSON if provided
            if self.cfg.model_path:
                tk = HFTokenizer.from_file(self.cfg.model_path)
                self.tk = tk
                self._configure_special_ids(tk)
                return
            # Train a BPE model with basic whitespace pre-tokenization
            model = models.BPE(unk_token="[UNK]")
            tk = HFTokenizer(model)
            tk.normalizer = normalizers.Sequence([normalizers.NFKC()])
            tk.pre_tokenizer = pre_tokenizers.Whitespace()
            trainer = trainers.BpeTrainer(
                vocab_size=int(self.cfg.vocab_size),
                min_frequency=int(self.cfg.min_frequency),
                special_tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]"],
            )
            # Train from in-memory iterator by writing to temp file for simplicity
            from tempfile import NamedTemporaryFile
            import os

            with NamedTemporaryFile("w", delete=False, encoding="utf-8") as fh:
                for t in texts:
                    if t and t.strip():
                        fh.write(t.replace("\n", " ") + "\n")
                corpus_path = fh.name
            try:
                tk.train([corpus_path], trainer=trainer)
            finally:
                try:
                    os.remove(corpus_path)
                except Exception:
                    pass
            self._configure_special_ids(tk)
            # Ensure our fixed ids map; if not aligned, add a decoder shim
            # For simplicity, we will keep wrapper ids fixed at 0..3 and remap in encode/decode.
            self.tk = tk

        def _ensure(self) -> HFTokenizer:
            if self.tk is None:
                raise RuntimeError("HFTokenizersWrapper must be fit() before encode()/decode().")
            return self.tk

        def encode(self, text: str, add_specials: bool = True) -> List[int]:
            tk = self._ensure()
            out = tk.encode(text)
            ids = [int(i) for i in out.ids]
            if self.cfg.hf_passthrough_ids:
                # Use raw token ids; add actual BOS/EOS ids if requested
                if add_specials:
                    bos = self._src_special_ids.get("bos", self.BOS)
                    eos = self._src_special_ids.get("eos", self.EOS)
                    ids = ([bos] if bos is not None else []) + ids + ([eos] if eos is not None else [])
                return ids

            # Remapped fixed ids path
            pad_id = self._ids.get("[PAD]", self.PAD)
            bos_id = self._ids.get("[BOS]", self.BOS)
            eos_id = self._ids.get("[EOS]", self.EOS)
            unk_id = self._ids.get("[UNK]", self.UNK)

            def _map(i: int) -> int:
                if i == pad_id:
                    return self.PAD
                if i == bos_id:
                    return self.BOS
                if i == eos_id:
                    return self.EOS
                if i == unk_id:
                    return self.UNK
                return i + 4

            ids = [_map(i) for i in ids]
            if add_specials:
                ids = [self.BOS] + ids + [self.EOS]
            return ids

        def decode(self, ids: Sequence[int], skip_specials: bool = True) -> str:
            tk = self._ensure()
            if self.cfg.hf_passthrough_ids:
                # Remove actual special ids if requested
                if skip_specials:
                    specials = {
                        v for v in self._src_special_ids.values() if v is not None
                    }
                    out_ids = [int(i) for i in ids if int(i) not in specials]
                else:
                    out_ids = [int(i) for i in ids]
                return tk.decode(out_ids)

            # Remapped path: remove fixed specials and unshift by 4
            out_ids: List[int] = []
            for i in ids:
                if skip_specials and i in (self.PAD, self.BOS, self.EOS):
                    continue
                if i >= 4:
                    out_ids.append(int(i) - 4)
            return tk.decode(out_ids)

    return HFTokenizersWrapper(cfg)


