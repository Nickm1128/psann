"""Tokenizer plugin interface for PSANN-LM.

Provides a small, dependency-free "simple" backend (char-level) as a
default when `backend="auto"`. Later, adapters for `sentencepiece` and
`tokenizers` can be added while keeping the same façade.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Optional


@dataclass
class TokenizerConfig:
    backend: str = "auto"  # "auto" | "simple" | "sentencepiece" | "tokenizers"
    vocab_size: int = 32000  # upper bound for learned vocab (where applicable)
    model_path: Optional[str] = None  # load prebuilt model if provided
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
    """Tokenizer façade with pluggable backends.

    Backends:
      - "simple" (default for "auto"): small char-level tokenizer
      - "sentencepiece" / "tokenizers": to be implemented
    """

    def __init__(self, cfg: TokenizerConfig = TokenizerConfig()) -> None:
        self.cfg = cfg
        backend = (cfg.backend or "auto").lower()
        if backend == "simple":
            self._impl = SimpleCharTokenizer()
        elif backend == "sentencepiece":
            self._impl = _make_sentencepiece_tokenizer(cfg)
        elif backend == "tokenizers":
            self._impl = _make_hf_tokenizers(cfg)
        elif backend == "auto":
            try:
                self._impl = _make_sentencepiece_tokenizer(cfg)
            except Exception:
                try:
                    self._impl = _make_hf_tokenizers(cfg)
                except Exception:
                    self._impl = SimpleCharTokenizer()
        else:
            raise NotImplementedError(f"Tokenizer backend '{backend}' is not available yet")

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
                spm.SentencePieceTrainer.Train(
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
                    input_sentence_size=int(self.cfg.sp_input_sentence_size) if int(self.cfg.sp_input_sentence_size) > 0 else None,
                    shuffle_input_sentence=bool(self.cfg.sp_shuffle_input_sentence),
                )
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
            # Reserve 0..3 for fixed specials; shift others by +4
            return 4 + int(self.tk.get_vocab_size())

        def fit(self, texts: Iterable[str]) -> None:
            # Load from JSON if provided
            if self.cfg.model_path:
                tk = HFTokenizer.from_file(self.cfg.model_path)
                self.tk = tk
                # Map special token ids
                self._ids = {
                    "[PAD]": int(tk.token_to_id("[PAD]")) if tk.token_to_id("[PAD]") is not None else 0,
                    "[BOS]": int(tk.token_to_id("[BOS]")) if tk.token_to_id("[BOS]") is not None else 1,
                    "[EOS]": int(tk.token_to_id("[EOS]")) if tk.token_to_id("[EOS]") is not None else 2,
                    "[UNK]": int(tk.token_to_id("[UNK]")) if tk.token_to_id("[UNK]") is not None else 3,
                }
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
            # Map special token ids
            self._ids = {
                "[PAD]": int(tk.token_to_id("[PAD]")),
                "[BOS]": int(tk.token_to_id("[BOS]")),
                "[EOS]": int(tk.token_to_id("[EOS]")),
                "[UNK]": int(tk.token_to_id("[UNK]")),
            }
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
            # Map to fixed ids for PAD/BOS/EOS/UNK if needed
            pad_id = self._ids.get("[PAD]", self.PAD)
            bos_id = self._ids.get("[BOS]", self.BOS)
            eos_id = self._ids.get("[EOS]", self.EOS)
            unk_id = self._ids.get("[UNK]", self.UNK)
            # We expose fixed ids (0..3). Shift other ids by +4 if they collide.
            def _map(i: int) -> int:
                if i == pad_id:
                    return self.PAD
                if i == bos_id:
                    return self.BOS
                if i == eos_id:
                    return self.EOS
                if i == unk_id:
                    return self.UNK
                return i + 4  # reserve 0..3

            ids = [_map(i) for i in ids]
            if add_specials:
                ids = [self.BOS] + ids + [self.EOS]
            return ids

        def decode(self, ids: Sequence[int], skip_specials: bool = True) -> str:
            tk = self._ensure()
            # inverse mapping (naive): remove specials and subtract 4 where applicable
            out_ids: List[int] = []
            for i in ids:
                if skip_specials and i in (self.PAD, self.BOS, self.EOS):
                    continue
                if i >= 4:
                    out_ids.append(i - 4)
            return tk.decode(out_ids)

    return HFTokenizersWrapper(cfg)
