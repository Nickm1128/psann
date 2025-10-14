from __future__ import annotations

from typing import Dict, Iterable, List, Optional


class BaseTokenizer:
    """Minimal tokenizer interface for PSANN-LM.

    Expected methods:
    - fit(corpus): build vocabulary
    - encode(text): -> List[int]
    - decode(ids): -> str
    - vocab_size: property
    """

    def fit(self, corpus: Iterable[str]) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> List[int]:  # pragma: no cover - interface
        raise NotImplementedError

    def decode(self, ids: Iterable[int]) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:  # pragma: no cover - interface
        raise NotImplementedError


class SimpleWordTokenizer(BaseTokenizer):
    """Whitespace tokenizer with a learnable vocabulary.

    Special tokens:
    - <PAD>=0, <UNK>=1, <BOS>=2, <EOS>=3
    """

    PAD = "<PAD>"
    UNK = "<UNK>"
    BOS = "<BOS>"
    EOS = "<EOS>"

    def __init__(self, *, lowercase: bool = True, max_vocab: Optional[int] = None) -> None:
        self.lowercase = bool(lowercase)
        self.max_vocab = int(max_vocab) if max_vocab is not None else None
        self._tok2id: Dict[str, int] = {self.PAD: 0, self.UNK: 1, self.BOS: 2, self.EOS: 3}
        self._id2tok: List[str] = [self.PAD, self.UNK, self.BOS, self.EOS]

    def fit(self, corpus: Iterable[str]) -> None:
        from collections import Counter

        cnt = Counter()
        for line in corpus:
            s = line.lower() if self.lowercase else line
            cnt.update(s.split())
        # Reserve indices for specials
        specials = set(self._tok2id.keys())
        words = [w for w, _ in cnt.most_common() if w not in specials]
        if self.max_vocab is not None:
            words = words[: max(0, self.max_vocab - len(self._id2tok))]
        for w in words:
            if w not in self._tok2id:
                self._tok2id[w] = len(self._id2tok)
                self._id2tok.append(w)

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        s = text.lower() if self.lowercase else text
        ids: List[int] = []
        if add_bos:
            ids.append(self._tok2id[self.BOS])
        for w in s.split():
            ids.append(self._tok2id.get(w, self._tok2id[self.UNK]))
        if add_eos:
            ids.append(self._tok2id[self.EOS])
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        toks = []
        for i in ids:
            if i < 0 or i >= len(self._id2tok):
                toks.append(self.UNK)
            else:
                tok = self._id2tok[i]
                if tok in (self.PAD, self.BOS, self.EOS):
                    continue
                toks.append(tok)
        return " ".join(toks)

    @property
    def vocab_size(self) -> int:
        return len(self._id2tok)

