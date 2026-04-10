# ruff: noqa: F403,F405
from __future__ import annotations

from .shared import *


def _detect_text_field(dataset) -> str:
    sample = None
    if hasattr(dataset, "take"):
        iterator = dataset.take(1)
        sample = next(iter(iterator), None)
    if sample is None:
        try:
            sample = dataset[0]
        except Exception:
            pass
    if sample is None:
        raise RuntimeError("Unable to inspect dataset schema for text field detection.")
    row = dict(sample)
    if "text" in row and isinstance(row["text"], str):
        field = "text"
        log_progress(f"Detected default text field '{field}'.")
        return field
    for key in ("content", "article", "document", "body"):
        if key in row and isinstance(row[key], str):
            log_progress(f"Detected fallback text field '{key}'.")
            return key
    for key, value in row.items():
        if isinstance(value, str):
            log_progress(f"Detected inferred text field '{key}'.")
            return key
    raise RuntimeError("Could not find any string field in dataset sample.")


def _normalize_row_text(row: dict, text_field: str) -> Optional[str]:
    if text_field in row and isinstance(row[text_field], str):
        text = row[text_field].strip()
        if text:
            return text
    for value in row.values():
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


class TextStream:
    """Re-iterable text iterator with optional shuffling."""

    def __init__(
        self,
        dataset,
        text_field: str,
        *,
        streaming: bool,
        shuffle: bool,
        seed: int,
        shuffle_buffer: int,
    ) -> None:
        self.dataset = dataset
        self.text_field = text_field
        self.streaming = streaming
        self.shuffle = shuffle
        self.seed = seed
        self.shuffle_buffer = max(1_000, int(shuffle_buffer)) if shuffle else 0
        self._epoch = 0
        log_progress(
            f"TextStream init -> streaming={self.streaming} shuffle={self.shuffle} buffer={self.shuffle_buffer}"
        )

    def __iter__(self) -> Iterator[str]:
        while True:
            ds = self.dataset
            if self.shuffle and hasattr(ds, "shuffle"):
                shuffle_seed = self.seed + self._epoch
                if self.streaming:
                    ds = ds.shuffle(seed=shuffle_seed, buffer_size=self.shuffle_buffer)
                else:
                    ds = ds.shuffle(seed=shuffle_seed)
            self._epoch += 1
            for row in ds:
                text = _normalize_row_text(dict(row), self.text_field)
                if text:
                    yield text


class SequenceBatcher:
    """Packs contiguous token streams into fixed-length batches."""

    def __init__(
        self,
        text_stream: TextStream,
        tokenizer,
        *,
        seq_len: int,
        micro_batch_size: int,
    ) -> None:
        from collections import deque

        self.stream = text_stream
        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        self.window = self.seq_len + 1
        self.micro_batch = max(1, int(micro_batch_size))
        self._buffer = deque()  # type: ignore[var-annotated]
        self._iter = iter(self.stream)
        log_progress(
            f"SequenceBatcher init -> seq_len={self.seq_len} micro_batch={self.micro_batch}"
        )

    def reset(self) -> None:
        self._buffer.clear()
        self._iter = iter(self.stream)

    def _ensure_tokens(self) -> None:
        while len(self._buffer) < self.window:
            try:
                text = next(self._iter)
            except StopIteration:
                self._iter = iter(self.stream)
                continue
            ids = self.tokenizer.encode(
                text,
                add_special_tokens=True,
            )
            if len(ids) < 2:
                continue
            self._buffer.extend(ids)

    def _next_sequence(self) -> Tuple[List[int], List[int]]:
        self._ensure_tokens()
        chunk = [self._buffer.popleft() for _ in range(self.window)]
        if not self._buffer:
            self._buffer.append(chunk[-1])
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs: List[List[int]] = []
        labels: List[List[int]] = []
        for _ in range(self.micro_batch):
            x, y = self._next_sequence()
            inputs.append(x)
            labels.append(y)
        input_tensor = torch.tensor(inputs, dtype=torch.long)
        label_tensor = torch.tensor(labels, dtype=torch.long)
        return input_tensor, label_tensor


def prepare_tokenizer(name_or_path: str, seq_len: int):
    log_progress(f"Preparing tokenizer '{name_or_path}' (seq_len={seq_len})")
    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "<|eos|>"})
    if tokenizer.bos_token is None:
        tokenizer.add_special_tokens({"bos_token": "<|bos|>"})
    tokenizer.model_max_length = seq_len + 1
    tokenizer.truncation_side = "right"
    tokenizer.padding_side = "right"
    log_progress(
        f"Tokenizer ready -> vocab_size={tokenizer.vocab_size} pad={tokenizer.pad_token} eos={tokenizer.eos_token}"
    )
    return tokenizer
