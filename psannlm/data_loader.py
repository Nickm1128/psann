"""Torch IterableDataset wrapper for HF streaming token iterators."""

from __future__ import annotations

from typing import Callable, Dict, Iterator, Any, Optional

import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info


class TokenStreamDataset(IterableDataset):
    """Wrap a callable iterator_fn -> iterator for DataLoader consumption."""

    def __init__(self, iterator_fn: Callable[[Optional[Any]], Iterator[Dict[str, Any]]]):
        super().__init__()
        self.iterator_fn = iterator_fn

    def __iter__(self):
        worker_info = get_worker_info()

        def _callable():
            try:
                return self.iterator_fn(worker_info)
            except TypeError:
                # Backward-compat: support iterator_fn with no params
                return self.iterator_fn()

        yield from _callable()


def build_stream_loader(
    iterator_fn: Callable[[Optional[Any]], Iterator[Dict[str, Any]]],
    *,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
) -> DataLoader:
    """Construct a DataLoader around TokenStreamDataset."""

    dataset = TokenStreamDataset(iterator_fn)
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        num_workers=max(0, int(num_workers)),
        pin_memory=bool(pin_memory),
    )
