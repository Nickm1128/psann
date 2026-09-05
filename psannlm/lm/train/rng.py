"""Portable RNG state for exact continuation of trainer checkpoints."""

from __future__ import annotations

import random
from typing import Any, Mapping, cast

import numpy as np
import torch


def capture_rng() -> dict[str, Any]:
    state = cast(tuple[Any, ...], np.random.get_state(legacy=True))
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python": random.getstate(),
        "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
    }


def restore_rng(state: Mapping[str, Any]) -> None:
    validate_rng(state)
    if not isinstance(state.get("torch"), torch.Tensor):
        raise TypeError("checkpoint.rng.torch must be a tensor.")
    try:
        torch.set_rng_state(state["torch"].cpu())
        random.setstate(state["python"])
        numpy_state = state["numpy"]
        np.random.set_state(
            (numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
        )
        if state["cuda"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])
    except (ValueError, TypeError, RuntimeError, KeyError) as exc:
        raise ValueError(f"checkpoint.rng is invalid: {exc}") from exc


def validate_rng(state: Mapping[str, Any]) -> None:
    """Validate with private generators, leaving the process RNGs untouched."""
    if not isinstance(state, Mapping) or set(state) != {"torch", "cuda", "python", "numpy"}:
        raise ValueError("checkpoint.rng must contain exactly torch, cuda, python, numpy.")
    try:
        value = state["torch"]
        if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1:
            raise ValueError("torch must be a byte vector")
        torch.Generator().set_state(value.cpu())
        cuda = state["cuda"]
        if not isinstance(cuda, list) or any(
            not isinstance(v, torch.Tensor) or v.dtype != torch.uint8 or v.ndim != 1 for v in cuda
        ):
            raise ValueError("cuda must be a list of byte vectors")
        random.Random().setstate(state["python"])
        numpy = state["numpy"]
        if not isinstance(numpy, list) or len(numpy) != 5:
            raise ValueError("numpy must have five state fields")
        np.random.RandomState().set_state(
            (numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:])
        )
    except (ValueError, TypeError, RuntimeError, IndexError) as exc:
        raise ValueError(f"checkpoint.rng is invalid: {exc}") from exc
