from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Optional, Sequence

TrainingEventName = Literal[
    "train_start",
    "fallback",
    "epoch_start",
    "nonfinite_step",
    "validation_end",
    "checkpoint",
    "epoch_end",
    "early_stop",
    "failure",
    "train_end",
]
CallbackErrorPolicy = Literal["raise", "warn"]


@dataclass(frozen=True)
class TrainingEvent:
    """Structured notification emitted by the supervised training loop."""

    name: TrainingEventName
    timestamp: float = field(default_factory=time.time)
    epoch: Optional[int] = None
    step: Optional[int] = None
    data: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "timestamp": float(self.timestamp),
            "epoch": self.epoch,
            "step": self.step,
            "data": dict(self.data),
        }


TrainingEventCallback = Callable[[TrainingEvent], None]


class TrainingEventDispatcher:
    """Deliver training events to callbacks and an optional standard logger."""

    def __init__(
        self,
        callbacks: Optional[Sequence[TrainingEventCallback]] = None,
        *,
        logger: Optional[logging.Logger] = None,
        callback_error_policy: CallbackErrorPolicy = "raise",
    ) -> None:
        if callback_error_policy not in {"raise", "warn"}:
            raise ValueError("callback_error_policy must be 'raise' or 'warn'.")
        self.callbacks = tuple(callbacks or ())
        self.logger = logger
        self.callback_error_policy = callback_error_policy

    def emit(
        self,
        name: TrainingEventName,
        *,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        data: Optional[Mapping[str, Any]] = None,
        suppress_callback_errors: bool = False,
    ) -> TrainingEvent:
        event = TrainingEvent(
            name=name,
            epoch=epoch,
            step=step,
            data=dict(data or {}),
        )
        if self.logger is not None:
            level = logging.ERROR if name == "failure" else logging.INFO
            self.logger.log(
                level,
                "psann.training.%s",
                name,
                extra={"psann_event": event.as_dict()},
            )
        for callback in self.callbacks:
            try:
                callback(event)
            except Exception as exc:
                if suppress_callback_errors:
                    continue
                if self.callback_error_policy == "warn":
                    warnings.warn(
                        f"Training event callback failed for {name!r}: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    continue
                raise
        return event


__all__ = [
    "CallbackErrorPolicy",
    "TrainingEvent",
    "TrainingEventCallback",
    "TrainingEventDispatcher",
    "TrainingEventName",
]
