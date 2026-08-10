"""Shared warning types for the deprecated whole-object checkpoint boundary."""

from __future__ import annotations


class LegacyCheckpointWarning(UserWarning):
    """Warn that legacy persistence is deprecated and may execute Python code."""


LEGACY_CHECKPOINT_MESSAGE = (
    "Legacy estimator save/load uses unrestricted Python pickle and may execute code "
    "while loading. Prefer TrainingRun.export(...'.psann') and psann.load_model(). "
    "Only load legacy files whose source and contents you trust."
)


__all__ = ["LEGACY_CHECKPOINT_MESSAGE", "LegacyCheckpointWarning"]
