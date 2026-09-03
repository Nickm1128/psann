"""Canonical estimator surface without eager fit-plumbing import cycles."""

from __future__ import annotations

__all__ = ["PSANNRegressor"]


def __getattr__(name: str):
    if name == "PSANNRegressor":
        from .regressor import PSANNRegressor

        globals()[name] = PSANNRegressor
        return PSANNRegressor
    raise AttributeError(name)
