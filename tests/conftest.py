"""Pytest configuration for PSANN tests.

This module registers targeted warning filters so that the default ``pytest``
run stays signal-rich while still exercising deprecated pathways that we
intentionally cover for backwards compatibility. Each filter matches a known
warning emitted by those regression tests; unexpected warnings continue to
surface normally.
"""

from __future__ import annotations

_WARNING_FILTERS = (
    (
        r"`conv_channels` has no effect when preserve_shape=False; ignoring value\.",
        "UserWarning",
    ),
    (
        r"`conv_channels` differs from `hidden_units`; using `conv_channels` for convolutional paths\.",
        "UserWarning",
    ),
    (
        r"conv_channels has no effect for WaveResNetRegressor; ignoring value\.",
        "RuntimeWarning",
    ),
    (
        r"conv_kernel_size has no effect for WaveResNetRegressor; ignoring value\.",
        "RuntimeWarning",
    ),
    (
        r".*`hidden_width` is deprecated; use `hidden_units` instead\.",
        "DeprecationWarning",
    ),
    (
        r".*`hidden_channels` is deprecated; use `conv_channels` instead\.",
        "DeprecationWarning",
    ),
    (
        r"`torch\.nn\.utils\.weight_norm` is deprecated in favor of `torch\.nn\.utils\.parametrizations\.weight_norm`\.",
        "FutureWarning",
    ),
)


def pytest_configure(config) -> None:
    for message, category_name in _WARNING_FILTERS:
        config.addinivalue_line(
            "filterwarnings",
            f"ignore:{message}:{category_name}",
        )
