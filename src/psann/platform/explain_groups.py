"""Feature naming, grouping, and partition linkages for SHAP games."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .explain_contracts import FeatureGroup, GroupStrategy
from .inference import InferenceRuntime, _core_estimator


def feature_names(runtime: InferenceRuntime, input_shape: tuple[int, ...]) -> tuple[str, ...]:
    core = _core_estimator(runtime.model)
    configured = tuple(str(item) for item in getattr(core, "feature_names_in_", ()))
    width = math.prod(input_shape)
    if len(input_shape) == 1 and len(configured) == width:
        return configured
    return tuple(
        "input[" + ",".join(str(index) for index in coordinate) + "]"
        for coordinate in np.ndindex(input_shape)
    )


def _model_spec(runtime: InferenceRuntime) -> Mapping[str, Any]:
    value = getattr(runtime.model, "_platform_model_spec_dict_", None)
    if value is None:
        value = getattr(_core_estimator(runtime.model), "_platform_model_spec_dict_", None)
    return value if isinstance(value, Mapping) else {}


def resolved_group_strategy(
    runtime: InferenceRuntime,
    input_shape: tuple[int, ...],
    requested: GroupStrategy,
) -> GroupStrategy:
    if requested != "auto":
        return requested
    if len(input_shape) == 1:
        return "feature"
    backbone = str(_model_spec(runtime).get("backbone", ""))
    core = _core_estimator(runtime.model)
    if backbone in {"sgr_psann", "wave_resnet"} and not bool(core.preserve_shape):
        return "time_step"
    if bool(core.preserve_shape) or "conv" in backbone:
        return "spatial_region"
    return "time_step"


def feature_groups(
    runtime: InferenceRuntime,
    input_shape: tuple[int, ...],
    strategy: GroupStrategy,
    names: tuple[str, ...],
) -> tuple[FeatureGroup, ...]:
    indices: np.ndarray = np.arange(math.prod(input_shape), dtype=np.int64).reshape(input_shape)
    if strategy == "feature":
        return tuple(
            FeatureGroup(name=names[index], indices=(index,), strategy=strategy)
            for index in range(len(names))
        )
    if strategy == "time_step":
        if len(input_shape) < 2:
            raise ValueError("time_step grouping requires input rank >= 2.")
        return tuple(
            FeatureGroup(
                name=f"time_step[{step}]",
                indices=tuple(int(item) for item in indices[step].reshape(-1)),
                strategy=strategy,
            )
            for step in range(input_shape[0])
        )
    if strategy == "channel":
        core = _core_estimator(runtime.model)
        axis = len(input_shape) - 1 if str(core.data_format) == "channels_last" else 0
        return tuple(
            FeatureGroup(
                name=f"channel[{channel}]",
                indices=tuple(
                    int(item) for item in np.take(indices, channel, axis=axis).reshape(-1)
                ),
                strategy=strategy,
            )
            for channel in range(input_shape[axis])
        )
    if strategy == "spatial_region":
        if len(input_shape) < 2:
            raise ValueError("spatial_region grouping requires input rank >= 2.")
        core = _core_estimator(runtime.model)
        channel_axis = len(input_shape) - 1 if str(core.data_format) == "channels_last" else 0
        spatial_shape = tuple(
            dimension for axis, dimension in enumerate(input_shape) if axis != channel_axis
        )
        groups: list[FeatureGroup] = []
        for coordinate in np.ndindex(spatial_shape):
            selector: list[Any] = []
            spatial_index = 0
            for axis in range(len(input_shape)):
                if axis == channel_axis:
                    selector.append(slice(None))
                else:
                    selector.append(coordinate[spatial_index])
                    spatial_index += 1
            members = tuple(int(item) for item in indices[tuple(selector)].reshape(-1))
            label = ",".join(str(item) for item in coordinate)
            groups.append(
                FeatureGroup(
                    name=f"spatial[{label}]",
                    indices=members,
                    strategy=strategy,
                )
            )
        return tuple(groups)
    raise ValueError(f"Unsupported group strategy {strategy!r}.")


def domain_linkage(groups: Sequence[FeatureGroup], feature_count: int) -> np.ndarray:
    if feature_count < 2:
        raise ValueError("Partition maskers require at least two features.")
    rows: list[list[float]] = []
    next_node = feature_count
    roots: list[tuple[int, int]] = []
    for group in groups:
        members = list(group.indices)
        if not members:
            continue
        root = members[0]
        count = 1
        for member in members[1:]:
            count += 1
            rows.append([float(root), float(member), 0.0, float(count)])
            root = next_node
            next_node += 1
        roots.append((root, count))
    if not roots:
        raise ValueError("Domain grouping produced no feature groups.")
    root, count = roots[0]
    for other, other_count in roots[1:]:
        count += other_count
        rows.append([float(root), float(other), 1.0, float(count)])
        root = next_node
        next_node += 1
    linkage = np.asarray(rows, dtype=np.float64)
    if linkage.shape != (feature_count - 1, 4):
        raise RuntimeError("Domain feature groups must partition every raw-input feature once.")
    return linkage


__all__ = [
    "domain_linkage",
    "feature_groups",
    "feature_names",
    "resolved_group_strategy",
]
