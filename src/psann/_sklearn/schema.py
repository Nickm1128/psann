"""Dependency-light feature and output schema handling for estimators."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def _schema_dict(estimator: Any) -> dict[str, Any]:
    schema = getattr(estimator, "_platform_data_schema_", None)
    if schema is None:
        return {}
    if hasattr(schema, "to_dict"):
        value = schema.to_dict()
    elif isinstance(schema, Mapping):
        value = dict(schema)
    else:
        raise TypeError(
            "_platform_data_schema_ must be a DataSchema or mapping; "
            f"received {type(schema).__name__}."
        )
    return dict(value)


def _column_names(value: Any) -> tuple[str, ...]:
    columns = getattr(value, "columns", None)
    if columns is None:
        return ()
    names = tuple(str(item) for item in list(columns))
    if len(set(names)) != len(names):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(f"Named features contain duplicate columns: {duplicates!r}.")
    return names


def _reorder_named_input(value: Any, expected: tuple[str, ...]) -> Any:
    locator = getattr(value, "loc", None)
    if locator is not None:
        return locator[:, list(expected)]
    try:
        return value[list(expected)]
    except Exception as exc:
        raise TypeError(
            "feature_policy='reorder' requires a dataframe-like input supporting "
            "column selection."
        ) from exc


def _validate_named_features(
    value: Any,
    *,
    expected: tuple[str, ...],
    policy: str,
    stage: str,
) -> Any:
    observed = _column_names(value)
    if not expected or policy == "positional":
        return value
    if not observed:
        shape = getattr(value, "shape", ())
        if len(shape) < 2 or int(shape[1]) != len(expected):
            received = int(shape[1]) if len(shape) >= 2 else None
            raise ValueError(f"{stage} expected {len(expected)} features but received {received}.")
        return value
    missing = [name for name in expected if name not in observed]
    unexpected = [name for name in observed if name not in expected]
    if missing or unexpected:
        raise ValueError(
            f"{stage} feature schema mismatch: missing={missing!r}, " f"unexpected={unexpected!r}."
        )
    if observed == expected:
        return value
    if policy == "reorder":
        return _reorder_named_input(value, expected)
    raise ValueError(
        f"{stage} feature order {observed!r} does not match fitted order {expected!r}. "
        "Use feature_policy='reorder' to opt into safe name-based reordering."
    )


def _input_shape(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        shape = np.asarray(value).shape
    if len(shape) < 2:
        raise ValueError(f"X must be at least 2D; received shape {tuple(shape)!r}.")
    return tuple(int(dim) for dim in shape[1:])


def capture_fit_schema(estimator: Any, X: Any, y: Any) -> Any:
    """Validate configured schema and capture sklearn-compatible fitted metadata."""

    schema = _schema_dict(estimator)
    configured_names = tuple(str(item) for item in schema.get("feature_names", ()))
    policy = str(
        schema.get(
            "feature_policy",
            getattr(estimator, "_feature_schema_policy_", "strict"),
        )
    )
    if policy not in {"strict", "reorder", "positional"}:
        raise ValueError("feature_policy must be strict, reorder, or positional.")
    X_value = _validate_named_features(
        X,
        expected=configured_names,
        policy=policy,
        stage="fit",
    )
    shape = _input_shape(X_value)
    configured_shape = tuple(int(item) for item in schema.get("input_shape", ()))
    if configured_shape and shape != configured_shape:
        raise ValueError(
            f"fit input shape {shape!r} does not match configured input_shape "
            f"{configured_shape!r}."
        )
    observed_names = _column_names(X_value)
    names = configured_names or observed_names
    if names:
        if len(shape) != 1:
            raise ValueError("Named feature schemas currently require 2D tabular inputs.")
        if len(names) != shape[0]:
            raise ValueError(
                f"Feature schema defines {len(names)} names for input shape {shape!r}."
            )
        estimator.feature_names_in_ = np.asarray(names, dtype=object)
    elif hasattr(estimator, "feature_names_in_"):
        delattr(estimator, "feature_names_in_")
    estimator.n_features_in_ = int(shape[0])
    estimator.input_shape_contract_ = shape
    estimator.input_dtype_ = str(
        schema.get("dtype") or getattr(getattr(X_value, "dtypes", None), "dtype", "")
    )
    estimator.feature_schema_policy_ = policy

    configured_outputs = tuple(str(item) for item in schema.get("output_names", ()))
    observed_outputs = _column_names(y) if y is not None else ()
    output_names = configured_outputs or observed_outputs
    if output_names:
        target = np.asarray(y)
        width = 1 if target.ndim == 1 else int(target.shape[1])
        if len(output_names) != width:
            raise ValueError(
                f"Output schema defines {len(output_names)} names for target width {width}."
            )
        estimator.output_names_ = np.asarray(output_names, dtype=object)
    estimator.data_format_ = str(schema.get("data_format", estimator.data_format))
    task_spec = getattr(estimator, "_platform_task_spec_", None)
    estimator.task_metadata_ = (
        task_spec.to_dict()
        if hasattr(task_spec, "to_dict")
        else dict(task_spec) if isinstance(task_spec, Mapping) else {"kind": "regression"}
    )
    return X_value


def validate_inference_schema(estimator: Any, X: Any) -> Any:
    """Validate or safely reorder named features against fitted metadata."""

    expected = tuple(str(item) for item in getattr(estimator, "feature_names_in_", ()))
    fitted_shape = tuple(
        int(item)
        for item in getattr(
            estimator,
            "input_shape_contract_",
            getattr(estimator, "input_shape_", ()),
        )
    )
    raw_shape = tuple(int(item) for item in getattr(X, "shape", np.asarray(X).shape))
    if not expected and fitted_shape and raw_shape == fitted_shape:
        return X
    policy = str(
        getattr(
            estimator,
            "feature_schema_policy_",
            getattr(estimator, "_feature_schema_policy_", "strict"),
        )
    )
    X_value = _validate_named_features(
        X,
        expected=expected,
        policy=policy,
        stage="predict",
    )
    shape = _input_shape(X_value)
    if fitted_shape and shape != fitted_shape:
        raise ValueError(
            f"predict input shape {shape!r} does not match fitted shape {fitted_shape!r}."
        )
    return X_value


def _structured_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _structured_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_structured_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"runtime_type": f"{type(value).__module__}.{type(value).__qualname__}"}


def capture_preprocessing_contract(estimator: Any) -> None:
    """Record fitted preprocessing and target-scaling state as structured data."""

    schema = _schema_dict(estimator)
    estimator.preprocessing_contract_ = {
        "input_scaler": {
            "kind": getattr(estimator, "_scaler_kind_", None),
            "parameters": _structured_value(getattr(estimator, "scaler_params", None)),
            "state": _structured_value(getattr(estimator, "_scaler_state_", None)),
        },
        "target_scaler": {
            "kind": getattr(estimator, "_target_scaler_kind_", None),
            "parameters": _structured_value(getattr(estimator, "target_scaler_params", None)),
            "state": _structured_value(getattr(estimator, "_target_scaler_state_", None)),
        },
        "declared": _structured_value(schema.get("preprocessing", {})),
        "declared_target_scaling": _structured_value(schema.get("target_scaling", {})),
        "categorical_encoder": schema.get("categorical_encoder"),
        "missing_value_imputer": schema.get("missing_value_imputer"),
    }


SCHEMA_STATE_FIELDS = (
    "n_features_in_",
    "feature_names_in_",
    "output_names_",
    "input_shape_contract_",
    "input_dtype_",
    "feature_schema_policy_",
    "data_format_",
    "task_metadata_",
    "preprocessing_contract_",
    "_platform_model_spec_dict_",
    "_platform_data_schema_",
    "_platform_task_spec_",
)


__all__ = [
    "SCHEMA_STATE_FIELDS",
    "capture_fit_schema",
    "capture_preprocessing_contract",
    "validate_inference_schema",
]
