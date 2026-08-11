from __future__ import annotations

import inspect
from typing import Any, Dict

import numpy as np


class BaseEstimator:
    """Small sklearn-compatible estimator base used when sklearn is unavailable."""

    @classmethod
    def _get_param_names(cls) -> list[str]:
        init = cls.__init__
        if init is object.__init__:
            return []
        parameters = inspect.signature(init).parameters.values()
        if any(parameter.kind == parameter.VAR_POSITIONAL for parameter in parameters):
            raise RuntimeError(
                f"{cls.__name__} estimators must declare constructor parameters explicitly."
            )
        return sorted(
            parameter.name
            for parameter in parameters
            if parameter.name != "self" and parameter.kind != parameter.VAR_KEYWORD
        )

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for name in self._get_param_names():
            value = getattr(self, name)
            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                for nested_name, nested_value in value.get_params().items():
                    params[f"{name}__{nested_name}"] = nested_value
            params[name] = value
        return params

    def set_params(self, **params: Any) -> BaseEstimator:
        if not params:
            return self
        valid_params = self.get_params(deep=True)
        nested_params: Dict[str, Dict[str, Any]] = {}
        for raw_name, value in params.items():
            name, delimiter, nested_name = raw_name.partition("__")
            if name not in valid_params:
                valid = ", ".join(sorted(self._get_param_names()))
                raise ValueError(
                    f"Invalid parameter {name!r} for estimator {self.__class__.__name__}. "
                    f"Valid parameters are: {valid}."
                )
            if delimiter:
                nested_params.setdefault(name, {})[nested_name] = value
            else:
                setattr(self, name, value)
                valid_params[name] = value
        for name, nested in nested_params.items():
            valid_params[name].set_params(**nested)
        return self


class RegressorMixin:
    pass


class ClassifierMixin:
    _estimator_type = "classifier"


def r2_score(y_true: Any, y_pred: Any) -> float:
    true_values = np.asarray(y_true)
    predicted_values = np.asarray(y_pred)
    residual = ((true_values - predicted_values) ** 2).sum()
    variance = ((true_values - true_values.mean()) ** 2).sum()
    return float(1.0 - (residual / variance if variance != 0 else np.nan))
