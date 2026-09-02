from __future__ import annotations

import copy
import inspect
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Union

import torch

from ..attention import AttentionConfig, ensure_attention_config
from .shared import (
    _deserialize_hisso_cfg,
    _deserialize_hisso_options,
    _serialize_hisso_cfg,
    _serialize_hisso_options,
)

if TYPE_CHECKING:
    from .base import PSANNRegressor


_LEGACY_GEOSPARSE_KEYS = {
    "geo_shape": "shape",
    "geo_k": "k",
    "geo_pattern": "pattern",
    "geo_radius": "radius",
    "geo_offsets": "offsets",
    "geo_wrap_mode": "wrap_mode",
    "geo_norm": "norm",
    "geo_drop_path_max": "drop_path_max",
    "geo_residual_alpha_init": "residual_alpha_init",
    "geo_bias": "bias",
    "geo_compute_mode": "compute_mode",
}
_EXECUTION_DEFAULTS = {
    "amp": False,
    "amp_dtype": "bfloat16",
    "compile": False,
    "compile_backend": "inductor",
    "compile_mode": "default",
    "compile_fullgraph": False,
    "compile_dynamic": False,
}
_DISCARDABLE_LEGACY_DRIFT: Dict[str, Dict[str, Any]] = {
    "ResPSANNRegressor": {
        **_EXECUTION_DEFAULTS,
        "context_builder": None,
        "context_builder_params": {},
    },
    "ResConvPSANNRegressor": {
        **_EXECUTION_DEFAULTS,
        "context_builder": None,
        "context_builder_params": {},
    },
    "WaveResNetRegressor": _EXECUTION_DEFAULTS,
    "SGRPSANNRegressor": {
        **_EXECUTION_DEFAULTS,
        "context_builder": None,
        "context_builder_params": {},
    },
    "GeoSparseRegressor": {"context_builder": None, "context_builder_params": {}},
}


def _is_discardable_legacy_value(class_name: str, key: str, value: Any) -> bool:
    """Return whether a documented old fallback-only value is semantically inert."""
    if class_name == "ResConvPSANNRegressor" and key == "attention":
        # The Phase 1 no-sklearn fallback serialized the inherited default as an
        # AttentionConfig(kind="none", ...), rather than as None.  Its other fields
        # cannot affect a disabled attention module, so normalize mappings through
        # the public parser and retain the strict rejection of enabled values.
        if value is None:
            return True
        if isinstance(value, (AttentionConfig, Mapping)):
            return not ensure_attention_config(value).is_enabled()
        return False

    expected = _DISCARDABLE_LEGACY_DRIFT.get(class_name, {})
    return key in expected and value == expected[key]


def _constructor_parameter_names(cls: type) -> set[str]:
    signature = inspect.signature(cls.__init__)  # type: ignore[misc]
    return {
        name
        for name, parameter in signature.parameters.items()
        if name != "self"
        and parameter.kind not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
    }


def _normalise_legacy_params(cls: type, raw_params: Any) -> Dict[str, Any]:
    """Translate only documented legacy payload drift into constructor parameters."""
    if not isinstance(raw_params, Mapping):
        raise TypeError(f"{cls.__name__} checkpoint params must be a mapping.")
    params = dict(raw_params)
    if cls.__name__ == "GeoSparseRegressor":
        for old_key, new_key in _LEGACY_GEOSPARSE_KEYS.items():
            if old_key not in params:
                continue
            old_value = params.pop(old_key)
            if new_key in params and params[new_key] != old_value:
                raise ValueError(
                    f"GeoSparseRegressor checkpoint has conflicting {old_key!r} and {new_key!r}."
                )
            params.setdefault(new_key, old_value)

    accepted = _constructor_parameter_names(cls)
    for key in list(params):
        if key in accepted:
            continue
        if _is_discardable_legacy_value(cls.__name__, key, params[key]):
            del params[key]
            continue
        raise ValueError(f"{cls.__name__} checkpoint contains unsupported parameter {key!r}.")
    return params


class _PSANNRegressorSerializationMixin:
    def _build_serialized_payload(self, model_cpu: torch.nn.Module) -> Dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "params": self.get_params(deep=True),
            "model": model_cpu,
            "scaler_kind": getattr(self, "_scaler_kind_", None),
            "scaler_state": getattr(self, "_scaler_state_", None),
            "scaler_spec": getattr(self, "_scaler_spec_", None),
            "scaler_obj": self.scaler if getattr(self, "_scaler_kind_", None) == "custom" else None,
            "target_scaler_kind": getattr(self, "_target_scaler_kind_", None),
            "target_scaler_state": getattr(self, "_target_scaler_state_", None),
            "target_scaler_spec": getattr(self, "_target_scaler_spec_", None),
            "target_scaler_obj": (
                self.target_scaler
                if getattr(self, "_target_scaler_kind_", None) == "custom"
                else None
            ),
            "input_shape": (
                tuple(self.input_shape_)
                if getattr(self, "input_shape_", None) is not None
                else None
            ),
            "internal_shape_cf": (
                tuple(self._internal_input_shape_cf_)
                if getattr(self, "_internal_input_shape_cf_", None) is not None
                else None
            ),
            "primary_dim": self._primary_dim_,
            "output_dim": self._output_dim_,
            "keep_column_output": bool(getattr(self, "_keep_column_output_", False)),
            "train_layout": self._train_inputs_layout_,
            "target_cf_shape": self._target_cf_shape_,
            "target_vector_dim": self._target_vector_dim_,
            "output_shape_tuple": self._output_shape_tuple_,
            "context_dim": self._context_dim_,
            "hisso_cfg": _serialize_hisso_cfg(getattr(self, "_hisso_cfg_", None)),
            "hisso_options": _serialize_hisso_options(getattr(self, "_hisso_options_", None)),
            "hisso_reward_fn": getattr(self, "_hisso_reward_fn_", None),
            "hisso_context_extractor": getattr(self, "_hisso_context_extractor_", None),
            "hisso_trained": bool(getattr(self, "_hisso_trained_", False)),
        }

    def save(self, path: str) -> None:
        self._ensure_fitted()
        model = self.model_
        orig_device = torch.device("cpu")
        for param in model.parameters():
            orig_device = param.device
            break
        model_cpu = copy.deepcopy(model).cpu()
        payload = self._build_serialized_payload(model_cpu)
        torch.save(payload, path)
        model.to(orig_device)

    @classmethod
    def load(
        cls,
        path: str,
        *,
        map_location: Optional[Union[str, torch.device]] = "cpu",
    ) -> "PSANNRegressor":
        try:
            payload = torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=map_location)
        class_name = payload.get("class")
        if class_name is not None and class_name != cls.__name__:
            raise ValueError(
                f"Checkpoint was created for '{class_name}', cannot load into '{cls.__name__}'."
            )
        params = _normalise_legacy_params(cls, payload.get("params", {}))
        estimator = cls(**params)
        if "model" not in payload:
            raise RuntimeError("Checkpoint is missing model weights.")
        estimator.model_ = payload["model"]
        if map_location is not None:
            # A caller choosing a map location expects inference to remain there,
            # including when the serialized estimator used device="auto".
            setattr(estimator, "device", torch.device(map_location))
        estimator.model_.to(estimator._device())
        estimator.model_.eval()

        estimator._scaler_kind_ = payload.get("scaler_kind")
        estimator._scaler_state_ = payload.get("scaler_state")
        estimator._scaler_spec_ = payload.get("scaler_spec")
        scaler_obj = payload.get("scaler_obj")
        if scaler_obj is not None:
            estimator.scaler = scaler_obj
            estimator._scaler_fitted_ = True

        estimator._target_scaler_kind_ = payload.get("target_scaler_kind")
        estimator._target_scaler_state_ = payload.get("target_scaler_state")
        estimator._target_scaler_spec_ = payload.get("target_scaler_spec")
        target_scaler_obj = payload.get("target_scaler_obj")
        if target_scaler_obj is not None:
            estimator.target_scaler = target_scaler_obj
            estimator._target_scaler_fitted_ = True

        input_shape = payload.get("input_shape")
        estimator.input_shape_ = tuple(input_shape) if input_shape is not None else None
        internal_cf = payload.get("internal_shape_cf")
        estimator._internal_input_shape_cf_ = (
            tuple(internal_cf) if internal_cf is not None else None
        )
        estimator._primary_dim_ = payload.get("primary_dim")
        estimator._output_dim_ = payload.get("output_dim")
        estimator._keep_column_output_ = bool(payload.get("keep_column_output", False))
        estimator._train_inputs_layout_ = payload.get("train_layout", "flat")
        target_cf = payload.get("target_cf_shape")
        estimator._target_cf_shape_ = tuple(target_cf) if target_cf is not None else None
        estimator._target_vector_dim_ = payload.get("target_vector_dim")
        output_shape_tuple = payload.get("output_shape_tuple")
        estimator._output_shape_tuple_ = (
            tuple(output_shape_tuple) if output_shape_tuple is not None else None
        )
        estimator._context_dim_ = payload.get("context_dim")

        estimator._hisso_cfg_ = _deserialize_hisso_cfg(payload.get("hisso_cfg"))
        estimator._hisso_options_ = _deserialize_hisso_options(payload.get("hisso_options"))
        estimator._hisso_reward_fn_ = payload.get("hisso_reward_fn")
        estimator._hisso_context_extractor_ = payload.get("hisso_context_extractor")
        estimator._hisso_trained_ = bool(payload.get("hisso_trained", False))
        estimator._hisso_trainer_ = None
        estimator._hisso_cache_ = None
        return estimator


__all__ = ["_PSANNRegressorSerializationMixin"]
