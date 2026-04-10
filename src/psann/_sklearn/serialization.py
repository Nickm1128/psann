from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import torch

from .shared import (
    _deserialize_hisso_cfg,
    _deserialize_hisso_options,
    _serialize_hisso_cfg,
    _serialize_hisso_options,
)

if TYPE_CHECKING:
    from .base import PSANNRegressor


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
        params = payload.get("params", {})
        estimator = cls(**params)
        if "model" not in payload:
            raise RuntimeError("Checkpoint is missing model weights.")
        estimator.model_ = payload["model"]
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
