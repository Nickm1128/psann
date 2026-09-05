"""Versioned model artifacts and explicit trainer-checkpoint reconstruction."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from .architectures import LMConfig, build_lm_model, normalize_lm_config, to_mapping
from .architectures.compat import legacy_lm_config
from .lm.data.tokenizer import Tokenizer

_PACKAGE_VERSION = "2.0.0"


def package_version() -> str:
    try:
        return version("psannlm")
    except PackageNotFoundError:
        return _PACKAGE_VERSION


def model_config(model: nn.Module) -> LMConfig:
    """Read canonical metadata, or translate a documented legacy model."""
    config = getattr(model, "lm_config", None)
    if config is not None:
        return normalize_lm_config(config, for_build=True)
    for name in ("module", "_orig_mod"):
        wrapped = getattr(model, name, None)
        if isinstance(wrapped, nn.Module):
            return model_config(wrapped)
    config = getattr(model, "cfg", None)
    if isinstance(config, LMConfig):
        return normalize_lm_config(config, for_build=True)
    if is_dataclass(config):
        bases = {
            "ResPSANNTransformerConfig": "respsann",
            "WaveResNetTransformerConfig": "waveresnet",
            "VanillaTransformerConfig": "transformer",
            "GeoSparseTransformerConfig": "geosparse",
        }
        base = bases.get(type(config).__name__)
        if base is not None:
            raw = {f.name: getattr(config, f.name) for f in fields(config)}
            # A constructed legacy config has already resolved the rope alias.
            raw.pop("rope", None)
            return normalize_lm_config(legacy_lm_config(base, raw, warn=False), for_build=True)
    raise ValueError("model.config is required for LM checkpoint reconstruction.")


def _exact_keys(raw: Mapping[str, Any], required: set[str], optional: set[str], path: str) -> None:
    unknown, missing = set(raw) - required - optional, required - set(raw)
    if unknown:
        raise ValueError(f"{path}.{sorted(unknown, key=str)[0]} is unknown.")
    if missing:
        raise ValueError(f"{path}.{sorted(missing)[0]} is required.")


def _state(value: object, path: str) -> Mapping[str, torch.Tensor]:
    if not isinstance(value, Mapping) or not value:
        raise TypeError(f"{path} must be a nonempty tensor mapping.")
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, torch.Tensor):
            raise TypeError(f"{path}.{key} must be a tensor.")
    return value


def load_model_state(
    model: nn.Module, state: Mapping[str, torch.Tensor], *, path: str = "checkpoint.state_dict"
) -> None:
    expected = model.state_dict()
    _exact_keys(state, set(expected), set(), path)
    for key, value in state.items():
        if value.shape != expected[key].shape:
            raise ValueError(f"{path}.{key}.shape is incompatible with config.")
        if value.is_floating_point() != expected[key].is_floating_point():
            raise TypeError(f"{path}.{key}.dtype is incompatible with config.")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(f"{path}: {exc}") from exc


def checkpoint_metadata(
    payload: object, *, legacy_config: LMConfig | None = None
) -> tuple[LMConfig, Mapping[str, torch.Tensor], str]:
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint must be a mapping.")
    schema = payload.get("schema")
    if "schema" in payload or "schema_version" in payload:
        if not isinstance(schema, str) or schema not in {"psannlm.model", "psannlm.trainer"}:
            raise ValueError("checkpoint.schema must be psannlm.model or psannlm.trainer.")
        v = payload.get("schema_version")
        if isinstance(v, bool) or not isinstance(v, int) or v != 1:
            raise ValueError("checkpoint.schema_version must be integer 1.")
        common = {"schema", "schema_version", "package_version", "config"}
        if schema == "psannlm.model":
            _exact_keys(
                payload, common | {"state_dict", "device", "tokenizer"}, set(), "checkpoint"
            )
            state = _state(payload["state_dict"], "checkpoint.state_dict")
            device = payload["device"]
            if not isinstance(device, str):
                raise TypeError("checkpoint.device must be a device string.")
            try:
                saved_device = torch.device(device)
            except (RuntimeError, ValueError) as exc:
                raise ValueError("checkpoint.device is invalid.") from exc
            if saved_device.type not in {"cpu", "cuda"}:
                raise ValueError("checkpoint.device must name CPU or CUDA.")
            if payload["tokenizer"] is not None:
                Tokenizer.from_state(payload["tokenizer"])
        else:
            _exact_keys(
                payload,
                common | {"model", "optim", "cfg", "state", "scaler", "scheduler", "rng"},
                {"data_state"},
                "checkpoint",
            )
            state = _state(payload["model"], "checkpoint.model")
            for key in ("optim", "cfg", "state", "scaler", "scheduler", "rng"):
                if not isinstance(payload[key], Mapping):
                    raise TypeError(f"checkpoint.{key} must be a mapping.")
            from .lm.config import normalize_train_config

            normalize_train_config(payload["cfg"])
            _exact_keys(payload["state"], {"step", "epoch"}, set(), "checkpoint.state")
            for key in ("step", "epoch"):
                value = payload["state"][key]
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"checkpoint.state.{key} must be a nonnegative integer.")
            _exact_keys(
                payload["rng"], {"torch", "cuda", "python", "numpy"}, set(), "checkpoint.rng"
            )
            from .lm.train.rng import validate_rng

            validate_rng(payload["rng"])
            _exact_keys(payload["optim"], {"state", "param_groups"}, set(), "checkpoint.optim")
            if not isinstance(payload["optim"]["state"], Mapping) or not isinstance(
                payload["optim"]["param_groups"], list
            ):
                raise TypeError("checkpoint.optim.state/param_groups have invalid types.")
        if not isinstance(payload["package_version"], str) or not payload["package_version"]:
            raise ValueError("checkpoint.package_version must be a nonempty string.")
        if not isinstance(payload["config"], Mapping):
            raise TypeError("checkpoint.config must be a tagged mapping.")
        if payload["config"].get("kind") != "lm":
            raise ValueError("checkpoint.config.kind must be 'lm'.")
        try:
            config = normalize_lm_config(payload["config"], for_build=True)
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"checkpoint.config: {exc}") from exc
        return config, state, schema
    if "state_dict" in payload:
        _exact_keys(payload, {"config", "state_dict"}, {"device"}, "checkpoint")
        raw = payload["config"]
        if not isinstance(raw, Mapping):
            raise TypeError("checkpoint.config must be a mapping.")
        allowed = {
            "base",
            "d_model",
            "n_layers",
            "n_heads",
            "d_mlp",
            "vocab_size",
            "rope",
            "positional_encoding",
            "sine",
            "overrides",
        }
        _exact_keys(raw, set(), allowed, "checkpoint.config")
        raw = dict(raw)
        base = raw.pop("base", "waveresnet")
        overrides = raw.pop("overrides", {})
        if not isinstance(overrides, Mapping):
            raise TypeError("checkpoint.config.overrides must be a mapping.")
        raw.update(overrides)
        # Historical high-level save could omit the resolved vocabulary.
        state = _state(payload["state_dict"], "checkpoint.state_dict")
        if raw.get("vocab_size") is None:
            if "embed.weight" not in state:
                raise ValueError("checkpoint.state_dict.embed.weight is required.")
            raw["vocab_size"] = state["embed.weight"].shape[0]
        config = normalize_lm_config(
            legacy_lm_config(base, raw, high_level=True, warn=False), for_build=True
        )
        return config, state, "psannlm.model"
    if "model" in payload:
        _exact_keys(
            payload, {"model", "optim", "state", "cfg"}, {"data_state", "config"}, "checkpoint"
        )
        if payload.get("config", legacy_config) is None:
            raise ValueError("checkpoint.config is required for legacy trainer reconstruction.")
        config = normalize_lm_config(payload.get("config", legacy_config), for_build=True)
        state = _state(payload["model"], "checkpoint.model")
        if state and all(key.startswith("module.") for key in state):
            state = {key.removeprefix("module."): item for key, item in state.items()}
        return config, state, "psannlm.trainer"
    if payload and all(
        isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in payload.items()
    ):
        if legacy_config is None:
            raise ValueError("checkpoint.config is required for legacy raw weights.")
        return (
            normalize_lm_config(legacy_config, for_build=True),
            _state(payload, "checkpoint.weights"),
            "psannlm.weights",
        )
    raise ValueError(
        "checkpoint.schema is absent and no supported legacy artifact keys were found."
    )


@dataclass(frozen=True)
class LoadedLMCheckpoint:
    model: nn.Module
    config: LMConfig
    tokenizer: Tokenizer | None
    artifact_kind: str
    payload: Mapping[str, Any]


def load_lm_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device | None = "cpu",
    legacy_config: LMConfig | None = None,
    require_model: bool = False,
) -> LoadedLMCheckpoint:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config, state, kind = checkpoint_metadata(payload, legacy_config=legacy_config)
    if require_model and kind != "psannlm.model":
        raise ValueError(f"checkpoint.schema is {kind}; use load_lm_checkpoint for this artifact.")
    model = build_lm_model(config).model
    load_model_state(
        model,
        state,
        path="checkpoint.state_dict" if kind == "psannlm.model" else "checkpoint.model",
    )
    tokenizer = (
        Tokenizer.from_state(payload["tokenizer"]) if payload.get("tokenizer") is not None else None
    )
    device = map_location
    if device is None:
        saved = payload.get("device", "cpu")
        device = saved if str(saved).startswith("cuda") and torch.cuda.is_available() else "cpu"
    model.to(torch.device(device))
    return LoadedLMCheckpoint(model, config, tokenizer, kind, payload)


def model_payload(model: nn.Module, tokenizer: Tokenizer | None) -> dict[str, Any]:
    config = model_config(model)
    return {
        "schema": "psannlm.model",
        "schema_version": 1,
        "package_version": package_version(),
        "config": to_mapping(config),
        "state_dict": model.state_dict(),
        "tokenizer": tokenizer.to_state() if tokenizer is not None else None,
        "device": str(next(model.parameters()).device),
    }
