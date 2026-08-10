"""Capability-gated derived deployment exports.

The native ``.psann`` bundle remains the source of truth. Derived exports contain
tensor execution only and are accompanied by a JSON preprocessing/postprocessing
contract that describes how to reproduce the fitted raw-input boundary.
"""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
import os
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import torch

from .inference import _artifact_fields, _core_estimator, _output_names, _task_kind

DerivedExportFormat = Literal["torch_export", "onnx"]


class DerivedExportError(RuntimeError):
    """Base error for derived-export evaluation and writing."""


class ExportNotSupportedError(DerivedExportError):
    """Raised when a derived format has not passed capability evaluation."""


@dataclass(frozen=True)
class ExportCapability:
    """One format's parity and dynamic-batch evaluation."""

    format: str
    supported: bool
    parity: bool
    dynamic_batch: bool
    max_abs_error: float | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "supported": self.supported,
            "parity": self.parity,
            "dynamic_batch": self.dynamic_batch,
            "max_abs_error": self.max_abs_error,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ExportReport:
    """Capability report used to decide what may be advertised."""

    backbone: str
    task: str
    input_shape: tuple[int, ...]
    capabilities: tuple[ExportCapability, ...]

    @property
    def supported_formats(self) -> tuple[str, ...]:
        """Return only formats that passed both parity and dynamic-batch checks."""

        return tuple(item.format for item in self.capabilities if item.supported)

    @property
    def advertised_formats(self) -> tuple[str, ...]:
        """Return the mandatory native format plus certified derived formats."""

        return ("native",) + self.supported_formats

    def to_dict(self) -> dict[str, Any]:
        return {
            "backbone": self.backbone,
            "task": self.task,
            "input_shape": list(self.input_shape),
            "capabilities": [item.to_dict() for item in self.capabilities],
            "advertised_formats": list(self.advertised_formats),
        }


@dataclass(frozen=True)
class DerivedExport:
    """Paths and certification evidence for one written derived export."""

    path: Path
    format: DerivedExportFormat
    contract_path: Path
    capability: ExportCapability


def _backbone(model: Any) -> str:
    value = getattr(model, "_platform_model_spec_dict_", None)
    if value is None:
        value = getattr(_core_estimator(model), "_platform_model_spec_dict_", None)
    if isinstance(value, Mapping):
        return str(value.get("backbone", "unknown"))
    return str(getattr(model, "backbone_id_", "unknown"))


def _prepared_tensors(
    model: Any,
    sample_inputs: Any,
    context: Any | None,
) -> tuple[torch.nn.Module, tuple[torch.Tensor, ...]]:
    core = _core_estimator(model)
    prepare = getattr(core, "_prepare_inference_inputs", None)
    if not callable(prepare):
        raise DerivedExportError(
            "Derived export requires the registered PSANN raw-input preparation contract."
        )
    prepared, _, prepared_context = prepare(sample_inputs, context)
    if int(prepared.shape[0]) < 2:
        raise DerivedExportError(
            "Derived export evaluation requires at least two example samples so the "
            "batch dimension can be proven dynamic."
        )
    module = getattr(core, "model_", None)
    if not isinstance(module, torch.nn.Module):
        raise DerivedExportError("Derived export requires a fitted Torch model.")
    export_module = copy.deepcopy(module).to("cpu").eval()
    if hasattr(export_module, "set_state_updates"):
        export_module.set_state_updates(False)
    arguments: tuple[torch.Tensor, ...] = (
        torch.from_numpy(np.asarray(prepared, dtype=np.float32)).cpu(),
    )
    if prepared_context is not None:
        arguments += (torch.from_numpy(np.asarray(prepared_context, dtype=np.float32)).cpu(),)
    return export_module, arguments


def _dynamic_shapes(arguments: tuple[torch.Tensor, ...]) -> tuple[dict[int, Any], ...]:
    # A practical upper bound avoids SGR's sentinel guard against ``int64.max``
    # while remaining far above any realistic online/batch serving request.
    batch = torch.export.Dim("batch", min=1, max=1_000_000)
    return tuple({0: batch} for _ in arguments)


def _alternate_arguments(arguments: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    count = int(arguments[0].shape[0])
    if count > 2:
        size = count - 1
        return tuple(value[:size].contiguous() for value in arguments)
    return tuple(torch.cat((value, value[:1]), dim=0).contiguous() for value in arguments)


def _tensor_output(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)) and len(value) == 1 and isinstance(value[0], torch.Tensor):
        return value[0]
    raise DerivedExportError(
        f"Exported model returned unsupported output type {type(value).__name__}."
    )


def _error_text(exc: BaseException) -> str:
    message = " ".join(str(exc).split())
    return f"{type(exc).__name__}: {message}"[:500]


def _parity(
    expected: torch.Tensor,
    observed: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> tuple[bool, float]:
    expected_cpu = expected.detach().cpu().to(torch.float32)
    observed_cpu = observed.detach().cpu().to(torch.float32)
    if expected_cpu.shape != observed_cpu.shape:
        return False, float("inf")
    error = float((expected_cpu - observed_cpu).abs().max().item()) if expected_cpu.numel() else 0.0
    return bool(torch.allclose(expected_cpu, observed_cpu, atol=atol, rtol=rtol)), error


def _evaluate_torch_export(
    module: torch.nn.Module,
    arguments: tuple[torch.Tensor, ...],
    *,
    atol: float,
    rtol: float,
) -> ExportCapability:
    try:
        program = torch.export.export(
            module,
            arguments,
            dynamic_shapes=_dynamic_shapes(arguments),
            strict=False,
        )
        alternate = _alternate_arguments(arguments)
        with torch.inference_mode():
            expected = _tensor_output(module(*alternate))
            observed = _tensor_output(program.module()(*alternate))
        parity, error = _parity(expected, observed, atol=atol, rtol=rtol)
        return ExportCapability(
            format="torch_export",
            supported=parity,
            parity=parity,
            dynamic_batch=True,
            max_abs_error=error,
            reason=None if parity else "Numerical parity exceeded the configured tolerance.",
        )
    except Exception as exc:
        return ExportCapability(
            format="torch_export",
            supported=False,
            parity=False,
            dynamic_batch=False,
            reason=_error_text(exc),
        )


def _onnx_dependencies_available() -> bool:
    return all(
        importlib.util.find_spec(name) is not None for name in ("onnx", "onnxruntime", "onnxscript")
    )


def _write_onnx(
    module: torch.nn.Module,
    arguments: tuple[torch.Tensor, ...],
    path: Path,
) -> None:
    # The exporter emits Unicode status glyphs. Redirecting its progress stream
    # avoids Windows legacy-console encoding failures in otherwise valid exports.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        result = torch.onnx.export(
            module,
            arguments,
            str(path),
            input_names=[f"input_{index}" for index in range(len(arguments))],
            output_names=["output"],
            dynamo=True,
            dynamic_shapes=_dynamic_shapes(arguments),
            external_data=False,
        )
    if not path.is_file() and hasattr(result, "save"):
        result.save(str(path))


def _evaluate_onnx(
    module: torch.nn.Module,
    arguments: tuple[torch.Tensor, ...],
    *,
    atol: float,
    rtol: float,
) -> ExportCapability:
    if not _onnx_dependencies_available():
        return ExportCapability(
            format="onnx",
            supported=False,
            parity=False,
            dynamic_batch=False,
            reason=(
                "ONNX evaluation requires the optional 'onnx', 'onnxruntime', and "
                "'onnxscript' packages."
            ),
        )
    try:
        import onnxruntime

        with tempfile.TemporaryDirectory(prefix="psann-onnx-evaluation-") as directory:
            path = Path(directory) / "model.onnx"
            _write_onnx(module, arguments, path)
            session = onnxruntime.InferenceSession(
                str(path),
                providers=["CPUExecutionProvider"],
            )
            alternate = _alternate_arguments(arguments)
            feed = {
                descriptor.name: value.detach().cpu().numpy()
                for descriptor, value in zip(session.get_inputs(), alternate)
            }
            observed_np = session.run(None, feed)[0]
            with torch.inference_mode():
                expected = _tensor_output(module(*alternate))
            observed = torch.from_numpy(np.asarray(observed_np))
            parity, error = _parity(expected, observed, atol=atol, rtol=rtol)
        return ExportCapability(
            format="onnx",
            supported=parity,
            parity=parity,
            dynamic_batch=True,
            max_abs_error=error,
            reason=None if parity else "Numerical parity exceeded the configured tolerance.",
        )
    except Exception as exc:
        return ExportCapability(
            format="onnx",
            supported=False,
            parity=False,
            dynamic_batch=False,
            reason=_error_text(exc),
        )


def evaluate_export_capabilities(
    model: Any,
    sample_inputs: Any,
    *,
    context: Any | None = None,
    formats: Sequence[DerivedExportFormat] = ("torch_export", "onnx"),
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> ExportReport:
    """Evaluate parity and dynamic batching before advertising derived formats."""

    if atol < 0 or rtol < 0:
        raise ValueError("atol and rtol must be non-negative.")
    requested = tuple(dict.fromkeys(str(item) for item in formats))
    invalid = [item for item in requested if item not in {"torch_export", "onnx"}]
    if invalid:
        raise ValueError(f"Unknown derived export formats: {invalid!r}.")
    module, arguments = _prepared_tensors(model, sample_inputs, context)
    capabilities: list[ExportCapability] = []
    for format_name in requested:
        if format_name == "torch_export":
            capability = _evaluate_torch_export(module, arguments, atol=atol, rtol=rtol)
        else:
            capability = _evaluate_onnx(module, arguments, atol=atol, rtol=rtol)
        capabilities.append(capability)
    core = _core_estimator(model)
    return ExportReport(
        backbone=_backbone(model),
        task=_task_kind(model),
        input_shape=tuple(int(item) for item in getattr(core, "input_shape_", ())),
        capabilities=tuple(capabilities),
    )


def preprocessing_contract(model: Any) -> Mapping[str, Any]:
    """Generate the raw-input contract required by a tensor-only export."""

    core = _core_estimator(model)
    schema = getattr(core, "_platform_data_schema_", None)
    schema_to_dict = getattr(schema, "to_dict", None)
    if callable(schema_to_dict):
        input_schema = schema_to_dict()
    elif isinstance(schema, Mapping):
        input_schema = dict(schema)
    else:
        input_schema = {
            "input_shape": list(getattr(core, "input_shape_", ())),
            "feature_names": list(getattr(core, "feature_names_in_", ())),
            "data_format": str(getattr(core, "data_format", "channels_first")),
            "dtype": "float32",
        }
    task = getattr(model, "task_spec_", None)
    if task is None:
        task = getattr(core, "_platform_task_spec_", None)
    task_to_dict = getattr(task, "to_dict", None)
    task_contract = task_to_dict() if callable(task_to_dict) else {"kind": _task_kind(model)}
    artifact_version, model_id, run_id = _artifact_fields(model)
    target_scaler = dict(getattr(core, "preprocessing_contract_", {})).get(
        "target_scaler",
        {},
    )
    input_shape = tuple(int(dimension) for dimension in core.input_shape_)
    if core.preserve_shape and (
        core.per_element or getattr(core, "_use_channel_first_train_inputs_", False)
    ):
        if core.data_format == "channels_last":
            prepared_shape = (input_shape[-1], *input_shape[:-1])
        else:
            prepared_shape = input_shape
    else:
        prepared_shape = (int(np.prod(input_shape)),)
    return {
        "contract_format": "psann.preprocessing",
        "contract_version": "1.0",
        "source_of_truth": "native_psann_artifact",
        "raw_input": input_schema,
        "tensor_input": {
            "dtype": "float32",
            "shape": ["batch", *prepared_shape],
            "preprocessing": dict(getattr(core, "preprocessing_contract_", {})),
            "context_dim": getattr(core, "_context_dim_", None),
        },
        "tensor_output": {
            "kind": "raw_logits" if _task_kind(model) != "regression" else "scaled_prediction",
            "output_names": list(_output_names(model)),
            "target_inverse_transform": target_scaler,
            "task": task_contract,
        },
        "artifact": {
            "artifact_version": artifact_version,
            "model_id": model_id,
            "run_id": run_id,
        },
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def export_derived(
    model: Any,
    path: str | os.PathLike[str],
    *,
    format: DerivedExportFormat,
    sample_inputs: Any,
    context: Any | None = None,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> DerivedExport:
    """Write a derived export only after parity and dynamic-batch certification."""

    destination = Path(path).resolve()
    expected_suffix = ".pt2" if format == "torch_export" else ".onnx"
    if destination.suffix.lower() != expected_suffix:
        raise ValueError(f"{format} exports must use the {expected_suffix!r} extension.")
    report = evaluate_export_capabilities(
        model,
        sample_inputs,
        context=context,
        formats=(format,),
        atol=atol,
        rtol=rtol,
    )
    capability = report.capabilities[0]
    if not capability.supported:
        raise ExportNotSupportedError(
            f"{format} is not certified for {report.backbone}/{report.task}: {capability.reason}"
        )

    module, arguments = _prepared_tensors(model, sample_inputs, context)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        if format == "torch_export":
            program = torch.export.export(
                module,
                arguments,
                dynamic_shapes=_dynamic_shapes(arguments),
                strict=False,
            )
            torch.export.save(program, temporary)
        else:
            _write_onnx(module, arguments, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    contract_path = destination.with_suffix(destination.suffix + ".preprocessing.json")
    contract = dict(preprocessing_contract(model))
    contract["derived_export"] = {
        "format": format,
        "path": destination.name,
        "certification": capability.to_dict(),
    }
    _atomic_json(contract_path, contract)
    return DerivedExport(
        path=destination,
        format=format,
        contract_path=contract_path,
        capability=capability,
    )


__all__ = [
    "DerivedExport",
    "DerivedExportError",
    "DerivedExportFormat",
    "ExportCapability",
    "ExportNotSupportedError",
    "ExportReport",
    "evaluate_export_capabilities",
    "export_derived",
    "preprocessing_contract",
]
