"""Optional reference HTTP service for native PSANN artifacts.

Install ``psann[serve]`` and run:

```
python -m psann.serving --artifact /artifacts/model.psann
```
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .platform.inference import InferenceRuntime, load_runtime
from .platform.specs import InferenceConfig

LOGGER = logging.getLogger("psann.serving")


def _json_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


@dataclass
class ServiceMetrics:
    """Small in-process aggregate plus structured per-request logging."""

    requests: int = 0
    errors: int = 0
    samples: int = 0
    total_latency_ms: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, event: Mapping[str, Any]) -> None:
        with self._lock:
            self.requests += 1
            self.errors += int(event.get("status") == "error")
            self.samples += int(event.get("batch_size", 0))
            self.total_latency_ms += float(event.get("latency_ms", 0.0))
        LOGGER.info(json.dumps({"event": "psann.inference_request", **dict(event)}, sort_keys=True))

    def snapshot(self) -> Mapping[str, Any]:
        with self._lock:
            average = self.total_latency_ms / self.requests if self.requests else 0.0
            return {
                "requests": self.requests,
                "errors": self.errors,
                "samples": self.samples,
                "average_latency_ms": average,
            }


def _optional_service_imports() -> tuple[Any, Any, Any, Any]:
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel, ConfigDict
    except ImportError as exc:  # pragma: no cover - exercised in a dependency-light wheel
        raise ImportError(
            "The reference service requires optional dependencies. "
            "Install them with `pip install 'psann[serve]'`."
        ) from exc
    return FastAPI, HTTPException, BaseModel, ConfigDict


def create_app(
    *,
    artifact_path: str | os.PathLike[str] | None = None,
    runtime: InferenceRuntime | None = None,
    config: InferenceConfig | Mapping[str, Any] | None = None,
) -> Any:
    """Create the optional FastAPI reference application.

    Model-loading failures keep liveness healthy while readiness returns 503. This
    supports container orchestrators without logging or returning raw request inputs.
    """

    FastAPI, HTTPException, BaseModel, ConfigDict = _optional_service_imports()

    class PredictionRequest(BaseModel):
        model_config = ConfigDict(extra="forbid")

        inputs: list[Any]
        context: list[Any] | None = None
        batch_size: int | None = None
        return_logits: bool | None = None

    app = FastAPI(title="PSANN reference inference service", version="1")
    app.state.metrics = ServiceMetrics()
    app.state.load_error = None
    app.state.runtime = runtime

    configured_path = artifact_path or os.environ.get("PSANN_ARTIFACT_PATH")
    if app.state.runtime is None and configured_path:
        try:
            app.state.runtime = load_runtime(Path(configured_path), config=config)
        except Exception:
            app.state.load_error = "Artifact failed validation or loading."
            LOGGER.exception("PSANN artifact failed to load")

    @app.get("/health")
    def health() -> Mapping[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> Mapping[str, str]:
        if app.state.runtime is None:
            raise HTTPException(
                status_code=503,
                detail=app.state.load_error or "No artifact is configured.",
            )
        return {"status": "ready"}

    @app.get("/metadata")
    def metadata() -> Mapping[str, Any]:
        active = app.state.runtime
        if active is None:
            raise HTTPException(status_code=503, detail="Model is not ready.")
        return dict(active.metadata())

    @app.get("/metrics")
    def metrics() -> Mapping[str, Any]:
        return dict(app.state.metrics.snapshot())

    def predict(payload: Any) -> Mapping[str, Any]:
        active = app.state.runtime
        if active is None:
            raise HTTPException(status_code=503, detail="Model is not ready.")
        started = time.perf_counter()
        sample_count = len(payload.inputs)
        event: dict[str, Any] = {
            "artifact_id": active.metadata().get("model_id"),
            "batch_size": sample_count,
            "device": str(active.device),
            "status": "ok",
        }
        try:
            result = active.predict(
                np.asarray(payload.inputs),
                context=(np.asarray(payload.context) if payload.context is not None else None),
                batch_size=payload.batch_size,
                return_logits=payload.return_logits,
            )
            return {
                "values": _json_value(result.values),
                "task": result.task,
                "output_names": list(result.output_names),
                "artifact_version": result.artifact_version,
                "model_id": result.model_id,
                "run_id": result.run_id,
                "metadata": _json_value(result.metadata),
                "top_k": _json_value(result.top_k),
            }
        except ValueError as exc:
            event["status"] = "error"
            event["error_type"] = type(exc).__name__
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            event["status"] = "error"
            event["error_type"] = type(exc).__name__
            LOGGER.exception("PSANN inference request failed")
            raise HTTPException(status_code=500, detail="Inference failed.") from exc
        finally:
            event["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
            app.state.metrics.record(event)

    predict.__annotations__["payload"] = PredictionRequest
    app.post("/predict")(predict)

    return app


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve one native PSANN artifact.")
    parser.add_argument(
        "--artifact",
        default=os.environ.get("PSANN_ARTIFACT_PATH"),
        help="Path to a mounted .psann artifact (or set PSANN_ARTIFACT_PATH).",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the reference service through Uvicorn."""

    arguments = _parser().parse_args(argv)
    if not arguments.artifact:
        raise SystemExit("--artifact or PSANN_ARTIFACT_PATH is required.")
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - dependency guidance path
        raise SystemExit("Install the service extra with `pip install 'psann[serve]'`.") from exc
    app = create_app(
        artifact_path=arguments.artifact,
        config=InferenceConfig(batch_size=arguments.batch_size, device=arguments.device),
    )
    uvicorn.run(app, host=arguments.host, port=arguments.port)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


__all__ = ["ServiceMetrics", "create_app", "main"]
