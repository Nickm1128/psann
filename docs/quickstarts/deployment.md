# Deploy a Native Artifact

The `.psann` bundle is the deployment source of truth:

```python
import psann

info = psann.inspect_artifact("artifacts/model.psann")
print(info.backbone, info.task, info.artifact_format_version)

runtime = psann.load_runtime(
    "artifacts/model.psann",
    config=psann.InferenceConfig(
        batch_size=256,
        device="cpu",
        device_transfer="per_batch",
        fallback_policy="error",
    ),
)
result = runtime.predict(batch)
print(result.values, result.metadata["chunks"])
```

`per_batch` keeps host-to-device transfer bounded. Ordinary runtime calls are
stateless; a fitted stateful model requires an explicit isolated session:

```python
with runtime.create_session(session_id=request_id) as session:
    first = session.step(row).values
    sequence = session.predict_sequence(next_rows).values
```

To expose the optional reference worker:

```bash
python -m pip install "psann[serve]"
python -m psann.serving --artifact artifacts/model.psann --device cpu
```

Use `/health`, `/ready`, `/metadata`, and `/predict`. The reference worker is an
inference process, not a hosted registry or control plane.
