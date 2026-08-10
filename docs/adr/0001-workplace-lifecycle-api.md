# ADR 0001: Workplace Lifecycle API

- Status: Accepted
- Date: 2026-07-27
- Decision owner: Nickm1128
- Tracking issue: https://github.com/Nickm1128/psann/issues/2

## Context

PSANN currently exposes effective sklearn-style regressors, but creation, training,
persistence, deployment, and explainability do not share one workplace-level contract.
The existing estimators must remain useful while a higher-level lifecycle is added.

## Decision

PSANN will add a typed high-level lifecycle API without duplicating the estimator
training implementation.

The stable target surface is:

- `TaskSpec`
- `ModelSpec`
- `TrainingConfig`
- `DataSchema`
- `InferenceConfig`
- `create_model(spec)`
- `train(model, train_data, validation_data=...) -> TrainingRun`
- `TrainingRun.export(path)`
- `load_model(path, device=...)`
- `explain(model_or_artifact, data, ...)`

The high-level objects must be serializable without arbitrary Python callables.
Registries may resolve approved extension identifiers into runtime implementations.

### API ownership

The sklearn-style estimators remain the compatibility and direct-use layer:

- constructors and `get_params` / `set_params`;
- `fit`, `predict`, and `score`;
- documented sequence, stateful, HISSO, and diagnostic methods;
- class-specific legacy `save` / `load` during the deprecation window in ADR 0005.

The high-level API owns:

- task-aware model construction;
- data and output schemas;
- training-run identity and structured history;
- resumable training checkpoints;
- safe deployment artifacts and generic loading;
- batched deployment inference;
- explainability orchestration.

Estimator `fit(...)` continues to return `self` for sklearn compatibility. The
high-level `train(...)` function returns a `TrainingRun`. Both call the same internal
fit and training helpers.

### Target workflow

```python
spec = psann.ModelSpec(
    task=psann.TaskSpec(kind="regression"),
    backbone="respsann_mlp",
    input_schema=psann.DataSchema(...),
)

model = psann.create_model(spec)
run = psann.train(model, train_data, validation_data=validation_data)
artifact = run.export("artifacts/model.psann")

deployed = psann.load_model(artifact, device="cpu")
predictions = deployed.predict(batch)
explanation = deployed.explain(batch, background=reference_data)
```

### Compatibility rules

- Existing stable estimators are not reimplemented behind a second training loop.
- High-level configuration uses canonical parameter names only.
- Legacy estimator aliases remain governed by ADR 0005 and
  `docs/deprecation_policy.md`.
- Public exceptions must identify the invalid field, received value, expected values,
  and lifecycle stage.
- Experimental capabilities must be explicitly marked in types, documentation, and
  artifact manifests.

## Consequences

- Users can adopt the new lifecycle incrementally.
- The estimator layer remains suitable for sklearn workflows.
- New task, artifact, and explanation behavior has one orchestration boundary.
- Internal refactors must preserve both estimator characterization tests and high-level
  lifecycle tests.

## Rejected alternatives

- Replacing the estimators with an unrelated platform API would create two training
  implementations and break existing users.
- Making raw `torch.nn.Module` the only public contract would lose scaling, schema,
  task, persistence, and sklearn behavior.
- Adding deployment and SHAP methods independently to every estimator subclass would
  duplicate policy and complicate compatibility.
