# SHAP Explainability

Workplace Phase 6 adds optional SHAP explanations over the same raw-input contract used by
`InferenceRuntime`. Preprocessing, layout conversion, task postprocessing, feature
names, and artifact identity are therefore shared with deployed prediction instead of
being reimplemented by application code.

## Install

```bash
python -m pip install "psann[explain]"
```

The base package does not import or install SHAP. The extra selects SHAP 0.50 or 0.51
on Python 3.11 and SHAP 0.50 through 0.52 on Python 3.12 and newer. SHAP 0.50+ uses
NumPy 2, so do not combine `psann[explain]` with the NumPy 1.26 `compat` snapshot.
Use a separate explanation environment if another application still requires NumPy
1.x.

See the official
[`shap.Explainer` documentation](https://shap.readthedocs.io/en/stable/generated/shap.Explainer.html)
and [`shap.Explanation`
documentation](https://shap.readthedocs.io/en/stable/generated/shap.Explanation.html)
for the upstream objects returned by this integration.

## Deployed raw-input workflow

An explanation background is always explicit. PSANN never recovers or silently uses
training data from a model or artifact.

```python
import psann

runtime = psann.load_runtime("artifacts/demand_forecast.psann")
config = psann.ExplainerConfig(
    algorithm="auto",
    output_kind="prediction",
    max_evaluations=2048,
    max_explanation_samples=32,
    seed=7,
)
explainer = runtime.make_explainer(
    reference_data=approved_reference_rows,
    config=config,
)
result = explainer.explain(raw_rows)

shap_values = result.explanation  # a standard shap.Explanation
print(result.feature_groups)
print(result.artifact_version, result.model_id, result.run_id)
```

The equivalent one-shot form is:

```python
result = runtime.explain(
    raw_rows,
    background=approved_background_rows,
    config=config,
)
```

`psann.make_explainer(...)` and `psann.explain(...)` also accept a fitted estimator.
For workplace use, prefer a loaded runtime so the explanation is tied to a validated
deployment artifact.

## Background policy

Exactly one source is required:

- `background=` uses the supplied rows directly, bounded by
  `max_background_samples`.
- `reference_data=` samples `background_size` rows deterministically using `seed`.
- `summary=` uses an existing `BackgroundSummary`.

To create a reviewable summary:

```python
summary = psann.summarize_background(
    approved_reference_rows,
    max_samples=50,
    seed=7,
    approved_for_persistence=True,
    metadata={"cohort": "2026-Q2 validation"},
)
psann.save_explainer_config(
    "explain.json",
    config,
    background=summary,
    include_background=True,
)
```

Configuration is persisted separately from `.psann`. Background values are written
only when `include_background=True` and the summary is marked
`approved_for_persistence=True`. Without both approvals, values are omitted or the
operation fails. Treat a persisted background as sensitive data.

## Outputs and shapes

`ExplanationResult.explanation` is a `shap.Explanation`; `values` and `base_values`
are convenience properties. Attribution values use:

```text
(samples, *raw_input_shape, selected_outputs)
```

Base values use `(samples, selected_outputs)`. The default output contract is:

| Task | Default explained output |
| --- | --- |
| Regression | Prediction |
| Multi-output regression | Every named prediction |
| Binary classification | Probability for both classes |
| Multiclass classification | Probability for every class |
| Multilabel classification | Probability for every label |

Use `output_kind="logit"` for classification logits and `output=` with an output name
or zero-based index to select one output. Output and feature names come from the
fitted schema/task contract.

## Model-agnostic maskers and groups

`algorithm="auto"` chooses permutation explanations for rank-one tabular inputs and
partition explanations for spatial or sequence inputs. These paths call the public
raw-input inference runtime, so they support deployed regression and classification
artifacts without bypassing preprocessing.

Masker choices are:

- `independent`: tabular marginal masking against the explicit background;
- `partition`: correlation-based hierarchical masking;
- `domain`: a partition tree that keeps declared time, channel, or spatial groups
  together before merging groups.

`group_strategy="auto"` selects individual features for tabular input, time-step
groups for flattened Wave/SGR sequences, and spatial-region groups for convolutional
or shape-preserving input. Explicit strategies are `feature`, `time_step`, `channel`,
and `spatial_region`. `data_format` controls which tensor axis is treated as channels.

Groups describe the explanation game; they do not prove that grouped variables are
independent or causal.

## Gradient and deep explanations

Gradient algorithms are opt-in:

```python
config = psann.ExplainerConfig(
    algorithm="gradient",
    gradient_samples=200,
    fallback="error",
)
result = runtime.explain(
    raw_rows,
    background=approved_background_rows,
    config=config,
)
```

`DifferentiableInferenceAdapter` clones the fitted Torch module, switches the clone to
evaluation mode, freezes parameters and state updates, and includes certified
built-in standard/min-max input scaling, target inverse scaling, tensor layout, cosine
context, and task output conversion.

`GradientExplainer` is capability-gated for registered backbones and differentiable
built-in transforms. `DeepExplainer` is narrower: it is certified for dense
`psann_mlp` and `respsann_mlp` models using ReLU, without stateful execution. Custom
scalers, callable context builders, explicit context, categorical/imputation schema
transforms, and per-element outputs are not certified for the gradient path.

With the default `fallback="model_agnostic"`, an unsupported request falls back to
permutation or partition and records `fallback_reason` in the result metadata. Use
`fallback="error"` when algorithm identity is a hard requirement.

Registered intermediate-layer aliases can be inspected with:

```python
print(psann.list_explainable_layers(runtime))
layer_result = runtime.explain(
    raw_rows,
    background=approved_background_rows,
    config=psann.ExplainerConfig(
        algorithm="gradient",
        layer="hidden_0",
        fallback="error",
    ),
)
```

Applications register additional reviewed aliases with
`register_explainable_layer(backbone, name, module_path)` rather than accepting module
paths from request input.

## Bounds, validation, and reporting

`ExplainerConfig` bounds background rows, explained rows, model evaluations, SHAP
batch size, and gradient samples. Keep service explanation endpoints separately
rate-limited; explanation work is substantially more expensive than prediction.

Every result records algorithm, fallback, masker, group strategy, background policy,
limits, input layout, output contract, state policy, artifact/run identity, and the
observed additivity error. PSANN validates finite numeric inputs and raw shape before
calling SHAP.

Offline aggregate comparison is available without writing row-level features:

```python
drift = psann.summarize_explanation_drift(reference_result, current_result)
psann.write_explanation_report("reports/explanation_drift.json", drift)
```

The report contains aggregate attribution drift and metadata, not explanation rows.

## Interpretation and privacy limitations

- SHAP values explain a model relative to the selected background and output; changing
  either can materially change the values.
- Attribution is not evidence of causation, intervention effect, model fairness, or
  feature necessity.
- Correlated inputs can share or redistribute credit. A partition/domain masker makes
  the coalition game more explicit but does not remove this ambiguity.
- A small or unrepresentative background can produce unstable or misleading
  explanations. Approve it as carefully as evaluation data.
- Raw feature values, background rows, output names, feature names, and attributions
  can expose sensitive information. Do not put them in service logs or broadly
  accessible artifacts.
- Additivity is checked numerically, not symbolically. Gradient estimates are
  approximate and require algorithm-specific tolerances.
- PSANN does not currently certify explanation requests that require caller-supplied
  context. Define and review the raw-input game before extending that boundary.

The Phase 6 test matrix covers task/output shapes, names, additivity, fixed-seed
determinism, artifact parity, preprocessing/layout parity, spatial and sequence
groups, state isolation, explicit context failures, fallbacks, and report generation.
