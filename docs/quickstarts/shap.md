# SHAP Explanations

Install the optional explanation stack and explain the same raw-input contract used
in deployment:

```bash
python -m pip install "psann[explain]"
```

```python
import psann

runtime = psann.load_runtime("artifacts/model.psann")
result = runtime.explain(
    rows_to_explain,
    background=reviewed_reference_rows,
    config=psann.ExplainerConfig(
        algorithm="permutation",
        max_background_samples=50,
        max_explanation_samples=8,
        max_evaluations=256,
        seed=17,
    ),
)

shap_explanation = result.explanation
print(result.output_names, result.metadata["additivity_error"])
```

The background is always explicit; PSANN never recovers training rows from an
artifact. For a classifier, select a named probability output:

```python
result = runtime.explain(
    rows_to_explain,
    background=reviewed_reference_rows,
    config=psann.ExplainerConfig(
        output_kind="probability",
        output="approved",
    ),
)
```

Spatial and sequence inputs retain their logical shape and receive domain groups.
Gradient/deep explanations are capability-gated. Unsupported custom preprocessing or
explicit context falls back with a recorded reason, or raises when
`fallback="error"` is configured.
