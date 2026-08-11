# Classification

## Binary probabilities and threshold policy

```python
import psann

spec = psann.ModelSpec(
    task=psann.TaskSpec(
        kind="binary",
        positive_label="approved",
        threshold=0.65,
    ),
    input_schema=psann.DataSchema(
        feature_names=("income", "risk", "tenure"),
        input_shape=(3,),
    ),
    parameters={"hidden_layers": 2, "hidden_units": 32, "random_state": 11},
)
run = psann.train(
    psann.create_model(spec),
    (train_x, train_labels),
    config=psann.TrainingConfig(epochs=25, metrics=("accuracy",)),
)

artifact = run.export("artifacts/approval.psann")
probabilities = psann.load_runtime(artifact).predict(batch_x)
labels = psann.load_runtime(
    artifact,
    config=psann.InferenceConfig(classification_output="label"),
).predict(batch_x)
```

Binary probability columns follow `model.classes_`; label output applies the
serialized threshold policy.

## Multiclass top-k

```python
runtime = psann.load_runtime(
    "artifacts/tier.psann",
    config=psann.InferenceConfig(top_k=2),
)
result = runtime.predict(batch_x)

full_probability_matrix = result.values
ranked_labels = result.top_k.labels
ranked_probabilities = result.top_k.probabilities
```

`top_k` is valid only for multiclass probability output. It fails if `k` exceeds the
fitted class count or if label/logit output is selected.
