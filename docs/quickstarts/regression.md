# Tabular Regression

Use named features and keep preprocessing inside the model artifact:

```python
import pandas as pd
import psann

train_x = pd.DataFrame(rows, columns=["amount", "tenure", "utilization"])
train_y = pd.Series(targets, name="demand")

spec = psann.ModelSpec(
    input_schema=psann.DataSchema(
        feature_names=tuple(train_x.columns),
        output_names=("demand",),
        input_shape=(3,),
        feature_policy="reorder",
        target_scaling={"kind": "standard"},
    ),
    parameters={
        "hidden_layers": 2,
        "hidden_units": 32,
        "scaler": "standard",
        "target_scaler": "standard",
        "random_state": 7,
    },
)

run = psann.train(
    psann.create_model(spec),
    (train_x, train_y),
    validation_data=(validation_x, validation_y),
    config=psann.TrainingConfig(
        epochs=50,
        batch_size=128,
        early_stopping=True,
        patience=5,
        deterministic=True,
    ),
)

print(run.evaluate((validation_x, validation_y)))
artifact = run.export("artifacts/demand.psann")
runtime = psann.load_runtime(artifact)
predictions = runtime.predict(batch_x).values
```

The artifact carries the fitted input and target scaler state. A reordered dataframe
is accepted only because the schema explicitly selected `feature_policy="reorder"`;
missing or unexpected columns still fail.
