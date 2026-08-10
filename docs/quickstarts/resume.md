# Resume Interrupted Training

Resume checkpoints and deployment artifacts are deliberately different:

```python
import psann

checkpoint_dir = "runs/demand/checkpoints"
first = psann.train(
    psann.create_model(spec),
    (train_x, train_y),
    validation_data=(validation_x, validation_y),
    config=psann.TrainingConfig(
        epochs=10,
        deterministic=True,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=1,
        checkpoint_keep=3,
    ),
)

resumed = psann.train(
    psann.create_model(spec),
    (train_x, train_y),
    validation_data=(validation_x, validation_y),
    config=psann.TrainingConfig(
        epochs=25,
        deterministic=True,
        resume_from=f"{checkpoint_dir}/latest.psann-train",
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=1,
        checkpoint_keep=3,
    ),
)

resumed.export("artifacts/demand.psann")
```

Resume validates the model, training data, deterministic mode, and scheduler
contract. A `.psann-train` file cannot be loaded for deployment; export the completed
`TrainingRun` to `.psann`.
