# Supervised Training Core

Status: Active

This document defines the supported optimizer-driven training behavior shared by the
core sklearn-style regressors. HISSO remains a separate episodic loop and rejects
supervised-only options such as `scheduler`, `metrics`, `callbacks`, `resume_from`,
and `checkpoint_dir`.

## Fail-fast configuration

`fit` validates the following before constructing a model:

- optimizer: `adam`, `adamw`, or `sgd`;
- loss: `mse`/`l2`, `l1`/`mae`, `smooth_l1`/`huber_smooth`, `huber`, or a callable;
- loss reduction: `mean` or `sum`;
- scheduler: `none`, `step`, or `cosine`;
- positive dimensions, epoch count, batch size, learning rates, and loss parameters;
- early-stopping patience, state-reset mode, data layout, output shape, and
  incompatible attention/LSM or scheduler/LR-bound combinations;
- requested device, fallback policy, non-finite policy, callback policy, metrics, and
  checkpoint-retention values.

`loss_reduction="none"` is deliberately rejected for optimizer-driven training because
PyTorch backward requires a scalar loss. A custom loss may return a tensor when
`mean` or `sum` is selected; PSANN applies that reduction before backward.

Before optimizer creation, PSANN runs a no-gradient sample forward and requires the
prediction and target shapes to match exactly. This prevents silent broadcasting in
loss functions.

## Input and non-finite policies

The data boundary has a fixed workplace-safe policy:

- `NaN` means missing data and is rejected in training, validation, target, context,
  and noise arrays. Impute missing values before calling `fit`.
- Positive or negative infinity is rejected at the same boundary.

Runtime loss or gradient non-finite behavior is controlled with:

| `nonfinite_policy` | Behavior |
| --- | --- |
| `error` | Default. Emit `nonfinite_step` and raise before the optimizer step. |
| `skip_step` | Clear gradients, skip only the affected step, and count it in history. |
| `continue` | Explicitly opt into the previous unsafe behavior; record the event and continue. |

If every batch in an epoch is skipped, training raises because the epoch made no
optimizer progress.

## Schedulers and custom metrics

```python
model.fit(
    X,
    y,
    scheduler="step",
    scheduler_params={"step_size": 5, "gamma": 0.5},
    metrics={
        "mae": lambda prediction, target: (prediction - target).abs().mean(),
    },
)
```

`cosine` accepts `t_max` (default: estimator `epochs`) and `eta_min` (default: `0.0`).
The legacy `lr_max`/`lr_min` pair remains a linear per-epoch schedule and cannot be
combined with a named scheduler.

Metrics receive detached prediction and target tensors. Tensor-valued metric outputs
are averaged to a scalar. Metric values appear in history as `train_<name>` and, when
validation data is supplied, `val_<name>`. Metrics do not affect the loss or backward
pass.

## Structured events and logging

Pass callbacks and an optional standard-library logger to `fit`:

```python
import logging

events = []
logger = logging.getLogger("work.models.demand")

model.fit(
    X,
    y,
    callbacks=[events.append],
    logger=logger,
    callback_error_policy="raise",
)
```

Callbacks receive a frozen `psann.TrainingEvent`. Stable event names are:

- `train_start` and `train_end`;
- `epoch_start` and `epoch_end`;
- `validation_end`;
- `nonfinite_step` and `fallback`;
- `checkpoint`;
- `early_stop`;
- `failure`.

Callback and legacy `gradient_hook`/`epoch_callback` exceptions raise by default.
`callback_error_policy="warn"` emits a runtime warning and continues. The `failure`
event is recorded before a training exception is re-raised.

`train_start` metadata records the effective device and dtype, seed, deterministic
mode, optimizer and learning rates, parameter counts, input/target shapes, scheduler,
AMP/compile decisions, and fallback policies. Per-epoch history records losses,
metrics, timings, learning rates, gradient norms, attempted/successful/skipped steps,
early-stopping state, and cumulative training time.

## Fallback policy

`fallback_policy` controls device, AMP, and compilation downgrades:

- `warn` (default): issue a runtime warning, emit `fallback`, and use the supported
  alternative;
- `error`: raise instead of changing the requested behavior.

An unavailable explicitly requested accelerator falls back to CPU only under `warn`.
`device="auto"` is a selection, not a fallback. AMP and the current compile path
require CUDA; requesting either on CPU follows the same policy. When `amp=True`,
`amp_dtype` must be float16 or bfloat16; disable AMP for float32 training.

## Deterministic mode

`deterministic=True` enables PyTorch deterministic algorithms, disables cuDNN
benchmark selection, and combines the estimator's `random_state` with a dedicated
data-loader generator and seeded workers.

Determinism is strongest when resuming on the same OS, device class, PyTorch build,
model configuration, and input data. Different accelerators, library versions,
threading implementations, or unsupported deterministic kernels may raise or produce
small numerical differences. Deterministic algorithms can reduce throughput.

## Resumable `.psann-train` checkpoints

```python
from pathlib import Path

checkpoint_dir = Path("runs/demand/checkpoints")

model.fit(
    X,
    y,
    deterministic=True,
    checkpoint_dir=checkpoint_dir,
    checkpoint_every=5,
    checkpoint_keep=3,
)

resumed = PSANNRegressor(
    hidden_layers=model.hidden_layers,
    hidden_units=model.hidden_units,
    epochs=model.epochs,
    batch_size=model.batch_size,
    random_state=model.random_state,
)
resumed.fit(
    X,
    y,
    deterministic=True,
    resume_from=checkpoint_dir / "latest.psann-train",
)
```

Every completed epoch atomically replaces `latest.psann-train`. An improving monitored
loss replaces `best.psann-train`. `checkpoint_every=N` additionally creates periodic
`epoch_XXXXXX.psann-train` files, bounded by `checkpoint_keep`.

The checkpoint persists:

- model, optimizer, named scheduler, and AMP scaler state;
- completed epoch, global optimizer step, best metric/epoch, patience, and history;
- Python, NumPy, Torch, CUDA, and data-loader RNG state;
- built-in input/target scaler state;
- model and hashed training-data signatures used for compatibility checks.

The file is an atomic ZIP container with a JSON manifest, SHA-256 checksum, and a
tensor/primitive state payload loaded with `weights_only=True`. Checksums detect
corruption but do not authenticate who produced a file.

Custom scaler objects are rejected when automatic checkpointing is enabled because
opaque Python objects are outside the restricted state contract.

### Training versus deployment persistence

`.psann-train` contains optimizer and resume state and is accepted only by
`fit(..., resume_from=...)`. `PSANNRegressor.load` rejects it.

The existing estimator `save`/`load` path is a trusted legacy inference snapshot. It
does not contain the state required for exact resume. Phase 4 owns the portable
deployment `.psann` bundle; a training checkpoint must never be served as a deployment
artifact.
