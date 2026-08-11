# ADR 0002: Task and Backbone Support

- Status: Accepted
- Date: 2026-07-27
- Decision owner: Nickm1128
- Tracking issue: https://github.com/Nickm1128/psann/issues/2

## Context

The stable public estimator surface is regression-oriented even though the underlying
Torch modules can produce classification logits. A workplace platform needs task-owned
losses, metrics, probability conversion, schemas, and deployment behavior.

## Decision

### Stable task contract for the workplace milestone

The first stable platform milestone supports:

1. regression;
2. binary classification;
3. multiclass classification;
4. multilabel classification.

Task adapters, not backbones, own target validation, default loss, metrics, output
activation, class labels, probability conversion, and threshold policy.

The required estimator behavior is:

| Task | Required prediction contract |
| --- | --- |
| Regression | Numeric predictions with named outputs where provided |
| Binary classification | Class labels, one positive-class probability, and logits when requested |
| Multiclass classification | Class labels and a probability column for each fitted class |
| Multilabel classification | Boolean/label matrix, per-label probability, and explicit thresholds |

### Stable registered backbone identifiers

The target stable registry contains:

- `psann_mlp`
- `respsann_mlp`
- `psann_conv1d`
- `psann_conv2d`
- `psann_conv3d`
- `respsann_conv2d`
- `wave_resnet`
- `sgr_psann`

A backbone becomes stable only after it passes every applicable task, schema,
training, artifact, deployment, and explanation contract. A registered identifier may
therefore remain "candidate" until its matrix is complete.

GeoSparse remains experimental. It may use the shared registry and artifacts for
testing, but its manifest and public result metadata must retain an experimental flag
until it passes the common stable matrix.

LSM remains a preprocessor/expander capability rather than a task or backbone.

### Stable activation identifiers

The target stable activation registry contains:

- `relu`
- `tanh`
- `sigmoid`
- `gelu`
- `silu`
- `psann`
- `relu_sigmoid_psann`

Parameterized activation configuration must be structured and validated. Mixed
activation remains experimental until its serialization and export behavior is
specified and tested.

### Extension tiers

- **Core registered**: full task, artifact, native deployment, and applicable SHAP
  guarantees.
- **Optional registered plugin**: native artifact support when the plugin and version
  are available; export support is capability-declared.
- **Arbitrary module adapter**: training and in-process inference only by default.
  Safe artifacts and exports require explicit registration.

No arbitrary Python callable is embedded in a stable model specification or deployment
artifact.

## Consequences

- Classification becomes a first-class task rather than an example-specific raw Torch
  loop.
- Output semantics stay consistent across backbones.
- Stable support is evidence-based per capability, not inferred from whether a module
  can execute a forward pass.
- Custom research remains possible without weakening the stable artifact trust model.

## Rejected alternatives

- Treating one-hot classification as regression would not provide correct losses,
  probabilities, metrics, or sklearn semantics.
- Promoting every existing research backbone immediately would make "stable" support
  unverifiable.
- Serializing arbitrary modules and callables would conflict with the safe artifact
  contract.
