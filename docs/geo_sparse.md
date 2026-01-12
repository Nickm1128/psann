# Geo-Sparse Networks (Geometric Connectivity)

This module implements a geometric-connectivity network where each neuron connects to a fixed fan-in subset of the next layer. Layers are arranged on a 2D grid and stacked through depth, with ResNet-style residual blocks for stability.

## Concepts and conventions

- **Fan-in k**: each output neuron connects to exactly `k` input neurons.
- **Geometry**: layer shape is `(H, W)` with `n = H * W`. Flatten order is row-major: `idx = row * W + col`.
- **Connectivity patterns**:
  - `local`: choose from fixed offsets in a radius window around each output.
  - `random`: sample from the same candidate set with a deterministic seed.
  - `hash`: deterministic per-output sampling using a seeded PRNG.
- **Edge handling**: `wrap_mode` in `{clamp, wrap}` controls how offsets map to valid indices.

## Components

- `GeoSparseLinear` in `src/psann/layers/geo_sparse.py`:
  - Fixed connectivity (precomputed indices).
  - Gather or scatter compute mode.
  - Autograd-friendly, no custom backward.
- `GeoSparseResidualBlock` and `GeoSparseNet` in `src/psann/nn_geo_sparse.py`:
  - Pre-norm residual blocks with `DropPath` and learnable residual scale `alpha`.
  - Activation options: `psann`, `phase_psann`, `relu`, `tanh`.
  - Activation config accepts both `ActivationConfig` and LM-style keys like `amp_init`, `freq_init`, `damp_init`, `trainable`, and bounds.

## Parameter scaling

Use helpers from `src/psann/params.py`:

- Sparse layer params: `n_out * k + (bias ? n_out : 0)`
- Dense layer params: `n_in * n_out + (bias ? n_out : 0)`
- Geo-sparse net params: `depth * (2 * sparse_layer + alpha) + head`

See `geo_sparse_net_params` and `dense_mlp_params` for exact formulas.

## Minimal usage (PyTorch)

```python
import torch
from psann.nn_geo_sparse import GeoSparseNet

model = GeoSparseNet(
    input_dim=16,
    output_dim=1,
    shape=(4, 4),
    depth=4,
    k=8,
    pattern="local",
    radius=1,
    activation_type="psann",
    activation_config={"amp_init": 1.0, "freq_init": 1.0, "damp_init": 0.1},
)

x = torch.randn(32, 4, 4)
y = model(x)
print(y.shape)  # (32, 1)
```

## Minimal usage (sklearn-style)

```python
import numpy as np
from psann import GeoSparseRegressor

X = np.random.randn(256, 4, 4).astype("float32")
y = X.reshape(X.shape[0], -1).sum(axis=1, keepdims=True).astype("float32")

est = GeoSparseRegressor(
    shape=(4, 4),
    hidden_layers=4,
    k=8,
    activation_type="relu",
    epochs=10,
    batch_size=64,
    lr=1e-3,
    random_state=0,
)
est.fit(X, y, verbose=0)
preds = est.predict(X[:10])
print(preds.shape)  # (10, 1)
```

## Benchmark script (matched parameters)

Runs GeoSparseNet vs dense ReLU MLP with matched parameter counts:

```bash
python scripts/benchmark_geo_sparse_vs_dense.py --shape 8x8 --depth 6 --k 8 --device cuda --amp --amp-dtype bfloat16 --tf32
```

Outputs:
- `reports/geo_sparse/<timestamp>/manifest.json`
- `reports/geo_sparse/<timestamp>/results.json`
- `reports/geo_sparse/<timestamp>/summary.csv`

## Microbenchmark (layer/block timing)

```bash
python scripts/benchmark_geo_sparse_micro.py --shape 16x16 --k 8 --device cuda --amp --amp-dtype bfloat16 --tf32
```

Outputs:
- `reports/geo_sparse_micro/<timestamp>/manifest.json`
- `reports/geo_sparse_micro/<timestamp>/results.json`
- `reports/geo_sparse_micro/<timestamp>/summary.csv`

## Experiment sweep plan

Run a sweep over `k`, depth, geometry size, and activation choices, then aggregate results:

```bash
python scripts/geo_sparse_sweep.py --shapes 4x4,8x8 --depths 4,8 --ks 4,8,16 --activations relu,psann --seeds 0,1 --device cuda --amp --amp-dtype bfloat16 --tf32 --plot
```

Outputs:
- `reports/geo_sparse_sweep/<timestamp>/summary.csv`
- `reports/geo_sparse_sweep/<timestamp>/summary_by_model.json`
- `reports/geo_sparse_sweep/<timestamp>/summary_plot.png` (if matplotlib available)

## Large sweep (GPU-focused)

Suggested larger sweep (use `--resume` so it can be restarted safely):

```bash
python scripts/geo_sparse_sweep.py --shapes 8x8,16x16 --depths 4,8,12 --ks 4,8,16,32 --activations relu,psann --seeds 0,1,2 --device cuda --epochs 25 --batch-size 256 --train-size 8192 --test-size 2048 --amp --amp-dtype bfloat16 --tf32 --compile --match-tolerance 0.01 --resume --plot
```

Tuning notes:
- Reduce `--train-size` / `--batch-size` if GPU memory is limited.
- Keep `--compile` on for longer sweeps; compile time is tracked per run.

## Performance notes for Grace Blackwell GPUs

- Prefer `bf16` with `--amp --amp-dtype bfloat16` and enable `--tf32` for stable matmul performance.
- Use `--compile` to measure `torch.compile` speedups; compile time is reported separately.
- Keep shapes and `k` static to maximize kernel reuse.
