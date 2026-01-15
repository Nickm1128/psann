# GeoSparse Benchmarks Summary (Short)

This is a brief index for the GeoSparse benchmark work. The full, detailed report
with raw outputs and environment metadata lives in:

- `docs/geo_sparse_benchmarks_report.md`

## What was tested

- Grid sweeps over shapes (8x8, 16x16), depths (4, 8, 12), and fan-in k (4, 8, 16, 32).
- GeoSparse activations: ReLU and PSANN.
- Dense baselines: ReLU and PSANN/Res variants (see per-sweep notes in the full report).
- Seeds: 0, 1, 2.

## Tasks and data

The benchmarks use synthetic regression tasks defined in the sweep scripts. The
exact task mix and scaling settings are recorded per sweep in the full report.

## Reproducing a sweep

See `docs/geo_sparse.md` for the canonical sweep command, or run:

```bash
python scripts/geo_sparse_sweep.py --shapes 8x8,16x16 --depths 4,8,12 --ks 4,8,16,32 \
  --activations relu,psann --seeds 0,1,2 --device cuda --epochs 25 --batch-size 256 \
  --train-size 8192 --test-size 2048 --amp --amp-dtype bfloat16 --tf32 --compile --resume --plot
```

## Notes

- Use the full report for numerical takeaways; this summary is intentionally lightweight.
- Outputs are written under `reports/geo_sparse_sweep/<timestamp>/`.

