Lightweight Colab Tests for PSANN: Targeted Replications + EAF Lite

What this is
- A small, compute-light test suite to replicate key findings with a couple of seeds and run a minimal real EAF next-step TEMP forecast. It complements the heavier parity suite and is designed to finish quickly on a Colab GPU/CPU.

Includes
- Jena (36h horizon, reduced subset) — PSANN conv-spine vs MLP, 2 seeds
- Beijing (single station, 6h horizon, reduced subset) — PSANN conv-spine vs MLP, 2 seeds
- EAF (real eaf_temp.csv, next-step TEMP, few heats) — PSANN conv-spine vs MLP
- Optional Jacobian participation ratio snapshots (start/mid/end) on Jena subset

How to run in Colab (do not run locally)
1) Open a new Google Colab notebook (GPU optional; CPU works but slower).
2) Upload the folder `colab_tests_light/` to your Colab working directory.
3) Ensure datasets are present:
   - Option A: Upload `datasets.zip` (from this repo) to the Colab working dir; the script will extract to `datasets/`.
   - Option B: Upload the `datasets/` directory directly.
4) Install dependencies (usually preinstalled on Colab):
   ```
   !pip -q install torch torchvision torchaudio pandas numpy scikit-learn
   ```
5) Run the script (examples):
   ```
   # Run all light tasks with two seeds on GPU if available
   !python colab_tests_light/run_light_probes.py --tasks jena,beijing,eaf --seeds 7 8 --device auto

   # Only EAF lite (fastest path), CPU only
   !python colab_tests_light/run_light_probes.py --tasks eaf --device cpu

   # Add Jacobian PR snapshots for Jena
   !python colab_tests_light/run_light_probes.py --tasks jena --pr-snapshots
   ```

Outputs
- `colab_results_light/metrics.csv` — aggregated metrics per task/model/seed
- `colab_results_light/jacobian_pr.csv` — (optional) PR snapshots for Jena

Notes
- These runs are intentionally light: small subsets, 8–12 epochs, small models. They are meant to tighten confidence without expanding scope or compute.
- If you need parity against additional baselines (e.g., TCN/LSTM), increase `--epochs` moderately and add models via `--models` once you have time budget.

