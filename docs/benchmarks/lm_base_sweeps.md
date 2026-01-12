# PSANN-LM Base Sweeps (2025-12)

This note summarizes what we learned from the recent quick proxy sweeps (small-model training runs)
so we can apply it to the next full-scale PSANN-LM pretraining run.

## What We Ran (Proxy Benchmark Setup)

All results below come from `scripts/bench_lm_bases.py` runs that are intentionally *apples-to-apples*
on the following axes:

- **Dataset**: `iohadrubin/wikitext-103-raw-v1` (HF streaming), `train` + `validation` splits.
- **Tokenizer**: fixed/reused HF `tokenizers` model at `runs/tokenizers/base_param_sweep_hypotheses_v1_small` (vocab=16384).
- **Budget**: `max_steps=500` with `batch_tokens=32768` (≈16.38M train tokens per run).
- **Model size**: ~13.1M parameters (proxy for architecture comparisons).
- **Seeds**: typically `[1337, 1338]` to sanity-check noise.
- **Logging**: `log_interval_steps=25` (plus `ppl_avg25` rolling average in stdout logs).

Important caveat: this benchmark is **not compute-matched** across bases. It is meant to compare
*sample efficiency* (loss/ppl per token) more than wall-clock efficiency.

## Summary of Key Outcomes

### 1) WaveResNet beats the transformer baseline (per token)

On the same proxy setup (same data/tokenizer/budget), WaveResNet achieved substantially lower
validation loss/perplexity than an equivalently sized transformer.

Representative results:

- **Transformer @ lr=0.0022** (2 seeds): mean `val_loss≈5.5697`, mean `val_ppl≈262.4`  
  Source: `reports/benchmarks/20251229_134407_transformer_lr0.0022_vs_waveresnet_v1_small/summary.csv`
- **WaveResNet (best earlier grid)**: mean `val_loss≈5.3501`, mean `val_ppl≈210.6`  
  (`lr=0.0022`, `freq_init=2.25`, `freq_init_std=0.25`)  
  Source: `reports/benchmarks/20251228_182315_waveresnet_freq_gauss_tune_v2_small/summary.csv`

That gap is large enough (Δloss ≈ 0.22 nats/token ≈ 0.32 bits/token) that it’s unlikely to be
explained by seed noise alone under this proxy.

### 2) Learning rate is the dominant knob in the proxy sweeps

Across the WaveResNet grids, higher LR consistently improved results within the tested range.

In the LR-only “creep” sweep (holding sine params fixed at the best known values):

- mean `val_ppl` improved monotonically from `lr=0.0022` → `0.0026`
- best mean across seeds was at `lr=0.0026` (but with higher seed variance than the mid-range LRs)

Source: `reports/benchmarks/20251229_140832_waveresnet_lr_creep_v1_small/summary.csv`

### 3) Sine frequency init and per-feature Gaussian noise matter (but less than LR)

From the WaveResNet sine-parameter sweep:

- **Best `freq_init` range**: ~`2.0–2.25` (worse at `1.75`).
- **Best `freq_init_std`**: `0.25` (better than `0` and slightly better/more stable than `0.5`).

Source: `reports/benchmarks/20251228_182315_waveresnet_freq_gauss_tune_v2_small/summary.csv`

These findings are specific to the current proxy architecture and budget, but they are strong
enough to justify treating them as the default starting point in the next full-scale run.

## Practical Recommendations for the Next Full-Scale Run

### Architecture

- Use **`base=waveresnet`** as the default backbone for the next major training run.
- Keep the comparison discipline: match tokenizer + eval shards so deltas stay interpretable.

### Sine parameter defaults (starting point)

Use the best proxy settings as the default initialization:

- `mlp_activation: sine`
- `sine_params`:
  - `amp_init: 1.0`
  - `freq_init: 2.25`
  - `freq_init_std: 0.25` (per-feature Gaussian init)
  - `damp_init: 0.001`
  - `trainable: true`

### Learning rate

- For the proxy (~13M) setup, `lr≈0.0024–0.0026` looked best.
- Do **not** assume the exact LR transfers to 300M/1B directly; treat this as a *directional*
  prior and run a short full-scale pilot (e.g., a few thousand steps) to validate stability.

### Evaluation hygiene

- Prefer fixed, reusable eval shards (local JSONL) over streaming for comparability and to avoid
  network hiccups impacting training or sidecar evaluation.
- Track both per-step metrics (loss/ppl) and a smoothed signal (e.g., rolling avg) so regressions
  show up quickly.

## Performance Notes (Blackwell / sin/exp)

Even on Blackwell-class GPUs, `sin`/`exp` remain “special function” ops. In practice, overall speed
is often dominated by kernel launch overhead + memory bandwidth, not raw SFU throughput.

High-impact next steps if we want to optimize the math path:

- Try **`torch.compile`** on the model to encourage fusion of the elementwise activation chain.
- Profile (`torch.profiler` or Nsight Systems) to confirm whether `sin/exp` kernels dominate.
- If needed, add an optional “fast approximation” path (e.g., a Triton polynomial approx) behind a
  flag for experiments where a small accuracy trade-off is acceptable.

## Reproducing the Key Sweeps

Commands:

- `python3 scripts/bench_lm_bases.py --config examples/lm/configs/waveresnet_freq_gauss_tune_v2_small.yaml --skip-existing`
- `python3 scripts/bench_lm_bases.py --config examples/lm/configs/waveresnet_lr_creep_v1_small.yaml --skip-existing`
- `python3 scripts/bench_lm_bases.py --config examples/lm/configs/transformer_lr0.0022_vs_waveresnet_v1_small.yaml --skip-existing`

Each run writes `summary.csv` and `leaderboard.md` under `reports/benchmarks/<timestamp>_<run_name>/`.

