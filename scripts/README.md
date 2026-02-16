# PSANN Utility Scripts

The utilities under `scripts/` assume the project is installed as a package, so
you no longer need to manually edit `sys.path` when running them.

## Quick Start

1. Create or activate a virtual environment.
2. Install the project in editable mode:
   ```bash
   pip install -e .
   ```
3. Optional: install profiling dependencies as needed (for example,
   `pip install torch torchvision` for GPU runs).

Once the package is available, run scripts directly:

```bash
python scripts/profile_hisso.py --epochs 4
```

Set `CUDA_VISIBLE_DEVICES` or `PYTORCH_CUDA_ALLOC_CONF` if you need to target
specific GPUs; no additional `PYTHONPATH` modifications are required.

## Script Conventions

When adding or updating scripts under `scripts/`:

- Include a top-level module docstring with a 1–2 sentence summary plus at least one example command.
- Use `argparse` or `typer` so `--help` documents the interface.
- Prefer `--out` / `--output-dir` to control where artifacts are written (default under `reports/` or `outputs/`).
- Print a short config header (timestamp, seed, device/dtype, output dir) so logs are self-contained.

## Available Scripts

### HISSO, parity, and diagnostics

- `profile_hisso.py` - quick sanity check for the lightweight HISSO trainer with
  a synthetic MLP.
- `benchmark_hisso_variants.py` - benchmarks residual dense vs. convolutional
  HISSO estimators, reporting wall-clock time and reward trends for each device
  (CPU/GPU if available). Supports `--dataset` (`synthetic` or `portfolio`),
  schedule `--modes` (`compat`, `fast`) with explicit B/U knobs, and `--output`
  to persist JSON summaries for docs/CI:
  ```bash
  python -m scripts.benchmark_hisso_variants --dataset portfolio --epochs 4 --devices cpu --variants dense,conv --modes compat,fast --compat-batch-episodes 1 --compat-updates-per-epoch 32 --fast-batch-episodes 8 --fast-updates-per-epoch 4 --output docs/benchmarks/hisso_variants_portfolio_cpu.json
  ```
- `compare_hisso_benchmarks.py` - compares two benchmark payloads with configurable tolerances (and optional `--modes` filtering such as `compat` or `fast`); used by CI to detect HISSO performance regressions.
- `fetch_benchmark_data.py` - downloads the trimmed AAPL price series (or any
  other ticker/date range) and writes a `date,open,close` CSV for HISSO runs.
- `run_light_probes.py` - executes the lightweight Colab probes locally. Use
  `--results-dir` to redirect metric dumps away from the repository if desired.
- `gpu_env_report.py` - prints a compact summary of GPU / CUDA / driver state for bug reports.

### GPU validation and CUDA test harness

- `run_gpu_validation.py` - unified GPU validation (GPU-01..08). Writes
  timestamped reports under `reports/gpu/`. Accepts `--only GPU-03` etc.
- `run_cuda_tests.py` / `run_gpu_tests.py` - lower-level CUDA and GPU smoke tests; usually called via `run_cuda_suite.sh`.
- `run_cuda_suite.sh` - convenience wrapper that sequentially runs `run_cuda_tests.py`,
  `run_gpu_tests.py`, and `run_gpu_validation.py` so a single command exercises the
  full CUDA + GPU validation battery. Emits artifacts under `reports/tests/`,
  `reports/tests/gpu_smoke/`, and `reports/gpu/`.
- `next_gpu_batch.sh` - one-sweep batch runner for RunPod/local GPUs. Runs the
  full validation, throughput sweeps at three batch token sizes, gradient
  checkpoint/memory step, and a tiny-corpus training. Produces benchmark
  artifacts in `reports/benchmarks/<timestamp>/`.

### Benchmarks, corpora, and log parsing

- `run_full_suite.py` - one-command runner for light probes, synthetic ablations, and GeoSparse
  benchmarks. Writes to `reports/full_suite/<timestamp>/` and can git-commit results:
  ```bash
  python scripts/run_full_suite.py --device cuda --git-commit
  ```
- `postprocess_full_suite.py` - generates compact CSV tables and plots for a full-suite run:
  ```bash
  python scripts/postprocess_full_suite.py --run reports/full_suite/<timestamp>
  ```
- `aggregate_benchmarks.py` - aggregates GPU validation outputs into
  `throughput.csv` and `memory.json` under a benchmark directory.
- `microbench_psann.py` - microbenchmarks PSANN vs dense/transformer baselines for
  throughput and memory, writing a JSON summary suitable for regression checks.
- `benchmark_geo_sparse_micro.py` - GeoSparse forward/backward microbench with
  optional `--compute-mode auto` to compare gather vs scatter paths.
- `run_geosparse_vs_relu_benchmarks.py` - GeoSparse activation-variant vs
  dense ReLU regression suite (synthetic + real sklearn datasets) with
  parameter matching and wall-clock timing; supports `--geo-activations`
  (default includes `psann,relu_sigmoid_psann`) and writes `results.jsonl`
  plus aggregated summaries under `reports/`.
- `benchmark_regressor_ablations.py` - runs small ablations of ResPSANN,
  WaveResNet, and SGR-PSANN regressors across diverse synthetic datasets, writing
  JSONL/CSV summaries under `reports/ablations/`.
- `finalize_bmrk01.py` - builds `metrics.json` for the tiny-corpus benchmark by
  parsing training metrics and computing validation loss/perplexity.
- `parse_trainer_log.py` - parses trainer stdout to `metrics.csv` and, if
  `matplotlib` is available, `loss_curve.png`.
- `plot_loss_from_csv.py` - quick helper to visualise loss curves from CSV logs.
- `make_tiny_corpus.py` - synthesizes a ~50MB `datasets/lm/tiny_books.txt` if a
  real corpus is not available in the pod.
- `run_bmrk01.sh` - one-shot runner for the BMRK-01 tiny-corpus benchmark. Emits
  `metrics.csv`, `metrics.json`, and optionally `loss_curve.png` into
  `reports/benchmarks/<timestamp>/`.

### Profiling helpers

- `profile_psann.py` - torch.profiler wrapper for PSANN/dense/transformer/GeoSparse
  models; exports a chrome trace plus a short summary table.

### Language modeling and PSANN-LM tooling

- `train_psann_lm.py` - shim around `psannlm.train` for one-command LM training (see `docs/lm.md`).
- `train_psannlm_chat.py` / `gen_psannlm_chat.py` - helpers for chat-style PSANN-LM runs and interactive generation.
- `run_lm_eval_psann.py` - ties PSANN-LM checkpoints into `lm-eval-harness` via a small adapter.
- `bench_lm_bases.py` - quick WikiText-103 base-estimator shootout with loss/perplexity/throughput summaries.
- `benchmark_kv_cache.py` / `compare_tokenizers.py` / `count_psannlm_params.py` / `ppl_wikitext_psann.py` - focused LM utilities for KV-cache, tokenizer benchmarks, parameter counting, and perplexity; mainly used in docs and internal benchmarks.
- `runpod_psannlm.sh`, `runpod_smoke_train.sh`, `runpod_train_1b.sh`, `runpod_train_300m.sh` - RunPod / multi-GPU orchestration scripts for LM training at different scales.
- `runpod_sft_300m.sh` - RunPod/local wrapper to supervised fine-tune (SFT) a pretrained 300M checkpoint on prompt/response pairs.

### Release tooling

- `release.py` - small helper used when cutting PyPI releases and tagging versions.

## Language Modeling

- Train (streaming, tokenizer-in-loop, FSDP-ready):
  - `python scripts/train_psann_lm.py --hf-dataset allenai/c4 --hf-name en --hf-split train --hf-text-key text --hf-keep-ascii-only --hf-lang en --base waveresnet --d-model 3072 --n-layers 30 --n-heads 24 --tokenizer-backend tokenizers --train-tokenizer --tokenizer-save-dir runs/tokenizer_3b --batch-tokens 65536 --grad-accum-steps 8 --amp bf16 --grad-checkpoint --fsdp full_shard --checkpoint-dir runs/lm/3b_en --export-dir artifacts/psannlm_3b_bundle`

- SFT (instruction tuning, prompt masked; example uses OpenAssistant/oasst1):
  - `PYTHONPATH=src python3 -m psannlm.sft --init-ckpt runs/lm/300m_en/ckpt_step078000.pt --tokenizer-dir runs/tokenizer_300m_shuffle_v4 --sft-source oasst1 --checkpoint-dir runs/lm/300m_en_sft_oasst1 --seq-len 2048 --batch-tokens 65536 --grad-accum-steps 2 --lr 5e-5 --warmup-steps 200 --max-steps 2000 --add-bos --add-eos --ascii-only --lang en --lang-threshold 0.85`

- Evaluate with lm-eval (chat template on MC):
  - `python scripts/run_lm_eval_psann.py --hf-repo <user>/<repo> --hf-filename psannlm_chat_final.pt --tokenizer-backend tokenizers --hf-tokenizer-repo <user>/<repo> --hf-tokenizer-filename tokenizer_final/tokenizer.json --tasks hellaswag,piqa,winogrande --device cuda --num-fewshot 5 --apply-chat-template --fewshot-as-multiturn --output eval_out/mc_chat.json`

- Data prep utilities:
  - Deduplicate shards: `python tools/dedupe.py --input shards.txt --output shards_unique.txt`
  - Heuristic decontamination: `python tools/decontaminate.py --input shards_unique.txt --refs wt2.txt lambada.txt --output shards_clean.txt`
  - Build a manifest from directories/globs: `python tools/build_manifest.py --roots /data/en --pattern "*.txt" --recurse --absolute --output /data/en_manifest.txt`
  - Use `--export-dir` on the training script to gather `model.pt`, tokenizer files, and metadata in one folder ready for `huggingface-cli upload`.

See also: `docs/lm_3b_quickstart.md` for a focused 3B quickstart.

## Current Limitations

- GPU runs depend on local PyTorch CUDA support; the benchmarking script
  auto-skips unavailable devices.
- The tiny-corpus benchmark uses a synthetic corpus by default if
  `datasets/lm/tiny_books.txt` is missing. Replace with a public-domain shard
  for meaningful perplexity numbers.
- The portfolio dataset ships a trimmed AAPL open/close series at
  `benchmarks/hisso_portfolio_prices.csv`; point `--dataset-path` at a custom CSV
  (columns: `open,close,...`) to benchmark other assets.

## TODO

- Expand the benchmarking harness to support custom datasets and export
  structured reports.
