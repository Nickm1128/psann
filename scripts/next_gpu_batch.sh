#!/usr/bin/env bash
set -euo pipefail

# Helper script to queue the standard GPU validation batch plus benchmark sweeps.
# Run this from the repo root on a machine with CUDA-capable GPUs.
#
# Usage:
#   chmod +x scripts/next_gpu_batch.sh
#   ./scripts/next_gpu_batch.sh
#
# Environment variables:
#   OUT_DIR         Where GPU validation reports will be stored (default: reports/gpu)
#   BENCH_OUT       Base directory for benchmark artifacts (default: reports/benchmarks)
#   TINY_CONFIG     YAML config for the tiny-corpus run (default: examples/lm/configs/tiny_corpus_benchmark.yaml)

OUT_DIR=${OUT_DIR:-reports/gpu}
BENCH_OUT=${BENCH_OUT:-reports/benchmarks}
TINY_CONFIG=${TINY_CONFIG:-examples/lm/configs/tiny_corpus_benchmark.yaml}

echo "[1/4] Full GPU validation suite -> ${OUT_DIR}"
python scripts/run_gpu_validation.py --out "${OUT_DIR}"

echo "[2/4] Throughput sweep only (GPU-03)"
python scripts/run_gpu_validation.py --out "${OUT_DIR}" --only GPU-03

echo "[3/4] Checkpointing/memory step (GPU-04)"
python scripts/run_gpu_validation.py --out "${OUT_DIR}" --only GPU-04

echo "[4/4] Tiny-corpus benchmark run -> ${BENCH_OUT}"
export PSANN_OUTPUT_DIR="${BENCH_OUT}"
python -m psann.lm.train.cli --config "${TINY_CONFIG}"

echo "[DONE] Commands queued; review reports under '${OUT_DIR}' and benchmark outputs under '${BENCH_OUT}'."
