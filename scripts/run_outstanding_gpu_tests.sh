#!/usr/bin/env bash
set -euo pipefail

# Run the remaining GPU validation tasks (full sweep + throughput grid) in one shot.
# Creates a tagged benchmark directory with logs so artifacts can be pulled back easily.
#
# Usage:
#   chmod +x scripts/run_outstanding_gpu_tests.sh
#   ./scripts/run_outstanding_gpu_tests.sh
#
# Environment variables:
#   OUT_DIR      Directory for GPU validation reports (default: reports/gpu)
#   BENCH_OUT    Base directory for benchmark artifacts/logs (default: reports/benchmarks)
#   TAG          Optional suffix for the benchmark folder (default: <UTC timestamp>_gpu_todo)

OUT_DIR=${OUT_DIR:-reports/gpu}
BENCH_OUT=${BENCH_OUT:-reports/benchmarks}
TAG=${TAG:-$(date -u +%Y%m%d_%H%M%S)_gpu_todo}

RUN_DIR="${BENCH_OUT}/${TAG}"
mkdir -p "${RUN_DIR}"
LOG_FILE="${RUN_DIR}/run_outstanding_gpu_tests.log"

echo "[gpu-todo] Writing aggregate logs to ${LOG_FILE}"
{
  echo "[gpu-todo] Started at $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "[gpu-todo] Using OUT_DIR=${OUT_DIR}"

  echo "[gpu-todo] Step 1/3: Full GPU validation suite"
  python scripts/run_gpu_validation.py --out "${OUT_DIR}"

  echo "[gpu-todo] Step 2/3: GPU-03 throughput sweeps (131k & 262k batch tokens)"
  for BTOK in 131072 262144; do
    echo "[gpu-todo]   -> GPU-03 with PSANN_GPU03_BATCH_TOKENS=${BTOK}"
    PSANN_GPU03_BATCH_TOKENS=${BTOK} \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      python scripts/run_gpu_validation.py --out "${OUT_DIR}" --only GPU-03
  done
  unset PSANN_GPU03_BATCH_TOKENS

  echo "[gpu-todo] Step 3/3: Aggregate GPU reports into ${RUN_DIR}"
  python scripts/aggregate_benchmarks.py --gpu-reports "${OUT_DIR}" --out "${RUN_DIR}"

  echo "[gpu-todo] Completed at $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "[gpu-todo] Artifacts ready under ${RUN_DIR}"
} | tee "${LOG_FILE}"

echo "[gpu-todo] Done. Pull ${RUN_DIR} (throughput.csv, memory.json, log) plus newest ${OUT_DIR}/<timestamp>/summary.json back to this machine."
