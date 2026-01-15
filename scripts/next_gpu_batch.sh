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
# Run GPU-03 three times with different effective batch_tokens (B*T)
for BTOK in 65536 131072 262144; do
  echo "  - GPU-03 with target batch_tokens=${BTOK}"
  PSANN_GPU03_BATCH_TOKENS=${BTOK} python scripts/run_gpu_validation.py --out "${OUT_DIR}" --only GPU-03
done
unset PSANN_GPU03_BATCH_TOKENS PSANN_GPU03_B PSANN_GPU03_T

echo "[3/4] Checkpointing/memory step (GPU-04)"
python scripts/run_gpu_validation.py --out "${OUT_DIR}" --only GPU-04

echo "[4/4] Tiny-corpus benchmark run -> ${BENCH_OUT}"
# Prepare benchmark directory with timestamp
TS=$(date -u +%Y%m%d_%H%M%S)
BENCH_DIR="${BENCH_OUT}/${TS}"
mkdir -p "${BENCH_DIR}"

# Ensure tiny corpus exists; if not, synthesize ~50MB locally
TINY_DATA=$(python - <<'PY'
import os
print(os.environ.get('TINY_DATA','datasets/lm/tiny_books.txt'))
PY
)
if [ ! -f "${TINY_DATA}" ]; then
  echo "  - Creating synthetic tiny corpus at ${TINY_DATA} (~50MB)"
  python scripts/make_tiny_corpus.py --out "${TINY_DATA}" --mb 50
fi

# Run the training CLI and tee logs for parsing
export PSANN_OUTPUT_DIR="${BENCH_DIR}"
python -m psannlm.lm.train.cli --config "${TINY_CONFIG}" 2>&1 | tee "${BENCH_DIR}/tiny_benchmark.log"

# Aggregate GPU reports into throughput/memory artifacts under the same benchmark dir
python scripts/aggregate_benchmarks.py --gpu-reports "${OUT_DIR}" --out "${BENCH_DIR}"

# Parse training log to metrics (CSV + optional plot if matplotlib available)
python scripts/parse_trainer_log.py --log "${BENCH_DIR}/tiny_benchmark.log" --out "${BENCH_DIR}" --plot || true

# Finalize BMRK-01 metrics.json (includes validation loss/perplexity)
python scripts/finalize_bmrk01.py \
  --config "${TINY_CONFIG}" \
  --bench-dir "${BENCH_DIR}" \
  --log "${BENCH_DIR}/tiny_benchmark.log" || true

echo "[DONE] Commands queued; review reports under '${OUT_DIR}' and benchmark outputs under '${BENCH_OUT}'."
