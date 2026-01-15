#!/usr/bin/env bash
set -euo pipefail

# Run only the BMRK-01 tiny-corpus benchmark on GPU and emit all artifacts.
#
# Usage:
#   chmod +x scripts/run_bmrk01.sh
#   ./scripts/run_bmrk01.sh
#
# Environment variables:
#   TINY_CONFIG  YAML config (default: examples/lm/configs/tiny_corpus_benchmark.yaml)
#   BENCH_OUT    Base directory for benchmark artifacts (default: reports/benchmarks)
#   TINY_DATA    Path to corpus file (default in config: datasets/lm/tiny_books.txt)

TINY_CONFIG=${TINY_CONFIG:-examples/lm/configs/tiny_corpus_benchmark.yaml}
BENCH_OUT=${BENCH_OUT:-reports/benchmarks}

TS=$(date -u +%Y%m%d_%H%M%S)
BENCH_DIR="${BENCH_OUT}/${TS}"
mkdir -p "${BENCH_DIR}"

# Ensure tiny corpus exists; synthesize ~50MB if missing
TINY_DATA=$(python - <<'PY'
import os
print(os.environ.get('TINY_DATA','datasets/lm/tiny_books.txt'))
PY
)
if [ ! -f "${TINY_DATA}" ]; then
  echo "[BMRK-01] Creating synthetic tiny corpus at ${TINY_DATA} (~50MB)"
  python scripts/make_tiny_corpus.py --out "${TINY_DATA}" --mb 50
fi

echo "[BMRK-01] Training via ${TINY_CONFIG} -> ${BENCH_DIR}"
export PSANN_OUTPUT_DIR="${BENCH_DIR}"
python -m psannlm.lm.train.cli --config "${TINY_CONFIG}" 2>&1 | tee "${BENCH_DIR}/tiny_benchmark.log"

echo "[BMRK-01] Parsing trainer log"
python scripts/parse_trainer_log.py --log "${BENCH_DIR}/tiny_benchmark.log" --out "${BENCH_DIR}" --plot || true

echo "[BMRK-01] Finalizing metrics.json (includes validation perplexity)"
python scripts/finalize_bmrk01.py \
  --config "${TINY_CONFIG}" \
  --bench-dir "${BENCH_DIR}" \
  --log "${BENCH_DIR}/tiny_benchmark.log" --plot || true

echo "[BMRK-01] Done -> ${BENCH_DIR}"
