#!/usr/bin/env bash

# Run the full CUDA test battery (full pytest + GPU-only markers) with a single command.
#
# Usage:
#   ./scripts/run_cuda_suite.sh
#   PYTHON=python3 OUT_BASE=reports/tests ./scripts/run_cuda_suite.sh -- -k lm
#
# Environment variables:
#   PYTHON    - Python interpreter to use (default: python)
#   OUT_BASE  - Base directory for run_cuda_tests.py artifacts (default: reports/tests)
#   GPU_OUT   - Directory for run_gpu_tests.py artifacts (default: reports/tests/gpu_smoke)
#
# Any arguments after optional env vars are forwarded to run_cuda_tests.py

set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
OUT_BASE="${OUT_BASE:-reports/tests}"
GPU_OUT="${GPU_OUT:-reports/tests/gpu_smoke}"
GPU_VAL_OUT="${GPU_VAL_OUT:-reports/gpu}"

echo "[cuda-suite] Using Python: ${PYTHON_BIN}"
echo "[cuda-suite] Artifact roots: ${OUT_BASE} (full suite), ${GPU_OUT} (gpu markers)"

"${PYTHON_BIN}" - <<'PY'
import sys

try:
    import torch  # type: ignore
except ImportError as exc:
    sys.exit(f"[cuda-suite] PyTorch import failed: {exc}")

if not torch.cuda.is_available():
    sys.exit("[cuda-suite] CUDA is not available. Aborting.")

print(f"[cuda-suite] Detected {torch.cuda.device_count()} CUDA device(s).")
PY

mkdir -p "${OUT_BASE}"
mkdir -p "${GPU_OUT}"

echo "[cuda-suite] Running full pytest suite via scripts/run_cuda_tests.py"
"${PYTHON_BIN}" scripts/run_cuda_tests.py --out "${OUT_BASE}" "$@"

echo "[cuda-suite] Running GPU-focused markers via scripts/run_gpu_tests.py"
"${PYTHON_BIN}" scripts/run_gpu_tests.py --outdir "${GPU_OUT}"

echo "[cuda-suite] Running full GPU validation block via scripts/run_gpu_validation.py"
"${PYTHON_BIN}" scripts/run_gpu_validation.py --out "${GPU_VAL_OUT}"

echo "[cuda-suite] Done. Artifacts stored under:"
echo "  - ${OUT_BASE}"
echo "  - ${GPU_OUT}"
echo "  - ${GPU_VAL_OUT}"
