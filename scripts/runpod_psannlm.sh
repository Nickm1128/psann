#!/usr/bin/env bash
# One-stop RunPod helper for PSANN-LM: setup, train, or generate
# Usage:
#   bash scripts/runpod_psannlm.sh setup
#   bash scripts/runpod_psannlm.sh train [extra train args]
#   bash scripts/runpod_psannlm.sh generate --ckpt <ckpt.pt> --tok <tok_dir> [--prompt "..."]

set -euo pipefail

CMD="${1:-}"; shift || true

case "$CMD" in
  setup)
    echo "[setup] creating venv .venv"; python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    echo "[setup] installing torch/cu121 (nightly)"
    python -m pip install --index-url https://download.pytorch.org/whl/nightly/cu121 \
      torch torchvision torchaudio
    echo "[setup] installing deps"
    python -m pip install datasets transformers pynvml matplotlib hf_transfer || true
    echo "[setup] installing repo"
    python -m pip install -e .[lm,viz]
    echo "[setup] done";;

  train)
    source .venv/bin/activate
    echo "[train] launching train_psannlm_chat.py"
    python scripts/train_psannlm_chat.py "$@";;

  generate)
    source .venv/bin/activate
    echo "[gen] launching gen_psannlm_chat.py"
    python scripts/gen_psannlm_chat.py "$@";;

  *)
    echo "Usage: bash scripts/runpod_psannlm.sh {setup|train|generate} [args...]"; exit 1;;
 esac
