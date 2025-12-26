#!/usr/bin/env bash
set -euo pipefail

# Runs a small batch of chat-style prompts through an SFT trainer checkpoint.
#
# Defaults assume you ran the OASST1 SFT example and saved checkpoints into:
#   runs/lm/300m_en_sft_300m_sft_oasst1/
#
# Usage:
#   bash scripts/run_sft_generations.sh
#
# Override defaults:
#   CKPT=... TOK_DIR=... PROMPTS_FILE=... TEMPERATURE=... TOP_P=... MAX_NEW_TOKENS=... \
#   bash scripts/run_sft_generations.sh
#
# Run *all* checkpoints in a directory (default):
#   CKPT_DIR=runs/lm/300m_en_sft_300m_sft_oasst1 bash scripts/run_sft_generations.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CKPT=${CKPT:-}
CKPT_DIR=${CKPT_DIR:-runs/lm/300m_en_sft_300m_sft_oasst1}
TOK_DIR=${TOK_DIR:-runs/tokenizer_300m_shuffle_v4}
PROMPTS_FILE=${PROMPTS_FILE:-scripts/prompts_sft_oasst1.txt}

PYTHON_BIN=${PYTHON_BIN:-python3}
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-0.9}
TOP_K=${TOP_K:-0}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
REPETITION_PENALTY=${REPETITION_PENALTY:-1.1}
NO_REPEAT_NGRAM_SIZE=${NO_REPEAT_NGRAM_SIZE:-3}
SEED=${SEED:-1337}

TOP_K_FLAG=""
if [ "${TOP_K}" -gt 0 ]; then
  TOP_K_FLAG="--top-k ${TOP_K}"
fi

CKPTS=()
if [ -n "$CKPT" ]; then
  CKPTS+=("$CKPT")
else
  mapfile -t CKPTS < <(find "$CKPT_DIR" -maxdepth 1 -type f -name "*.pt" -print | sort -V)
fi
if [ "${#CKPTS[@]}" -eq 0 ]; then
  echo "[error] No checkpoints found (CKPT='$CKPT', CKPT_DIR='$CKPT_DIR')." >&2
  exit 1
fi

for ckpt_path in "${CKPTS[@]}"; do
  echo "================================================================"
  echo "[ckpt] $ckpt_path"
  echo "================================================================"

  PYTHONPATH=src "$PYTHON_BIN" scripts/generate_from_trainer_ckpt.py \
    --ckpt "$ckpt_path" \
    --tokenizer-dir "$TOK_DIR" \
    --add-bos \
    --prompts-file "$PROMPTS_FILE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --repetition-penalty "$REPETITION_PENALTY" \
    --no-repeat-ngram-size "$NO_REPEAT_NGRAM_SIZE" \
    $TOP_K_FLAG \
    --seed "$SEED"
done
