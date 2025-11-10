#!/usr/bin/env bash
# One-shot RunPod training script for ~1.3B PSANN-LM

set -euo pipefail

REPO=${REPO:-/workspace/psann}
cd "$REPO"

LOG_DIR=${LOG_DIR:-$REPO/logs}
mkdir -p "$LOG_DIR" artifacts runs/tokenizer_1b
RUN_NAME=${RUN_NAME:-psannlm_1b_$(date +%Y%m%d_%H%M%S)}
LOG_FILE="$LOG_DIR/${RUN_NAME}.log"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export NCCL_ASYNC_ERROR_HANDLING=1

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[lm]
pip install hf_transfer langdetect datasets tokenizers bitsandbytes accelerate

NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L | wc -l)}
BATCH_TOKENS=${BATCH_TOKENS:-32768}

CMD="torchrun --nproc_per_node=${NUM_GPUS} scripts/train_psann_lm.py \
  --hf-dataset allenai/c4 --hf-name en --hf-text-key text \
  --hf-keep-ascii-only --hf-lang en --hf-lang-threshold 0.85 \
  --tokenizer-backend tokenizers --train-tokenizer \
  --tokenizer-save-dir runs/tokenizer_1b --tokenizer-sample-limit 200000 \
  --base waveresnet --d-model 2048 --n-layers 22 --n-heads 16 --d-mlp 8192 \
  --batch-tokens ${BATCH_TOKENS} --grad-accum-steps 8 \
  --lr 2.5e-4 --weight-decay 0.01 \
  --amp bf16 --fsdp full_shard --fsdp-auto-wrap size \
  --grad-checkpoint --steps-per-epoch 2000 --epochs 120 \
  --checkpoint-dir runs/lm/1b_en \
  --export-dir artifacts/psannlm_1b_bundle"

echo "[$(date -Iseconds)] Starting 1B run" | tee "$LOG_FILE"
eval $CMD 2>&1 | tee -a "$LOG_FILE"
echo "[$(date -Iseconds)] Training complete" | tee -a "$LOG_FILE"
