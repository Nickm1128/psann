#!/usr/bin/env bash
# Quick smoke test for the PSANN-LM RunPod pipeline.

set -euo pipefail

REPO=${REPO:-/workspace/psann}
cd "$REPO"

LOG_DIR=${LOG_DIR:-$REPO/logs}
mkdir -p "$LOG_DIR" artifacts runs/tokenizer_smoke
RUN_NAME=${RUN_NAME:-psannlm_smoke_$(date +%Y%m%d_%H%M%S)}
LOG_FILE="$LOG_DIR/${RUN_NAME}.log"

export TORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export NCCL_ASYNC_ERROR_HANDLING=1

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[lm]
pip install hf_transfer langdetect datasets tokenizers bitsandbytes accelerate

NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L | wc -l)}
BATCH_TOKENS=${BATCH_TOKENS:-4096}

CMD="torchrun --nproc_per_node=${NUM_GPUS} scripts/train_psann_lm.py \
  --hf-dataset allenai/c4 --hf-name en --hf-text-key text \
  --hf-keep-ascii-only --hf-lang en \
  --tokenizer-backend tokenizers --train-tokenizer \
  --tokenizer-save-dir runs/tokenizer_smoke --tokenizer-sample-limit 5000 \
  --base waveresnet --d-model 512 --n-layers 4 --n-heads 4 --d-mlp 2048 \
  --batch-tokens ${BATCH_TOKENS} --grad-accum-steps 1 \
  --lr 5e-4 --weight-decay 0.01 \
  --amp bf16 --fsdp off \
  --grad-checkpoint \
  --steps-per-epoch 25 --epochs 2 \
  --log-interval-steps 5 \
  --checkpoint-dir runs/lm/smoke_en \
  --export-dir artifacts/psannlm_smoke_bundle"

echo "[$(date -Iseconds)] Starting smoke run" | tee "$LOG_FILE"
eval $CMD 2>&1 | tee -a "$LOG_FILE"
echo "[$(date -Iseconds)] Smoke run complete" | tee -a "$LOG_FILE"
