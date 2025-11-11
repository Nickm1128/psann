#!/usr/bin/env bash
# RunPod training script for ~300M PSANN-LM (63B tokens target)

set -euo pipefail

REPO=${REPO:-/workspace/psann}
cd "$REPO"

LOG_DIR=${LOG_DIR:-$REPO/logs}
mkdir -p "$LOG_DIR" artifacts runs/tokenizer_300m
RUN_NAME=${RUN_NAME:-psannlm_300m_$(date +%Y%m%d_%H%M%S)}
LOG_FILE="$LOG_DIR/${RUN_NAME}.log"

export TORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=true
# Prefer the new TORCH_NCCL_ASYNC_ERROR_HANDLING; keep the old var unset
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
unset NCCL_ASYNC_ERROR_HANDLING 2>/dev/null || true

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[lm]
pip install hf_transfer langdetect datasets tokenizers bitsandbytes accelerate

NUM_GPUS=${NUM_GPUS:-1}

# reduce per-step peak; keep tokens/step with accumulation
BATCH_TOKENS=${BATCH_TOKENS:-4096}
GRAD_ACCUM=${GRAD_ACCUM:-4}

# pick FSDP even for 1 GPU if you hit OOM
if [ "$NUM_GPUS" -gt 1 ]; then
  FSDP_FLAGS="--fsdp full_shard --fsdp-auto-wrap size"
else
  # enable CPU offload to protect HBM on single GPU
  FSDP_FLAGS="--fsdp full_shard --fsdp-auto-wrap size --fsdp-param-offload cpu --fsdp-grad-offload cpu"
fi

# prefer SDPA/Flash attention if your trainer supports it
ATTN_FLAGS="--attn-backend sdpa"   # or: --attn-backend flash2 / xformers

# use 8-bit AdamW (you already install bitsandbytes)
OPT_FLAGS="--optim adamw_8bit"     # rename to your trainer’s flag if different

CMD="torchrun --nproc_per_node=${NUM_GPUS} scripts/train_psann_lm.py \
  --hf-dataset allenai/c4 --hf-name en --hf-text-key text \
  --hf-keep-ascii-only --hf-lang en --hf-lang-threshold 0.85 \
  --tokenizer-backend tokenizers --train-tokenizer \
  --tokenizer-save-dir runs/tokenizer_300m --tokenizer-sample-limit 150000 \
  --base waveresnet --d-model 1536 --n-layers 18 --n-heads 12 --d-mlp 6144 \
  --batch-tokens ${BATCH_TOKENS} --grad-accum-steps ${GRAD_ACCUM} \
  --lr 3e-4 --weight-decay 0.01 \
  --amp bf16 ${FSDP_FLAGS} ${ATTN_FLAGS} ${OPT_FLAGS} \
  --grad-checkpoint --steps-per-epoch 2000 --epochs 120 \
  --log-interval-steps 25 \
  --checkpoint-dir runs/lm/300m_en \
  --export-dir artifacts/psannlm_300m_bundle"

echo "[$(date -Iseconds)] Starting 300M run" | tee "$LOG_FILE"
eval $CMD 2>&1 | tee -a "$LOG_FILE"
echo "[$(date -Iseconds)] Training complete" | tee -a "$LOG_FILE"
