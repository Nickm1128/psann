#!/usr/bin/env bash
# RunPod training script for ~300M PSANN-LM (63B tokens target)

set -euo pipefail

REPO=${REPO:-/workspace/psann}
cd "$REPO"

ATTN_FLAGS="${ATTN_FLAGS:-}"
DATA_FLAGS="${DATA_FLAGS:-}"
AMP_FLAGS="${AMP_FLAGS:-}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"

LOG_DIR=${LOG_DIR:-$REPO/logs}
mkdir -p "$LOG_DIR" artifacts runs runs/tokenizer_300m
RUN_NAME=${RUN_NAME:-psannlm_300m_$(date +%Y%m%d_%H%M%S)}
LOG_FILE="$LOG_DIR/${RUN_NAME}.log"

SEQ_LEN=${SEQ_LEN:-2048}
TOKENS_TARGET_OVERRIDDEN="${TOKENS_TARGET:-}"
TOKENS_TARGET_GB=${TOKENS_TARGET_GB:-50}
if [ -n "$TOKENS_TARGET_OVERRIDDEN" ]; then
  TOKENS_TARGET=$TOKENS_TARGET_OVERRIDDEN
else
  TOKENS_TARGET=$(( TOKENS_TARGET_GB * 1000000000 ))
fi
TOKENS_PER_STEP_HINT=${TOKENS_PER_STEP_HINT:-0}

export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/workspace/.hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/workspace/.hf_cache}"
export TOKENIZERS_PARALLELISM=true
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[lm]
pip install hf_transfer langdetect datasets tokenizers accelerate

NUM_GPUS=${NUM_GPUS:-1}
GRAD_ACCUM=${GRAD_ACCUM:-1}

if [ -z "${BATCH_TOKENS:-}" ]; then
  if [ "$TOKENS_PER_STEP_HINT" -gt 0 ]; then
    divisor=$(( NUM_GPUS * GRAD_ACCUM ))
    if [ "$divisor" -le 0 ]; then
      divisor=1
    fi
    per_device_tokens=$(( TOKENS_PER_STEP_HINT / divisor ))
    if [ "$per_device_tokens" -lt "$SEQ_LEN" ]; then
      per_device_tokens=$SEQ_LEN
    fi
    per_device_tokens=$(( per_device_tokens / SEQ_LEN * SEQ_LEN ))
    if [ "$per_device_tokens" -lt "$SEQ_LEN" ]; then
      per_device_tokens=$SEQ_LEN
    fi
    BATCH_TOKENS=$per_device_tokens
  else
    BATCH_TOKENS=4096
  fi
fi

TOKENS_PER_STEP_ACTUAL=$(( BATCH_TOKENS * GRAD_ACCUM * NUM_GPUS ))
if [ "$TOKENS_PER_STEP_ACTUAL" -le 0 ]; then
  TOKENS_PER_STEP_ACTUAL=$SEQ_LEN
fi
MAX_STEPS=$(( TOKENS_TARGET / TOKENS_PER_STEP_ACTUAL ))
if [ "$MAX_STEPS" -le 0 ]; then
  MAX_STEPS=1
fi
export MAX_STEPS
export TOKENS_PER_STEP_ACTUAL

if [ "$NUM_GPUS" -gt 1 ]; then
  FSDP_FLAGS="--fsdp full_shard --fsdp-auto-wrap size"
else
  FSDP_FLAGS="--fsdp off"
fi

OPT_FLAGS="--optimizer adamw"

if [ "$NUM_GPUS" -gt 1 ]; then
  LAUNCHER="torchrun --nproc_per_node=${NUM_GPUS} -m psannlm.train"
else
  LAUNCHER="python -u -m psannlm.train"
fi

CMD="${LAUNCHER} \
  --hf-dataset allenai/c4 --hf-name en --hf-text-key text \
  --hf-keep-ascii-only --hf-lang en --hf-lang-threshold 0.85 \
  --tokenizer-backend tokenizers --train-tokenizer \
  --tokenizer-save-dir runs/tokenizer_300m --tokenizer-sample-limit 150000 \
  --hf-cache-limit-gb 40 \
  --base waveresnet --d-model 1024 --n-layers 16 --n-heads 16 --d-mlp 4096 \
  --seq_len ${SEQ_LEN} --dataset_streaming true --max_steps ${MAX_STEPS} \
  --tokens_target ${TOKENS_TARGET} \
  --dataloader_num_workers 2 \
  --batch-tokens ${BATCH_TOKENS} --grad-accum-steps ${GRAD_ACCUM} \
  --lr 3e-4 --weight-decay 0.01 \
  --amp bf16 ${FSDP_FLAGS} ${OPT_FLAGS} \
  --grad-checkpoint \
  --log-interval-steps 25 \
  --checkpoint-dir runs/lm/300m_en \
  --export-dir artifacts/psannlm_300m_bundle \
  ${ATTN_FLAGS:+$ATTN_FLAGS} \
  ${DATA_FLAGS:+$DATA_FLAGS} \
  ${AMP_FLAGS:+$AMP_FLAGS} \
  ${EXTRA_FLAGS:+$EXTRA_FLAGS}"

echo "[$(date -Iseconds)] Starting 300M run" | tee "$LOG_FILE"
echo "[config] seq_len=${SEQ_LEN} batch_tokens=${BATCH_TOKENS} grad_accum=${GRAD_ACCUM} num_gpus=${NUM_GPUS} tokens_per_step=${TOKENS_PER_STEP_ACTUAL} tokens_target=${TOKENS_TARGET} max_steps=${MAX_STEPS}" | tee -a "$LOG_FILE"
eval $CMD 2>&1 | tee -a "$LOG_FILE"
echo "[$(date -Iseconds)] Training complete" | tee -a "$LOG_FILE"
