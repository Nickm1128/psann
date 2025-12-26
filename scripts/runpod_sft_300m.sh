#!/usr/bin/env bash
# RunPod/local helper to SFT a pretrained ~300M PSANN-LM checkpoint.
#
# Required env:
#   INIT_CKPT=...               (or set RESUME_CKPT=... to continue an SFT run)
#   TOKENIZER_DIR=...           (directory with tokenizer.json + special_tokens_map.json)
#
# Common knobs:
#   SFT_SOURCE=oasst1|pairs
#   SFT_DATASET=OpenAssistant/oasst1
#   SFT_SPLIT=train
#   SFT_TEMPLATE=chat|alpaca
#   MAX_STEPS=2000             (or TOKENS_TARGET_GB=1 to derive steps)
#
# Example:
#   INIT_CKPT=runs/lm/300m_en/ckpt_step078000.pt \
#   TOKENIZER_DIR=runs/tokenizer_300m_shuffle_v4 \
#   RUN_NAME=300m_sft_oasst1 \
#   MAX_STEPS=2000 \
#   bash scripts/runpod_sft_300m.sh

set -euo pipefail

REPO=${REPO:-/workspace/psann}
cd "$REPO"

PYTHON_BIN=${PYTHON_BIN:-python3}
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[error] Could not find Python interpreter '$PYTHON_BIN'." >&2
  exit 1
fi

LOG_DIR=${LOG_DIR:-$REPO/logs}
mkdir -p "$LOG_DIR" runs
RUN_NAME=${RUN_NAME:-psannlm_300m_sft_$(date +%Y%m%d_%H%M%S)}
LOG_FILE="$LOG_DIR/${RUN_NAME}.log"

SEQ_LEN=${SEQ_LEN:-2048}
AMP_MODE=${AMP_MODE:-bf16}
BATCH_TOKENS=${BATCH_TOKENS:-32768}
GRAD_ACCUM=${GRAD_ACCUM:-4}
LR=${LR:-5e-5}
WARMUP_STEPS=${WARMUP_STEPS:-200}
SAVE_INTERVAL_STEPS=${SAVE_INTERVAL_STEPS:-500}
LOG_INTERVAL_STEPS=${LOG_INTERVAL_STEPS:-25}

SFT_SOURCE=${SFT_SOURCE:-oasst1}
SFT_DATASET=${SFT_DATASET:-OpenAssistant/oasst1}
SFT_NAME=${SFT_NAME:-}
SFT_SPLIT=${SFT_SPLIT:-train}
SFT_TEMPLATE=${SFT_TEMPLATE:-chat}
SFT_MAX_PAIRS=${SFT_MAX_PAIRS:-0}
SFT_PROMPT_KEY=${SFT_PROMPT_KEY:-prompt}
SFT_RESPONSE_KEY=${SFT_RESPONSE_KEY:-response}
SFT_DATA_FILES=${SFT_DATA_FILES:-}
SFT_ASCII_ONLY=${SFT_ASCII_ONLY:-1}
SFT_LANG=${SFT_LANG:-en}
SFT_LANG_THRESHOLD=${SFT_LANG_THRESHOLD:-0.85}

ADD_BOS=${ADD_BOS:-1}
ADD_EOS=${ADD_EOS:-1}
ADD_BOS_FLAG=""
ADD_EOS_FLAG=""
if [ "$ADD_BOS" -eq 1 ]; then ADD_BOS_FLAG="--add-bos"; fi
if [ "$ADD_EOS" -eq 1 ]; then ADD_EOS_FLAG="--add-eos"; fi

ASCII_FLAG=""
if [ "$SFT_ASCII_ONLY" -eq 1 ]; then
  ASCII_FLAG="--ascii-only"
fi
LANG_FLAGS=""
if [ -n "$SFT_LANG" ]; then
  IFS=',' read -ra LANG_ARR <<< "$SFT_LANG"
  for lang in "${LANG_ARR[@]}"; do
    if [ -n "$lang" ]; then
      LANG_FLAGS+=" --lang $lang"
    fi
  done
fi

CHECKPOINT_DIR=${CHECKPOINT_DIR:-runs/lm/300m_en_sft_${RUN_NAME}}

TOKENS_TARGET_GB=${TOKENS_TARGET_GB:-0}
TOKENS_TARGET=${TOKENS_TARGET:-0}
MAX_STEPS=${MAX_STEPS:-0}

if [ "$TOKENS_TARGET" -le 0 ] && [ "$TOKENS_TARGET_GB" -gt 0 ]; then
  TOKENS_TARGET=$(( TOKENS_TARGET_GB * 1000000000 ))
fi

MICRO_BATCH=$(( BATCH_TOKENS / SEQ_LEN ))
if [ "$MICRO_BATCH" -le 0 ]; then
  MICRO_BATCH=1
fi
TOKENS_PER_STEP=$(( MICRO_BATCH * SEQ_LEN * GRAD_ACCUM ))
if [ "$MAX_STEPS" -le 0 ] && [ "$TOKENS_TARGET" -gt 0 ]; then
  MAX_STEPS=$(( TOKENS_TARGET / TOKENS_PER_STEP ))
fi
if [ "$MAX_STEPS" -le 0 ]; then
  echo "[error] Set MAX_STEPS or TOKENS_TARGET(_GB)." >&2
  exit 1
fi

SKIP_VENV=${SKIP_VENV:-0}
VENV_DIR=${VENV_DIR:-.venv_sft}
# Default to system site packages so we reuse the container's CUDA-enabled PyTorch.
# (Without this on aarch64, pip will often install a CPU-only torch wheel.)
VENV_FLAGS=${VENV_FLAGS---system-site-packages}
if [ "$SKIP_VENV" -eq 0 ]; then
  $PYTHON_BIN -m venv ${VENV_FLAGS} "$VENV_DIR"
  # shellcheck source=/dev/null
  source "$VENV_DIR/bin/activate"
else
  echo "[env] SKIP_VENV=1; using existing Python environment."
fi

$PYTHON_BIN -m pip install --upgrade pip
$PYTHON_BIN -m pip install -e .[lm]
$PYTHON_BIN -m pip install datasets tokenizers accelerate hf_transfer langdetect || true
$PYTHON_BIN - <<'PY'
import torch

print(f"[env] torch={torch.__version__} cuda_available={torch.cuda.is_available()}", flush=True)
if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA is not available in this environment. "
        "If you're inside an NVIDIA NGC container, set VENV_FLAGS=--system-site-packages "
        "(or SKIP_VENV=1) so the venv can see the container's CUDA-enabled PyTorch."
    )

props = torch.cuda.get_device_properties(0)
total_gb = props.total_memory / float(1024**3)
print(
    f"[gpu] detected device={props.name} capability={props.major}.{props.minor} "
    f"total_mem_gb={total_gb:.2f}",
    flush=True,
)
PY

INIT_CKPT=${INIT_CKPT:-}
RESUME_CKPT=${RESUME_CKPT:-}
if [ -n "$RESUME_CKPT" ]; then
  CKPT_FLAG="--resume-ckpt $RESUME_CKPT"
elif [ -n "$INIT_CKPT" ]; then
  CKPT_FLAG="--init-ckpt $INIT_CKPT"
else
  echo "[error] Set INIT_CKPT=... or RESUME_CKPT=..." >&2
  exit 1
fi

TOKENIZER_DIR=${TOKENIZER_DIR:-}
if [ -z "$TOKENIZER_DIR" ]; then
  echo "[error] Set TOKENIZER_DIR=... (directory with tokenizer.json)." >&2
  exit 1
fi

NAME_FLAG=""
if [ -n "$SFT_NAME" ]; then
  NAME_FLAG="--name $SFT_NAME"
fi
DATA_FILES_FLAG=""
if [ -n "$SFT_DATA_FILES" ]; then
  DATA_FILES_FLAG="--data-files $SFT_DATA_FILES"
fi

CMD="$PYTHON_BIN -u -m psannlm.sft \
  $CKPT_FLAG \
  --tokenizer-dir $TOKENIZER_DIR \
  --sft-source $SFT_SOURCE \
  --dataset $SFT_DATASET $NAME_FLAG --split $SFT_SPLIT $DATA_FILES_FLAG \
  --prompt-key $SFT_PROMPT_KEY --response-key $SFT_RESPONSE_KEY \
  --max-pairs $SFT_MAX_PAIRS \
  --template $SFT_TEMPLATE $ADD_BOS_FLAG $ADD_EOS_FLAG \
  $ASCII_FLAG $LANG_FLAGS --lang-threshold $SFT_LANG_THRESHOLD \
  --seq-len $SEQ_LEN --batch-tokens $BATCH_TOKENS --grad-accum-steps $GRAD_ACCUM \
  --lr $LR --warmup-steps $WARMUP_STEPS --weight-decay 0.0 \
  --amp $AMP_MODE \
  --log-interval-steps $LOG_INTERVAL_STEPS --save-interval-steps $SAVE_INTERVAL_STEPS \
  --max-steps $MAX_STEPS \
  --checkpoint-dir $CHECKPOINT_DIR"

echo "[$(date -Iseconds)] Starting SFT ($RUN_NAME)" | tee "$LOG_FILE"
echo "[config] seq_len=${SEQ_LEN} batch_tokens=${BATCH_TOKENS} grad_accum=${GRAD_ACCUM} tokens_per_step=${TOKENS_PER_STEP} max_steps=${MAX_STEPS} ckpt_dir=${CHECKPOINT_DIR}" | tee -a "$LOG_FILE"
eval $CMD 2>&1 | tee -a "$LOG_FILE"
echo "[$(date -Iseconds)] SFT complete" | tee -a "$LOG_FILE"
