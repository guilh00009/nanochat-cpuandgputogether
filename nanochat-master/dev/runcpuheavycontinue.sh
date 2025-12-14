#!/usr/bin/env bash
set -euo pipefail

# GPU detection
HAVE_NVIDIA=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  HAVE_NVIDIA=1
fi

if [ "${HAVE_NVIDIA}" -eq 1 ]; then
  NGPUS=$(nvidia-smi -L | wc -l)
  NGPUS=$(echo "${NGPUS}" | xargs) # trim
else
  NGPUS=1
fi
echo "Detected ${NGPUS} GPUs."

# match your previous run: you trained with device_batch_size=1
DEVICE_BATCH=1

# 1) MID-TRAIN
echo "==> mid-train (repo-native args only)"
torchrun --standalone --nproc_per_node="${NGPUS}" -m scripts.mid_train -- --device_batch_size=${DEVICE_BATCH}

# 2) EVAL (mid)
echo "==> eval (mid)"
torchrun --standalone --nproc_per_node="${NGPUS}" -m scripts.chat_eval -- -i mid

# 3) SFT
echo "==> sft (same device_batch_size)"
torchrun --standalone --nproc_per_node="${NGPUS}" -m scripts.chat_sft -- --device_batch_size=${DEVICE_BATCH}

# 4) EVAL (sft)
echo "==> eval (sft)"
torchrun --standalone --nproc_per_node="${NGPUS}" -m scripts.chat_eval -- -i sft

# 5) report
echo "==> report"
python -m nanochat.report generate

echo "DONE."
