#!/usr/bin/env bash
set -euo pipefail

# GPU detection
HAVE_NVIDIA=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  HAVE_NVIDIA=1
fi

HAVE_XPU=0
if python -c "import torch; import intel_extension_for_pytorch; assert torch.xpu.is_available()" >/dev/null 2>&1; then
  HAVE_XPU=1
fi

if [ "${HAVE_NVIDIA}" -eq 1 ]; then
  NGPUS=$(nvidia-smi -L | wc -l)
  NGPUS=$(echo "${NGPUS}" | xargs) # trim
elif [ "${HAVE_XPU}" -eq 1 ]; then
  NGPUS=$(python -c "import torch; import intel_extension_for_pytorch; print(torch.xpu.device_count())")
else
  NGPUS=1
fi
echo "Detected ${NGPUS} GPUs."

# match your previous run: you trained with device_batch_size=1
DEVICE_BATCH=1

# 1) EVAL (mid)
#echo "==> eval (mid)"
#torchrun --standalone --nproc_per_node="${NGPUS}" -m scripts.chat_eval -- -i mid

# 2) SFT
echo "==> sft (same device_batch_size)"
torchrun --standalone --nproc_per_node="${NGPUS}" -m scripts.chat_sft -- --device_batch_size=${DEVICE_BATCH}

# 3) EVAL (sft)
#echo "==> eval (sft)"
#torchrun --standalone --nproc_per_node="${NGPUS}" -m scripts.chat_eval -- -i sft

# 4) report
echo "==> report"
python -m nanochat.report generate

echo "DONE."
