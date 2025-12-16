#!/usr/bin/env bash
set -euo pipefail

# GPU detection
HAVE_NVIDIA=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  HAVE_NVIDIA=1
fi

HAVE_XPU=0
if python -c "import torch; try: assert torch.xpu.is_available(); except: exit(1)" >/dev/null 2>&1; then
  HAVE_XPU=1
fi

if [ "${HAVE_NVIDIA}" -eq 1 ]; then
  NGPUS=$(nvidia-smi -L | wc -l)
  NGPUS=$(echo "${NGPUS}" | xargs) # trim
elif [ "${HAVE_XPU}" -eq 1 ]; then
  NGPUS=$(python -c "import torch; print(torch.xpu.device_count())")
else
  NGPUS=1
fi
echo "Detected ${NGPUS} GPUs."

# match your previous run: you trained with device_batch_size=1
# Try to detect from latest base checkpoint meta
DEVICE_BATCH=1

NANOCHAT_BASE_DIR="${HOME}/.cache/nanochat"
if [ -d "${NANOCHAT_BASE_DIR}/base_checkpoints" ]; then
    DETECTED_DBSZ=$(python -c "
import os
import glob
import json
try:
    base = '${NANOCHAT_BASE_DIR}/base_checkpoints'
    metas = glob.glob(os.path.join(base, 'd*', 'meta_*.json'))
    if metas:
        def get_step(p):
            try: return int(os.path.basename(p).split('_')[1].split('.')[0])
            except: return -1
        metas.sort(key=get_step, reverse=True)
        latest = metas[0]
        with open(latest) as f:
            meta = json.load(f)
        b = meta.get('user_config', {}).get('device_batch_size')
        if b is not None:
             print(b)
except:
    pass
")
    if [ -n "${DETECTED_DBSZ}" ]; then
        DEVICE_BATCH="${DETECTED_DBSZ}"
        echo "Auto-detected DEVICE_BATCH=${DEVICE_BATCH} from previous checkpoint."
    fi
fi


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
