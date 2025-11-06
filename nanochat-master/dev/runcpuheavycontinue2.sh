#!/usr/bin/env bash
set -euo pipefail

# match your previous run: you trained with device_batch_size=1
DEVICE_BATCH=1

USE_MULTI_GPU=${USE_MULTI_GPU:-1}
NANOCHAT_NPROC=${NANOCHAT_NPROC:-}

detect_nproc() {
  local requested="${NANOCHAT_NPROC}"
  if [ -n "${requested}" ]; then
    echo "${requested}"
    return
  fi

  if ! [[ "${USE_MULTI_GPU}" =~ ^(1|true|TRUE|yes|YES)$ ]]; then
    echo 1
    return
  fi

  local count=0
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    count=$(nvidia-smi -L | grep -c '^GPU ' || true)
  fi

  if [ "${count}" -lt 1 ] && command -v python >/dev/null 2>&1; then
    count=$(python - <<'PY'
import torch
try:
    print(torch.cuda.device_count())
except Exception:
    print(0)
PY
)
    count=$(echo "${count}" | tr -d '[:space:]')
  fi

  if ! [[ "${count}" =~ ^[0-9]+$ ]]; then
    count=0
  fi

  if [ "${count}" -lt 1 ]; then
    count=1
  fi

  echo "${count}"
}

NPROC_PER_NODE=$(detect_nproc)
TORCHRUN=(torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}")
echo "Using torchrun with ${NPROC_PER_NODE} process(es). Set USE_MULTI_GPU=0 or NANOCHAT_NPROC to override."

# 1) EVAL (mid)
#echo "==> eval (mid)"
#"${TORCHRUN[@]}" -m scripts.chat_eval -- -i mid

# 2) SFT
echo "==> sft (same device_batch_size)"
"${TORCHRUN[@]}" -m scripts.chat_sft -- --device_batch_size=${DEVICE_BATCH}

# 3) EVAL (sft)
#echo "==> eval (sft)"
#"${TORCHRUN[@]}" -m scripts.chat_eval -- -i sft

# 4) report
echo "==> report"
python -m nanochat.report generate

echo "DONE."
