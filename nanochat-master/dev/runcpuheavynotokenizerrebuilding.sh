#!/usr/bin/env bash
set -euo pipefail

# Max-fit nanochat for ~40GB VRAM / ~110GB RAM
# tokenizer -> base -> loss/eval -> mid (repo-native) -> sft (repo-native) -> eval -> report
# IMPORTANT: seq_len will NEVER be < 2048. If 2048 doesn't fit, we abort.

export OMP_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
NANOCHAT_BASE_DIR="${HOME}/.cache/nanochat"
mkdir -p "${NANOCHAT_BASE_DIR}"

# ---------- 1) Bootstrap ------------------------------------------------------
if ! command -v uv &>/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  if [ -f "${HOME}/.cargo/bin/uv" ]; then export PATH="${HOME}/.cargo/bin:${PATH}"; fi
fi
[ -d ".venv" ] || uv venv

HAVE_NVIDIA=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  HAVE_NVIDIA=1
fi

if [ "${HAVE_NVIDIA}" -eq 1 ]; then
  uv sync --extra cuda
else
  uv sync --extra cpu
fi

source .venv/bin/activate

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
  if [ "${HAVE_NVIDIA}" -eq 1 ]; then
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

: "${WANDB_RUN:=big40}"
python -m nanochat.report reset

# ---------- 2) Rust + rustbpe -------------------------------------------------
if ! command -v cargo >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
# shellcheck source=/dev/null
source "${HOME}/.cargo/env"
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# ---------- 3) Eval bundle + identity ----------------------------------------
EVAL_BUNDLE_URL="https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"
if [ ! -d "${NANOCHAT_BASE_DIR}/eval_bundle" ]; then
  curl -L -o eval_bundle.zip "${EVAL_BUNDLE_URL}"
  unzip -q eval_bundle.zip && rm eval_bundle.zip
  mv eval_bundle "${NANOCHAT_BASE_DIR}"
fi
curl -L -o "${NANOCHAT_BASE_DIR}/identity_conversations.jsonl" \
  https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# ---------- 4) Tokenizer & data (big RAM) ------------------------------------
TOK_MAX_CHARS=2000000000     # 2B chars
TOK_SHARDS=32
ALL_SHARDS=800               # like the full run
# ---------- 5) Fixed training configuration ----------------------------------
if [ "${HAVE_NVIDIA}" -ne 1 ]; then
  echo "No NVIDIA GPU detected; this script expects a 40GB GPU. Exiting."
  exit 1
fi

BEST_DBSZ="${DEVICE_BATCH_SIZE_OVERRIDE:-1}"
BEST_DEPTH="${DEPTH_OVERRIDE:-92}"
BEST_SEQLEN="${SEQ_LEN_OVERRIDE:-4096}"

# ---------- 6) Global knobs for 40GB -----------------------------------------
BASE_TOTAL_BATCH=262144     # 256k tokens
EVAL_EVERY=25
EVAL_TOKENS=4096
BASE_ITERS=8000             # long run; adjust if needed

echo "=== FINAL CONFIG (40GB, seq_len >= 2048) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "DEPTH:          ${BEST_DEPTH}"
echo "SEQ_LEN:        ${BEST_SEQLEN}"
echo "DEVICE_BATCH:   ${BEST_DBSZ}"
echo "BASE_TOTAL:     ${BASE_TOTAL_BATCH}"
echo "BASE_ITERS:     ${BASE_ITERS}"
echo "====================================================="

# ---------- 7) Base training --------------------------------------------------
"${TORCHRUN[@]}" -m scripts.base_train \
  --depth="${BEST_DEPTH}" \
  --max_seq_len="${BEST_SEQLEN}" \
  --device_batch_size="${BEST_DBSZ}" \
  --total_batch_size="${BASE_TOTAL_BATCH}" \
  --eval_every="${EVAL_EVERY}" \
  --eval_tokens="${EVAL_TOKENS}" \
  --core_metric_every="${EVAL_EVERY}" \
  --core_metric_max_per_task=12 \
  --sample_every="${EVAL_EVERY}" \
  --num_iterations="${BASE_ITERS}"

# ---------- 8) base_loss / base_eval -----------------------------------------
"${TORCHRUN[@]}" -m scripts.base_loss -- --device_batch_size="${BEST_DBSZ}" --split_tokens="${EVAL_TOKENS}"
"${TORCHRUN[@]}" -m scripts.base_eval -- --max-per-task=32

# ======================================================================
# FROM HERE ON: repo-native mid/SFT (your working style)
# ======================================================================

# 9) MID-TRAIN
echo "==> mid-train (repo-native args only)"
"${TORCHRUN[@]}" -m scripts.mid_train -- --device_batch_size=${BEST_DBSZ}

# 10) EVAL (mid)
echo "==> eval (mid)"
"${TORCHRUN[@]}" -m scripts.chat_eval -- -i mid

# 11) SFT
echo "==> sft (same device_batch_size)"
"${TORCHRUN[@]}" -m scripts.chat_sft -- --device_batch_size=${BEST_DBSZ}

# 12) EVAL (sft)
echo "==> eval (sft)"
"${TORCHRUN[@]}" -m scripts.chat_eval -- -i sft

# 13) report
echo "==> report"
python -m nanochat.report generate

echo "DONE."
