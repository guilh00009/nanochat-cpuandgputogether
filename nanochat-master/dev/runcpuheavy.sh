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
  # Assuming CPU, but script below might fail if it strictly expects GPU for probing
fi
echo "Detected ${NGPUS} GPUs."

if [ "${HAVE_NVIDIA}" -eq 1 ]; then
  uv sync --extra cuda
else
  uv sync --extra cpu
fi

source .venv/bin/activate

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

python -m nanochat.dataset -n "${TOK_SHARDS}"
python -m nanochat.dataset -n "${ALL_SHARDS}" &

python -m scripts.tok_train --max_chars="${TOK_MAX_CHARS}"
python -m scripts.tok_eval

# ---------- 5) Probe for max model (Dynamic Resource Detection) ---------------- #
# We want to find the MAX DEPTH that fits in memory, with at least seq_len=2048.
# Priority: Depth (Model Size) > SeqLen > Batch Size.

echo "=== Probing maximum depth/seq-len/dbsz based on available VRAM ==="

if [ "${HAVE_NVIDIA}" -ne 1 ] && [ "${HAVE_XPU}" -ne 1 ]; then
  echo "No NVIDIA or Intel Arc GPU detected. This script implementation requires GPUs."
  exit 1
fi

# Detect VRAM of the first GPU (assuming homogeneous cluster for now)
# Unit: MiB
if [ "${HAVE_NVIDIA}" -eq 1 ]; then
  GPU_VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
elif [ "${HAVE_XPU}" -eq 1 ]; then
  GPU_VRAM_MIB=$(python -c "import torch; import intel_extension_for_pytorch; print(int(torch.xpu.get_device_properties(0).total_memory / 1024 / 1024))")
fi
echo "Detected VRAM per GPU: ${GPU_VRAM_MIB} MiB"

# Define Candidate Lists based on VRAM Scale
# If > 70GB (e.g. A100/H100 80GB), try deeper models.
# If > 35GB (e.g. A100 40GB/A6000), standard range.
# If > 20GB (e.g. 3090/4090), lower range.
# Else (consumer small), minimal range.

if [ "$GPU_VRAM_MIB" -gt 70000 ]; then
    # ~80GB: Can probably go significantly deeper or larger batch
    CANDIDATE_DEPTHS=(80 72 64 60 56 52 48 44 40)
    CANDIDATE_DEVICE_BATCHES=(16 8 4 2 1)
elif [ "$GPU_VRAM_MIB" -gt 35000 ]; then
    # ~40GB-48GB
    CANDIDATE_DEPTHS=(52 48 44 40 36 32 28 24)
    CANDIDATE_DEVICE_BATCHES=(8 4 2 1)
elif [ "$GPU_VRAM_MIB" -gt 20000 ]; then
    # ~24GB
    CANDIDATE_DEPTHS=(32 28 24 20 16 12)
    CANDIDATE_DEVICE_BATCHES=(4 2 1)
else
    # ~16GB or less
    CANDIDATE_DEPTHS=(24 20 16 12 8)
    CANDIDATE_DEVICE_BATCHES=(2 1)
fi

# Ideally we want larger SeqLen, but 2048 is the hard floor.
CANDIDATE_SEQLENS=(4096 2048)

BEST_DBSZ=""
BEST_DEPTH=""
BEST_SEQLEN=""

try_probe () {
  local db="$1"
  local depth="$2"
  local seqlen="$3"
  
  # For very small depths, increase batch size minimum to avoid inefficient training
  if [ "$depth" -le 16 ] && [ "$db" -eq 1 ]; then
      # Optional optimization: Skip batch=1 for small models unless strictly necessary?
      # For now, we keep it to ensure *something* runs.
      :
  fi
  
  echo "  ... trying dbsz=${db} depth=${depth} seq_len=${seqlen} in subprocess ..."
  
  # Run a quick probe: 2 iterations
  torchrun --standalone --nproc_per_node="${NGPUS}" -m scripts.base_train \
    --depth="${depth}" \
    --max_seq_len="${seqlen}" \
    --device_batch_size="${db}" \
    --total_batch_size=32768 \
    --num_iterations=2 \
    --eval_every=999999 \
    --sample_every=999999 \
    --core_metric_every=999999 \

    --eval_tokens=2048 \
    >/dev/null 2>&1
}

# PROBING LOOP:
# We iterate Dept (High->Low) -> SeqLen (High->Low) -> Batch (High->Low)
# The first one that succeeds is our winner.

found=0
for depth in "${CANDIDATE_DEPTHS[@]}"; do
  for seqlen in "${CANDIDATE_SEQLENS[@]}"; do
    for db in "${CANDIDATE_DEVICE_BATCHES[@]}"; do
      echo "Probe: Depth=${depth} | SeqLen=${seqlen} | Batch=${db} [GPUs=${NGPUS}]"
      
      if try_probe "${db}" "${depth}" "${seqlen}"; then
        BEST_DEPTH="${depth}"
        BEST_SEQLEN="${seqlen}"
        BEST_DBSZ="${db}"
        found=1
        echo "  -> SUCCESS! Found max fit."
        break 3
      else
        echo "  -> OOM or Fail."
      fi
    done
  done
done

if [ "${found}" -eq 0 ]; then
  # Fallback to absolute minimums if specific probe list failed, essentially a Hail Mary
  echo "WARN: Standard probes failed. Trying safety fallback (Depth=8, SeqLen=2048, Batch=1)..."
  if try_probe "1" "8" "2048"; then
      BEST_DEPTH=8
      BEST_SEQLEN=2048
      BEST_DBSZ=1
      echo "  -> Safety fallback passed."
  else
      echo "FATAL: Could not fit even the smallest model (Depth=8)."
      exit 1
  fi
fi

# ---------- 6) Global knobs optimized for detected settings ------------------
# Adjust Total Batch Size based on scale? For now keep fixed large batch for stability.
BASE_TOTAL_BATCH=262144     # 256k tokens
EVAL_EVERY=250
# If we have a huge context, we might want to eval more tokens, but 4096 is standard.
EVAL_TOKENS=4096
# Adjust iterations? If model is huge, maybe fewer steps? 
# Current logic: fixed iters.
BASE_ITERS=8000

echo "=== FINAL CONFIG (Auto-Detected) ==="
if [ "${HAVE_NVIDIA}" -eq 1 ]; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n 1)
elif [ "${HAVE_XPU}" -eq 1 ]; then
    GPU_INFO=$(python -c "import torch; import intel_extension_for_pytorch; print(f'{torch.xpu.get_device_name(0)}, {int(torch.xpu.get_device_properties(0).total_memory/1024/1024)} MiB')")
fi
echo "GPU info:       ${GPU_INFO} x ${NGPUS}"
echo "DEPTH:          ${BEST_DEPTH}"
echo "SEQ_LEN:        ${BEST_SEQLEN}"
echo "DEVICE_BATCH:   ${BEST_DBSZ}"
echo "BASE_TOTAL:     ${BASE_TOTAL_BATCH}"
echo "BASE_ITERS:     ${BASE_ITERS}"
echo "====================================================="

# ---------- 7) Base training --------------------------------------------------
torchrun --standalone --nproc_per_node="${NGPUS}" -m scripts.base_train \
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
torchrun --standalone --nproc_per_node="${NGPUS}" -m scripts.base_loss -- --device_batch_size="${BEST_DBSZ}" --split_tokens="${EVAL_TOKENS}"
torchrun --standalone --nproc_per_node="${NGPUS}" -m scripts.base_eval -- --max-per-task=32

# ======================================================================
# FROM HERE ON: repo-native mid/SFT (your working style)
# ======================================================================

# 9) MID-TRAIN
echo "==> mid-train (repo-native args only)"
torchrun --standalone --nproc_per_node="${NGPUS}" -m scripts.mid_train -- --device_batch_size=${BEST_DBSZ}

# 10) EVAL (mid)
echo "==> eval (mid)"
torchrun --standalone --nproc_per_node="${NGPUS}" -m scripts.chat_eval -- -i mid

# 11) SFT
echo "==> sft (same device_batch_size)"
torchrun --standalone --nproc_per_node="${NGPUS}" -m scripts.chat_sft -- --device_batch_size=${BEST_DBSZ}

# 12) EVAL (sft)
echo "==> eval (sft)"
torchrun --standalone --nproc_per_node="${NGPUS}" -m scripts.chat_eval -- -i sft

# 13) report
echo "==> report"
python -m nanochat.report generate

echo "DONE."
