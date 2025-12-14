#!/usr/bin/env bash
set -euo pipefail

# Max-fit nanochat for Low-End GPUs (e.g., RTX 3050 6GB)
# Optimized for 8-bit optimizer and FP16 to minimize VRAM usage.

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
fi
echo "Detected ${NGPUS} GPUs."

if [ "${HAVE_NVIDIA}" -eq 1 ]; then
  uv sync --extra cuda
else
  uv sync --extra cpu
fi

source .venv/bin/activate

: "${WANDB_RUN:=lowend3050}"
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

if [ -f "${NANOCHAT_BASE_DIR}/tokenizer/tokenizer.pkl" ] || [ -f "${NANOCHAT_BASE_DIR}/tokenizer/tokenizer.json" ]; then
    echo "Tokenizer already exists, skipping training."
else
    python -m nanochat.dataset -n "${TOK_SHARDS}"
    python -m nanochat.dataset -n "${ALL_SHARDS}" &
    
    python -m scripts.tok_train --max_chars="${TOK_MAX_CHARS}"
    if [ ! -f "${NANOCHAT_BASE_DIR}/tokenizer/tokenizer.pkl" ] && [ ! -f "${NANOCHAT_BASE_DIR}/tokenizer/tokenizer.json" ]; then
        echo "FATAL: Tokenizer training failed."
        exit 1
    fi
fi
# We always run tok_eval just to be sure it loads, but it's fast
python -m scripts.tok_eval

# ---------- 5) Probe for max model (Dynamic Resource Detection) ---------------- #
# Low-end Strategy: Priotize generic 'working' over 'max fitting'.
# Safe defaults for 6GB VRAM + 8bit opt.

BEST_DBSZ=""
BEST_DEPTH=""
BEST_SEQLEN=""

# Smart Resume: Check if we have a checkpoint to resume from
LATEST_META=""
if [ -d "${NANOCHAT_BASE_DIR}/base_checkpoints" ]; then
    LATEST_META=$(python -c "
import os
import glob
base = '${NANOCHAT_BASE_DIR}/base_checkpoints'
metas = glob.glob(os.path.join(base, 'd*', 'meta_*.json'))
if metas:
    def get_step(p):
        try:
            return int(os.path.basename(p).split('_')[1].split('.')[0])
        except:
            return -1
    metas.sort(key=get_step, reverse=True)
    print(metas[0])
")
fi

if [ -n "${LATEST_META}" ]; then
    echo "Found existing checkpoint meta: ${LATEST_META}"
    READ_CONFIG=$(python -c "
import json
try:
    with open('${LATEST_META}') as f:
        meta = json.load(f)
    uc = meta.get('user_config', {})
    mc = meta.get('model_config', {})
    d = uc.get('depth', mc.get('n_layer'))
    s = uc.get('max_seq_len', mc.get('sequence_len'))
    b = uc.get('device_batch_size')
    print(f'{d} {s} {b}')
except Exception:
    print('')
")
    if [ -n "${READ_CONFIG}" ]; then
        read_depth=$(echo "${READ_CONFIG}" | awk '{print $1}')
        read_seqlen=$(echo "${READ_CONFIG}" | awk '{print $2}')
        read_dbsz=$(echo "${READ_CONFIG}" | awk '{print $3}')
        
        if [ "${read_depth}" != "None" ] && [ "${read_seqlen}" != "None" ] && [ "${read_dbsz}" != "None" ]; then
             BEST_DEPTH="${read_depth}"
             BEST_SEQLEN="${read_seqlen}"
             BEST_DBSZ="${read_dbsz}"
             echo "=== RESUMING with detected config: Depth=${BEST_DEPTH}, SeqLen=${BEST_SEQLEN}, Batch=${BEST_DBSZ} ==="
        fi
    fi
fi


if [ -z "${BEST_DEPTH}" ]; then
    echo "=== Probing maximum depth/seq-len/dbsz based on available VRAM (Low-End) ==="
    
    GPU_VRAM_MIB=0
    if [ "${HAVE_NVIDIA}" -eq 1 ]; then
      GPU_VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
    elif [ "${HAVE_XPU}" -eq 1 ]; then
      GPU_VRAM_MIB=$(python -c "import torch; import intel_extension_for_pytorch; print(int(torch.xpu.get_device_properties(0).total_memory / 1024 / 1024))")
    fi
    echo "Detected VRAM per GPU: ${GPU_VRAM_MIB} MiB"

    # Candidates optimized for 6GB-8GB VRAM
    if [ "$GPU_VRAM_MIB" -gt 6000 ]; then
        # 6GB+
        CANDIDATE_DEPTHS=(16 12 10 8)
        CANDIDATE_DEVICE_BATCHES=(4 2 1)
    elif [ "$GPU_VRAM_MIB" -gt 3500 ]; then
        # 4GB+
        CANDIDATE_DEPTHS=(8 6)
        CANDIDATE_DEVICE_BATCHES=(2 1)
    else
        # Very low VRAM or CPU fallback
        CANDIDATE_DEPTHS=(4)
        CANDIDATE_DEVICE_BATCHES=(1)
    fi
    CANDIDATE_SEQLENS=(1024 512)

    try_probe () {
      local db="$1"
      local depth="$2"
      local seqlen="$3"
      
      echo "  ... trying dbsz=${db} depth=${depth} seq_len=${seqlen} in subprocess ..."
      
      # Invoke scripts.train_lowend with --use_8bit_optimizer=True
      torchrun --standalone --nproc_per_node="${NGPUS}" -m scripts.train_lowend \
        --depth="${depth}" \
        --max_seq_len="${seqlen}" \
        --device_batch_size="${db}" \
        --total_batch_size=8192 \
        --num_iterations=2 \
        --eval_every=999999 \
        --sample_every=999999 \
        --core_metric_every=999999 \
        --eval_tokens=1024 \
        --use_8bit_optimizer=True \
        >/dev/null 2>&1
    }
    
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
      echo "WARN: Standard probes failed. Trying safety fallback (Depth=4, SeqLen=512, Batch=1)..."
      if try_probe "1" "4" "512"; then
          BEST_DEPTH=4
          BEST_SEQLEN=512
          BEST_DBSZ=1
          echo "  -> Safety fallback passed."
      else
          echo "FATAL: Could not fit even the smallest model."
          exit 1
      fi
    fi
fi

# ---------- 6) Global knobs optimized for detected settings ------------------
BASE_TOTAL_BATCH=65536     # Reduced from 256k/512k for faster updates on low-end
EVAL_EVERY=250
EVAL_TOKENS=2048           # Reduced
BASE_ITERS=4000            # More iters since batch is smaller? Or keep similar.

echo "=== FINAL CONFIG (Auto-Detected) ==="
echo "DEPTH:          ${BEST_DEPTH}"
echo "SEQ_LEN:        ${BEST_SEQLEN}"
echo "DEVICE_BATCH:   ${BEST_DBSZ}"
echo "BASE_TOTAL:     ${BASE_TOTAL_BATCH}"
echo "BASE_ITERS:     ${BASE_ITERS}"
echo "====================================================="

# ---------- 7) Base training (Low-End) ---------------------------------------
torchrun --standalone --nproc_per_node="${NGPUS}" -m scripts.train_lowend \
  --depth="${BEST_DEPTH}" \
  --max_seq_len="${BEST_SEQLEN}" \
  --device_batch_size="${BEST_DBSZ}" \
  --total_batch_size="${BASE_TOTAL_BATCH}" \
  --eval_every="${EVAL_EVERY}" \
  --eval_tokens="${EVAL_TOKENS}" \
  --core_metric_every="${EVAL_EVERY}" \
  --core_metric_max_per_task=12 \
  --sample_every="${EVAL_EVERY}" \
  --num_iterations="${BASE_ITERS}" \
  --use_8bit_optimizer=True

# ---------- 8) base_loss / base_eval -----------------------------------------
# Note: These use base_loss/eval (standard scripts). They might need adaptation if they are too heavy.
# But usually eval is lighter than train. If they fail, user might need 'eval_lowend.py' too.
# For now, we assume they fit since they don't hold backward graph.

echo "==> base_loss"
torchrun --standalone --nproc_per_node="${NGPUS}" -m scripts.base_loss -- --device_batch_size="${BEST_DBSZ}" --split_tokens="${EVAL_TOKENS}"

echo "==> base_eval"
# We should probably reduce max-per-task for low end
torchrun --standalone --nproc_per_node="${NGPUS}" -m scripts.base_eval -- --max-per-task=16

# ======================================================================
# FROM HERE ON: repo-native mid/SFT
# ======================================================================

# TODO: mid_train and chat_sft might invoke standard GPT. If we want low-end continuous pipeline, we'd need mid_train_lowend etc.
# But for now, user asked specifically for "make we support an option that makes the model train in 8bit"
# We focused on the base training. If user wants full pipeline low-end, we might need more files.
# For now, let's try running them with the detected batch size. 
# Warning: mid_train imports nanochat.gpt.GPT, so it won't use our gptlowend.
# This might break providing weights. 
# Ideally, we should stop here or advise user that only base training uses the new optimizations.
# But let's act as if the user wants to train base model mainly.
# We will comment out the rest or warn.

echo "DONE with Base Training."
echo "NOTE: Downstream stages (mid_train, sft) currently use standard GPT implementation."
