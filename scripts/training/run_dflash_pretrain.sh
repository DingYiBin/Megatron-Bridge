#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Pretrain a small MLA + MoE model with DFlash speculative decoding.
#
# Uses 3rdparty/Megatron-LM/pretrain_dflash.py as the entry point.
#
# Usage:
#   bash scripts/training/run_dflash_pretrain.sh <DATA_PATH>
#
# Example:
#   bash scripts/training/run_dflash_pretrain.sh /path/to/data/mydataset_text_document

set -euo pipefail

# ── Script path resolution ──────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── Megatron-LM path (for pretrain_dflash.py + dflash_builders.py) ──────────
MEGATRON_PATH="${REPO_ROOT}/3rdparty/Megatron-LM"
export PYTHONPATH="${MEGATRON_PATH}:${PYTHONPATH}"

# ── Distributed (single node) ───────────────────────────────────────────────
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR=localhost
export MASTER_PORT=7000
GPUS_PER_NODE=${GPUS_PER_NODE:-4}  # override via env, default 4

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes 1
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
    --node_rank 0
)

# ── Experiment paths ────────────────────────────────────────────────────────
DATA_PATH="${1:?Usage: $0 <DATA_PATH>}"
EXPERIMENT_PATH="${EXPERIMENT_PATH:-/tmp/dflash-dev}"
SAVE_PATH="${EXPERIMENT_PATH}/ckpt"
LOGS_PATH="${EXPERIMENT_PATH}/logs"

rm -rf "${EXPERIMENT_PATH}"
mkdir -p "${SAVE_PATH}" "${LOGS_PATH}"

LOGS_FILE="${EXPERIMENT_PATH}/train.log"
echo "Logs: ${LOGS_FILE}"

# ── Tiny DeepSeek-V4-style model ────────────────────────────────────────────
NUM_LAYERS=8
HIDDEN_SIZE=1024
NUM_ATTENTION_HEADS=16
NUM_QUERY_GROUPS=4
KV_CHANNELS=128
FFN_HIDDEN_SIZE=4096
MAX_POSITION_EMBEDDINGS=4096
ROTARY_BASE=10000
SEQ_LENGTH=2048

# MLA config
MLA_Q_LORA_RANK=256
MLA_KV_LORA_RANK=256
MLA_QK_POS_EMB_HEAD_DIM=64
MLA_V_HEAD_DIM=128

# MoE config
NUM_EXPERTS=4
MOE_FFN_HIDDEN_SIZE=2048
MOE_ROUTER_TOPK=2
EXPERT_MODEL_PARALLEL_SIZE=1

# Training
GLOBAL_BATCH_SIZE=64
MICRO_BATCH_SIZE=1
NUM_TRAIN_ITERS=100
SAVE_INTERVAL=50

# DFlash config
DFLASH_NUM_LAYERS=2
DFLASH_NUM_DRAFT_TOKENS=2
DFLASH_TARGET_LAYER_INDICES=(2 5 7)

# ── Tokenizer ────────────────────────────────────────────────────────────────
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/public/llm_models/DeepSeek/DeepSeek-V4-Flash-DSpark}"

# ── Model args ──────────────────────────────────────────────────────────────
GPT_MODEL_ARGS=(
    --transformer-impl transformer_engine
    --no-persist-layer-norm
    --num-layers ${NUM_LAYERS}
    --hidden-size ${HIDDEN_SIZE}
    --ffn-hidden-size ${FFN_HIDDEN_SIZE}
    --num-attention-heads ${NUM_ATTENTION_HEADS}
    --num-query-groups ${NUM_QUERY_GROUPS}
    --kv-channels ${KV_CHANNELS}
    --max-position-embeddings ${MAX_POSITION_EMBEDDINGS}
    --rotary-base ${ROTARY_BASE}
    --seq-length ${SEQ_LENGTH}

    --position-embedding-type rope
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --disable-bias-linear
    --no-masked-softmax-fusion
    --no-rope-fusion
    --no-gradient-accumulation-fusion
    --attention-backend auto

    --multi-latent-attention
    --q-lora-rank ${MLA_Q_LORA_RANK}
    --kv-lora-rank ${MLA_KV_LORA_RANK}
    --qk-pos-emb-head-dim ${MLA_QK_POS_EMB_HEAD_DIM}
    --v-head-dim ${MLA_V_HEAD_DIM}
    --qk-layernorm
    --swiglu

    --num-experts ${NUM_EXPERTS}
    --moe-ffn-hidden-size ${MOE_FFN_HIDDEN_SIZE}
    --moe-router-topk ${MOE_ROUTER_TOPK}
    --moe-router-load-balancing-type seq_aux_loss
    --moe-aux-loss-coeff 1e-4

    --make-vocab-size-divisible-by 1280
    --untie-embeddings-and-output-weights

    --activation-func-clamp-value 10
    --enable-hyper-connections
    --num-residual-streams 4
    --mhc-sinkhorn-iterations 20
    --use-fused-mhc
    --recompute-modules mhc

    # DSpark head (optional, uncomment to enable)
    # --dspark-markov-rank 256
)

# ── DFlash args ─────────────────────────────────────────────────────────────
DFLASH_ARGS=(
    --dflash-num-layers ${DFLASH_NUM_LAYERS}
    --dflash-num-draft-tokens ${DFLASH_NUM_DRAFT_TOKENS}
    --dflash-target-layer-indices ${DFLASH_TARGET_LAYER_INDICES[@]}
    --dflash-loss-scaling-factor 1.0
)

# ── Training args ───────────────────────────────────────────────────────────
TRAINING_ARGS=(
    --bf16
    --lr 1e-4 --min-lr 1e-5
    --lr-decay-style cosine
    --lr-decay-iters 50 --lr-warmup-iters 5
    --weight-decay 0.1 --clip-grad 1.0
    --micro-batch-size ${MICRO_BATCH_SIZE}
    --global-batch-size ${GLOBAL_BATCH_SIZE}
    --train-iters ${NUM_TRAIN_ITERS}
    --enable-experimental
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size ${EXPERT_MODEL_PARALLEL_SIZE}
)

DATA_ARGS=(
    --data-path ${DATA_PATH}
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model ${TOKENIZER_MODEL}
    --split 90,10,0
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval ${SAVE_INTERVAL}
    --eval-interval 163840
    --eval-iters 10
    --save ${SAVE_PATH}
    --load ${SAVE_PATH}
    --tensorboard-dir ${LOGS_PATH}
)

echo "========================================"
echo "DFlash Pretrain (Megatron-Bridge)"
echo "  Experiment:  ${EXPERIMENT_PATH}"
echo "  Model:       ${NUM_LAYERS}L, H=${HIDDEN_SIZE}, MLA, MoE(N=${NUM_EXPERTS},K=${MOE_ROUTER_TOPK})"
echo "  DFlash:      d=${DFLASH_NUM_LAYERS}, s=${DFLASH_NUM_DRAFT_TOKENS}"
echo "  Target L:    ${DFLASH_TARGET_LAYER_INDICES[*]}"
echo "  GPUs:        ${GPUS_PER_NODE}"
echo "  Entry:       ${MEGATRON_PATH}/pretrain_dflash.py"
echo "========================================"

# ── Launch ──────────────────────────────────────────────────────────────────
uv run python -m torch.distributed.run "${DISTRIBUTED_ARGS[@]}" \
    "${MEGATRON_PATH}/pretrain_dflash.py" \
    "${GPT_MODEL_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${DFLASH_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${EVAL_AND_LOGGING_ARGS[@]}" \
    &> "${LOGS_FILE}"
