#!/usr/bin/env bash
set -e

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/activate.sh

export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MASTER_PORT="${MASTER_PORT:-29500}"

torchrun \
  --nproc_per_node=4 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port="${MASTER_PORT}" \
  train.py \
  exp_name=train_hoi4d_clean_4gpu_v1 \
  data=hoi4d_clean \
  experiment=wilor_vit_refinenet \
  trainer=ddp \
  trainer.strategy=ddp_find_unused_parameters_true \
  trainer.devices=4 \
  trainer.num_nodes=1 \
  TRAIN.BATCH_SIZE=8 \
  GENERAL.NUM_WORKERS=8 \
  GENERAL.TOTAL_STEPS=100000 \
  GENERAL.CHECKPOINT_SAVE_TOP_K=3
