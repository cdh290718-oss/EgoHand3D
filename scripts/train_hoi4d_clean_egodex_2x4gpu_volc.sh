#!/usr/bin/env bash
set -e

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/activate.sh

echo "MLP_WORKER_NUM=${MLP_WORKER_NUM}"
echo "MLP_WORKER_GPU=${MLP_WORKER_GPU}"
echo "MLP_ROLE_INDEX=${MLP_ROLE_INDEX}"
echo "MLP_WORKER_0_HOST=${MLP_WORKER_0_HOST}"
echo "MLP_WORKER_0_PORT=${MLP_WORKER_0_PORT}"

export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MASTER_PORT="${MLP_WORKER_0_PORT:-29500}"

torchrun \
  --nproc_per_node="${MLP_WORKER_GPU:-4}" \
  --nnodes="${MLP_WORKER_NUM:-2}" \
  --node_rank="${MLP_ROLE_INDEX:-0}" \
  --master_addr="${MLP_WORKER_0_HOST}" \
  --master_port="${MLP_WORKER_0_PORT:-29500}" \
  train.py \
  exp_name=train_hoi4d_clean_egodex_no2d_2x4gpu_v1 \
  data=hoi4d_clean_egodex_no2d \
  experiment=wilor_vit_refinenet \
  trainer=ddp \
  trainer.strategy=ddp_find_unused_parameters_true \
  trainer.devices="${MLP_WORKER_GPU:-4}" \
  trainer.num_nodes="${MLP_WORKER_NUM:-2}" \
  TRAIN.BATCH_SIZE=8 \
  GENERAL.NUM_WORKERS=8 \
  GENERAL.TOTAL_STEPS=100000 \
  GENERAL.CHECKPOINT_SAVE_TOP_K=3 \
  DATASETS.BETAS_REG=False
