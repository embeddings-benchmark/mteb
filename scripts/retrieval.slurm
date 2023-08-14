#!/bin/bash
#SBATCH --job-name=mteb-retrieval
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --mem-per-cpu=11G # This is essentially 1.1T / 96
#SBATCH --exclusive
#SBATCH --partition=production-cluster
#SBATCH -o /fsx/nouamane/logs/%x-%j-train.out

set -x -e

# source ~/.bashrc
# TODO Replace with your env name
# conda activate 2-0-cu-117

echo "START TIME: $(date)"

MTEB_REPO=/fsx/nouamane/projects/mteb

pushd $MTEB_REPO

GPUS_PER_NODE=8
NNODES=$SLURM_NNODES

# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

export USE_DIST=1

CMD=" \
    scripts/retrieval_multigpu.py
    "

export LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --tee 3 \
    "

echo $CMD

# hide duplicated errors using this hack - will be properly fixed in pt-1.12
# export TORCHELASTIC_ERROR_FILE=/tmp/torch-elastic-error.json

# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=COLL
# export NCCL_SOCKET_NTHREADS=1
# export NCCL_NSOCKS_PERTHREAD=1
# export CUDA_LAUNCH_BLOCKING=1
# export NCCL_ALGO=Ring

# AWS specific
export NCCL_PROTO=simple
export RDMAV_FORK_SAFE=1
export FI_EFA_FORK_SAFE=1
export FI_EFA_USE_DEVICE_RDMA=1
export FI_PROVIDER=efa
export FI_LOG_LEVEL=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=ens

export CUDA_HOME=/usr/local/cuda-11.7
# This is needed for torch1.12.1 otherwise it doesn't link correctly, not sur what the issue was.
# export LD_PRELOAD=$CUDA_HOME/lib/libnccl.so # warning: doesn't work with torch 2.0
export LD_LIBRARY_PATH=$CUDA_HOME/efa/lib:$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# srun error handling:
# --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
# --kill-on-bad-exit=1: terminate a step if any task exits with a non-zero exit code
# SRUN_ARGS=" \
#     --wait=60 \
#     --kill-on-bad-exit=1 \
#     "

srun $SRUN_ARGS -u bash -c "$LAUNCHER --node_rank $SLURM_PROCID --role $SLURMD_NODENAME: $CMD"

# srun doesn't work if this script was called with `srun` itself
# bash -c "$LAUNCHER $CMD"

echo "END TIME: $(date)"
