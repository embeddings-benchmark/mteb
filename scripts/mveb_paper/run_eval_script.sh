#!/bin/bash
# Usage: sbatch scripts/run_eval_script.sh <model_name> <batch_size> [num_frames] [tasks_comma_separated]

#SBATCH --job-name=misc-pm
#SBATCH --partition=guest
#SBATCH --gres=gpu:4
#SBATCH --exclude=slinky-5
#SBATCH --time=72:00:00
#SBATCH --output=logs/misc_pm_%j.out
#SBATCH --error=logs/misc_pm_%j.err

MODEL="$1"
BATCH="${2:-4}"
NUM_FRAMES="$3"
TASKS="$4"

# Create log dir before anything else (SBATCH output/error needs it)
mkdir -p /data/home/niklas/deepshah/mteb/logs
cd /data/home/niklas/deepshah/mteb

# ── Logging header ──────────────────────────────────────────────────
echo "================================================================"
echo "Job $SLURM_JOB_ID started at $(date -Iseconds)"
echo "Host: $(hostname)  |  Model: $MODEL  |  Batch: $BATCH  |  Num Frames: ${NUM_FRAMES:-N/A}"
echo "Partition: $SLURM_JOB_PARTITION  |  Node: $SLURM_NODELIST"
echo "================================================================"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
echo "----------------------------------------------------------------"

# ── Environment ─────────────────────────────────────────────────────
export LD_LIBRARY_PATH=/data/home/niklas/adnan/mteb/.ffmpeg/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHDYNAMO_DISABLE=1

# ── Task formatting ─────────────────────────────────────────────────
if [ -n "$TASKS" ]; then
    # Convert comma-separated string to space-separated for Python
    TASK_ARG="--tasks $(echo "$TASKS" | tr ',' ' ')"
else
    TASK_ARG=""
fi

# ── Run evaluation (do NOT set -e; we want to capture the exit code) ─
if [ -n "$NUM_FRAMES" ]; then
    FRAME_ARG="--num-frames $NUM_FRAMES"
    OUTPUT_FOLDER="results/num_frame_${NUM_FRAMES}"
else
    FRAME_ARG="--fps 2.0"
    OUTPUT_FOLDER="results"
fi

python scripts/mveb_paper/eval_suite.py \
    --model "$MODEL" \
    --output-folder "$OUTPUT_FOLDER" \
    $FRAME_ARG \
    --batch-size $BATCH \
    $TASK_ARG
EXIT_CODE=$?

# ── Logging footer ──────────────────────────────────────────────────
echo "================================================================"
echo "Job $SLURM_JOB_ID finished at $(date -Iseconds)"
echo "Exit code: $EXIT_CODE"
if [ $EXIT_CODE -ne 0 ]; then
    echo "FAILED: model=$MODEL batch=$BATCH"
else
    echo "SUCCESS: model=$MODEL batch=$BATCH"
fi
echo "================================================================"
