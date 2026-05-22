#!/bin/bash
# Usage: bash scripts/mveb_paper/run_eval_script.sh <model_name> <batch_size> [num_frames] [tasks_comma_separated] [prefetch_factor] [cpus_per_task] [mem]

MODEL="$1"
BATCH="${2:-4}"
NUM_FRAMES="$3"
TASKS="$4"
PREFETCH_FACTOR="$5"
CPUS="${6:-1}"
MEM="${7:-16G}"

if [ -z "$MODEL" ]; then
    echo "Usage: bash scripts/mveb_paper/run_eval_script.sh <model_name> <batch_size> [num_frames] [tasks_comma_separated] [prefetch_factor] [cpus_per_task] [mem]"
    exit 1
fi

# Create log dir before anything else
mkdir -p /data/home/niklas/deepshah/logs

# Switch directory allowing the bash to find if result already exists or not
cd /data/home/niklas/deepshah/mteb
if [ -z "$TASKS" ]; then
    TASKS_ARRAY=("")
else
    IFS=',' read -ra TASKS_ARRAY <<< "$TASKS"
fi

for TASK in "${TASKS_ARRAY[@]}"; do
    if [ -n "$TASK" ]; then
        JOB_NAME="${TASK}"
        TASK_ARG="--tasks ${TASK}"
    else
        JOB_NAME="misc-pm"
        TASK_ARG=""
    fi

    if [ -n "$NUM_FRAMES" ]; then
        JOB_NAME="${JOB_NAME}_nf${NUM_FRAMES}"
        FRAME_ARG="--num-frames $NUM_FRAMES"
        OUTPUT_FOLDER="results/num_frame_${NUM_FRAMES}"
    else
        FRAME_ARG="--fps 2.0"
        OUTPUT_FOLDER="results"
    fi

    if [ -n "$PREFETCH_FACTOR" ]; then
        PREFETCH_ARG="--prefetch-factor $PREFETCH_FACTOR"
    else
        PREFETCH_ARG=""
    fi

    if [ -n "$TASK" ]; then
        MODEL_SAFE="${MODEL//\//__}"
        if ls "${OUTPUT_FOLDER}/${MODEL_SAFE}"/*/"${TASK}.json" >/dev/null 2>&1; then
            echo "Skipping task: ${TASK} (already exists for model ${MODEL})"
            continue
        fi
    fi

    echo "Submitting job for task: ${TASK:-ALL}"
    
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=guest
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=72:00:00
#SBATCH --output=/data/home/niklas/deepshah/logs/${JOB_NAME}_%j.out
#SBATCH --error=/data/home/niklas/deepshah/logs/${JOB_NAME}_%j.err

cd /data/home/niklas/deepshah/mteb

# ── Logging header ──────────────────────────────────────────────────
echo "================================================================"
echo "Job \$SLURM_JOB_ID started at \$(date -Iseconds)"
echo "Host: \$(hostname)  |  Model: $MODEL  |  Batch: $BATCH  |  Num Frames: ${NUM_FRAMES:-N/A} | Task: ${TASK:-ALL}"
echo "Partition: \$SLURM_JOB_PARTITION  |  Node: \$SLURM_NODELIST"
echo "================================================================"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
echo "----------------------------------------------------------------"

# ── Environment ─────────────────────────────────────────────────────
export LD_LIBRARY_PATH=/data/home/niklas/adnan/mteb/.ffmpeg/lib:\$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHDYNAMO_DISABLE=1

# ── Run evaluation ──────────────────────────────────────────────────
/data/home/niklas/deepshah/mteb_env_ds/bin/python3 scripts/mveb_paper/eval_suite.py \\
    --model "$MODEL" \\
    --output-folder "$OUTPUT_FOLDER" \\
    $FRAME_ARG \\
    --batch-size $BATCH \\
    $PREFETCH_ARG \
    $TASK_ARG
EXIT_CODE=\$?

# ── Logging footer ──────────────────────────────────────────────────
echo "================================================================"
echo "Job \$SLURM_JOB_ID finished at \$(date -Iseconds)"
echo "Exit code: \$EXIT_CODE"
if [ \$EXIT_CODE -ne 0 ]; then
    echo "FAILED: model=$MODEL batch=$BATCH task=${TASK:-ALL}"
else
    echo "SUCCESS: model=$MODEL batch=$BATCH task=${TASK:-ALL}"
fi
echo "================================================================"
exit \$EXIT_CODE
EOF
done
