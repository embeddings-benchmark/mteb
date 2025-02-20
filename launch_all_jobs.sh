#!/bin/bash
#SBATCH --job-name=mteb
#SBATCH --partition=h100,nvl
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --exclude=c001,h01,n04,n06,n01
#SBATCH --cpus-per-task=40
#SBATCH --mem=512G
#SBATCH --output=/home/oweller2/my_scratch/mteb/logs-runs/%x_%j_%a.log
#SBATCH --error=/home/oweller2/my_scratch/mteb/logs-runs/%x_%j_%a.log
#SBATCH --array=0-843%12

models=(orionweller/c1-llama3-8b orionweller/c1-14b orionweller/c1-7b orionweller/c1-mistral-2501-24b)
datasets_and_subtasks=(
    # "InstructIR default"
    # "mFollowIRCrossLingual eng-fas"
    # "mFollowIRCrossLingual eng-rus"
    # "mFollowIRCrossLingual eng-zho"
    # "mFollowIR fas"
    # "mFollowIR rus"
    # "mFollowIR zho"
    "BrightRetrieval aops"
    "BrightRetrieval biology"
    "BrightRetrieval earth_science"
    "BrightRetrieval economics"
    "BrightRetrieval leetcode"
    "BrightRetrieval pony"
    "BrightRetrieval psychology"
    "BrightRetrieval robotics"
    "BrightRetrieval stackoverflow"
    "BrightRetrieval sustainable_living"
    "BrightRetrieval theoremqa_questions"
    "BrightRetrieval theoremqa_theorems"
    "ArguAna default"
    "ClimateFEVER default"
    "DBPedia default"
    "FiQA2018 default"
    "NFCorpus default"
    # "NQ default"
    "SCIDOCS default"
    "SciFact default"
    "TRECCOVID default"
    "Touche2020 default"
    # "NevIR default"
)
# Print configuration for logging
echo "Available models:"
printf '%s\n' "${models[@]}"
echo -e "\nAvailable datasets and subtasks:"
printf '%s\n' "${datasets_and_subtasks[@]}"

# Calculate total combinations
total_combinations=$((${#models[@]} * ${#datasets_and_subtasks[@]}))
echo -e "\nTotal number of combinations: $total_combinations"

# Verify array task ID is within bounds
if [[ -z "$SLURM_ARRAY_TASK_ID" ]]; then
    echo "Error: SLURM_ARRAY_TASK_ID is not set"
    exit 1
fi

array_id=$SLURM_ARRAY_TASK_ID

if [[ $array_id -ge $total_combinations ]]; then
    echo "Error: SLURM_ARRAY_TASK_ID ($array_id) is out of bounds. Maximum value should be $((total_combinations - 1))"
    exit 1
fi

# Calculate which model and dataset to use
model_idx=$((array_id / ${#datasets_and_subtasks[@]}))
dataset_idx=$((array_id % ${#datasets_and_subtasks[@]}))

# Get the model and dataset/subtask for this array task
model=${models[$model_idx]}
dataset_and_subtask=${datasets_and_subtasks[$dataset_idx]}

# Split dataset and subtask
dataset=${dataset_and_subtask% *}
subtask=${dataset_and_subtask#* }

# Log the configuration for this run
echo -e "\nRunning combination $array_id:"
echo "Model: $model"
echo "Dataset: $dataset"
echo "Subtask: $subtask"

cd /home/oweller2/my_scratch/mteb

# Run the job
bash launch_job.sh $model $dataset $subtask 1
