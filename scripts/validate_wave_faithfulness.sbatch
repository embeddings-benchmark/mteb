#!/bin/bash
# Run the WAVE-7B numerical faithfulness parity harness on a GPU node.
#
#   sbatch scripts/validate_wave_faithfulness.sbatch
#
# Assumes the env was built by scripts/setup_wave_env.sh. Override the workspace with
#   WORK=/path/to/wave-mteb sbatch scripts/validate_wave_faithfulness.sbatch
# WAVE is bf16 -> request A100/L40S/H100 (not V100). ffmpeg is needed both for synthesizing
# the test clips and for torchcodec's libav* at decode time.
#SBATCH --job-name=wave-parity
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=wave-parity-%j.out
set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WORK="${WORK:-/expscratch/$USER/wave-mteb}"

# shellcheck disable=SC1091
source "$WORK/.venv/bin/activate"
export HF_HOME="${HF_HOME:-$WORK/.cache/huggingface}"
export WAVE_BEATS_PATH="${WAVE_BEATS_PATH:-$WORK/beats/BEATs_iter3_plus_AS2M.pt}"
module load ffmpeg/6.0.1 2>/dev/null || echo "WARN: could not module-load ffmpeg/6.0.1; relying on PATH"

cd "$REPO_DIR"
python scripts/validate_wave_faithfulness.py --work-dir "$WORK/parity_media" "$@"
