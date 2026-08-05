#!/bin/bash
# Set up everything needed to evaluate tsinghua-ee/WAVE-7B with MTEB on a new
# (internet-connected) cluster or login node. See MTEB-WAVE-7B.md for background.
#
# Usage:
#   bash scripts/setup_wave_env.sh [WORK_DIR] [--prefetch-model] [--prefetch-data TASK1,TASK2,...]
#
#   WORK_DIR          fast scratch workspace (default: /expscratch/$USER/wave-mteb,
#                     else $SCRATCH/wave-mteb, else $HOME/wave-mteb)
#   --prefetch-model  download the 18 GB WAVE-7B snapshot now (otherwise on first get_model)
#   --prefetch-data   warm the HF cache for the given MTEB task names before GPU time
#
# After it finishes it prints the exports needed in job scripts.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WAVE_REVISION="6d42651d34bf1a7d83d5779397d6ce0316a4cf4f"
FLASH_ATTN_RELEASE="v2.7.4.post1"

# ---- args -------------------------------------------------------------------
WORK=""
PREFETCH_MODEL=0
PREFETCH_DATA=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefetch-model) PREFETCH_MODEL=1 ;;
        --prefetch-data) PREFETCH_DATA="$2"; shift ;;
        *) WORK="$1" ;;
    esac
    shift
done
if [[ -z "$WORK" ]]; then
    if [[ -d "/expscratch/$USER" ]]; then WORK="/expscratch/$USER/wave-mteb"
    elif [[ -n "${SCRATCH:-}" ]]; then WORK="$SCRATCH/wave-mteb"
    else WORK="$HOME/wave-mteb"; fi
fi
echo "==> workspace: $WORK"
mkdir -p "$WORK/beats" "$WORK/logs" "$WORK/.cache"

# ---- caches off /home -------------------------------------------------------
export XDG_CACHE_HOME="$WORK/.cache"
export HF_HOME="$WORK/.cache/huggingface"
export UV_CACHE_DIR="$WORK/.cache/uv"
export PIP_CACHE_DIR="$WORK/.cache/pip"
mkdir -p "$HF_HOME"

# ---- code: submodule --------------------------------------------------------
cd "$REPO_DIR"
git submodule update --init --recursive external/WAVE
test -f external/WAVE/qwenvl/model/qwen2_5_omni/modeling_qwen2_5_omni.py

# ---- env --------------------------------------------------------------------
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
if [[ ! -x "$WORK/.venv/bin/python" ]]; then
    uv venv "$WORK/.venv" --python 3.10
fi
# shellcheck disable=SC1091
source "$WORK/.venv/bin/activate"
uv pip install -e ".[wave,audio,video]"

# ---- flash-attn: matching prebuilt wheel (no source builds) ------------------
SPEC=$(python - <<'PY'
import sys, torch
py  = f"cp{sys.version_info.major}{sys.version_info.minor}"
tch = ".".join(torch.__version__.split("+")[0].split(".")[:2])
cu  = f"cu{torch.version.cuda.split('.')[0]}" if torch.version.cuda else "cpu"
abi = "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE"
print(f"{cu}torch{tch}cxx11abi{abi}-{py}")
PY
)
CU_TORCH_ABI="${SPEC%-*}"
PYTAG="${SPEC#*-}"
WHEEL="flash_attn-${FLASH_ATTN_RELEASE#v}+${CU_TORCH_ABI}-${PYTAG}-${PYTAG}-linux_x86_64.whl"
echo "==> flash-attn wheel: $WHEEL"
uv pip install --no-build-isolation \
    "https://github.com/Dao-AILab/flash-attention/releases/download/${FLASH_ATTN_RELEASE}/${WHEEL}"

# ---- BEATs checkpoint (required at model load; NOT auto-downloaded) ----------
BEATS="$WORK/beats/BEATs_iter3_plus_AS2M.pt"
if [[ ! -s "$BEATS" ]]; then
    curl -fL --retry 3 -o "$BEATS" \
        "https://huggingface.co/datasets/Bencr/beats-checkpoints/resolve/main/BEATs_iter3_plus_AS2M.pt"
fi
export WAVE_BEATS_PATH="$BEATS"

# ---- optional prefetches ------------------------------------------------------
if [[ "$PREFETCH_MODEL" == 1 ]]; then
    python - <<PY
from huggingface_hub import snapshot_download
p = snapshot_download("tsinghua-ee/WAVE-7B", revision="$WAVE_REVISION")
print("model snapshot:", p)
PY
fi
if [[ -n "$PREFETCH_DATA" ]]; then
    python - <<PY
import mteb
for name in "$PREFETCH_DATA".split(","):
    task = mteb.get_tasks(tasks=[name.strip()])[0]
    print("prefetching:", name)
    task.load_data()
PY
fi

# ---- preflight ----------------------------------------------------------------
echo "==> preflight"
python -c "import mteb; m = mteb.get_model_meta('tsinghua-ee/WAVE-7B'); print('registry OK:', m.name, m.embed_dim)"
python -c "from torchcodec.decoders import VideoDecoder" 2>/dev/null \
    && echo "torchcodec OK (ffmpeg libs found)" \
    || echo "WARNING: torchcodec cannot find FFmpeg libs - load an ffmpeg 4-7 module or install ffmpeg (needed for video tasks)"
command -v nvidia-smi >/dev/null \
    && nvidia-smi --query-gpu=name --format=csv,noheader | head -1 \
    || echo "NOTE: no GPU on this node - run evaluations via your scheduler (WAVE needs bf16: A100/L40S/H100, not V100)"

cat <<EOF

Done. In job scripts, set:
  source $WORK/.venv/bin/activate
  export HF_HOME=$HF_HOME
  export WAVE_BEATS_PATH=$BEATS
  # plus ffmpeg libs for video tasks, e.g. on HLTCOE: module load ffmpeg/6.0.1

Smoke test (GPU node):
  python -c "import mteb; m = mteb.get_model('tsinghua-ee/WAVE-7B'); print(type(m).__name__)"
EOF
