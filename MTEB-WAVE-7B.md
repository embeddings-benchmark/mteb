# Adding WAVE-7B to MTEB

A concise, shareable record of how `tsinghua-ee/WAVE-7B` was integrated into this MTEB clone
as a first-class registry model. Companion to **MTEB benchmark runbook.md**.

## What WAVE-7B is

| Item | Value |
| :--- | :--- |
| Model | [`tsinghua-ee/WAVE-7B`](https://huggingface.co/tsinghua-ee/WAVE-7B) (arXiv:2509.21990, ICLR 2026 oral) |
| Code | [github.com/TCL606/WAVE](https://github.com/TCL606/WAVE) |
| Base | Fine-tune of `Qwen/Qwen2.5-Omni-7B` (Thinker), hidden size **3584** |
| Modalities | text, audio, silent video, synchronized audio-visual (one shared space) |
| Embeddings | "prompt-aware"; any-to-any retrieval. SOTA on MMEB-v2-video (59.9) |
| License | apache-2.0 |
| Registry name | `tsinghua-ee/WAVE-7B`, revision `6d42651d34bf1a7d83d5779397d6ce0316a4cf4f` |

## The one non-obvious fact that shapes everything

The HF repo ships **only weights + a vanilla `Qwen2_5OmniThinkerForConditionalGeneration`
config** — no custom modeling code, no `auto_map`. Loading it with stock `transformers` (or the
existing `QwenOmniWrapper`) will **not** reproduce WAVE's embeddings, because WAVE adds:

- a **BEATs dual audio encoder** (a second audio tower whose features are interleaved with the
  Whisper audio features — each `<|AUDIO|>` token is doubled), and
- **hierarchical "all-layer" fusion**: the embedding is the last-token hidden state of **every**
  transformer layer concatenated, then passed through a learned `classify_linear` head.

That logic lives only in the GitHub repo (`qwenvl/model/qwen2_5_omni/`), driven at eval time by
`--use_beats --classify_type all_layer --pred_embeds`. So a faithful integration must run WAVE's
own code, not stock transformers.

## What was added (4 pieces)

1. **Submodule** — WAVE's upstream code, vendored at `external/WAVE` (`.gitmodules`):
   ```
   git submodule add https://github.com/TCL606/WAVE.git external/WAVE
   git submodule update --init --recursive external/WAVE
   ```
2. **Wrapper** — `mteb/models/model_implementations/wave_models.py` → `Wave7BWrapper(AbsEncoder)`.
   It adds `external/WAVE` to `sys.path`, loads WAVE's `Qwen2_5OmniThinkerForConditionalGeneration`
   (config flags `train_classify=True, classify_type="all_layer"`), loads the BEATs backbone
   checkpoint, and reproduces WAVE's `--pred_embeds` path:

   ```text
   per item -> build WAVE inputs (reusing process_audio / process_omni_conversations /
               replace_multimodal_special_tokens) -> model(**inputs, pred_embeds=True)
            -> outputs.mllm_embeds -> L2-normalize
   ```

   Items are encoded one at a time (matching WAVE's batch-size-1 eval). MTEB's `AudioCollator` /
   `VideoCollator` resample audio to 16 kHz and sample video frames at `fps=2.0` (WAVE's
   `base_interval=0.5`), capped at `max_frames=128`.
3. **`ModelMeta`** (`wave_7b` in the same file) — auto-discovered by the registry (no
   `__init__.py` edit). `embed_dim=3584`, `modalities=["text","audio","video"]`,
   `model_type=["dense"]`, `use_instructions=True`, `extra_requirements_groups=["wave"]`.
4. **Optional-deps group** — `wave` in `pyproject.toml` (mirrors WAVE's `requirements.txt`:
   `transformers==4.51.3`, `liger_kernel==0.5.10`, `torchvision==0.21.0`, plus `decord`,
   `soundfile`, `peft`, `accelerate`, `flash-attn`). Registered under `[tool.uv].conflicts`
   because the `transformers==4.51.3` pin is incompatible with several other model extras.

## What did NOT need to change

- **Datasets** — unchanged. Tasks auto-download their own data.
- **Evaluators** — unchanged. They consume the `encode()` embeddings and compute cosine
  similarity; nothing model-specific.
- **Tasks** — none added. MTEB already covers every WAVE modality combination:
  - text↔audio: `ClothoT2ARetrieval`/`ClothoA2TRetrieval`, `UrbanSound8K{T2A,A2T}Retrieval`,
    `AudioCaps{A2T,T2A}Retrieval` (+ other MAEB retrieval tasks)
  - text↔video: `MSVD{T2V,V2T}Retrieval`, `MSRVTT*`, `ActivityNetCaptionsV2TRetrieval`,
    `TUNABench{T2V,V2T}Retrieval`
  - audio-visual: `AudioCapsAV*`, `VGGSoundAV*`, `DiDeMo*`, `AVMemeExam*` (v2t/t2v/va2t/v2a/a2v/…)

## Two external artifacts required at load time (not in this repo)

1. **WAVE-7B weights** — downloaded from the HF Hub automatically.
2. **BEATs backbone** — `BEATs_iter3_plus.pt` (Microsoft BEATs iter3+). Point to it with the
   `beats_path` loader kwarg or `WAVE_BEATS_PATH` env var. The wrapper raises a clear error if
   it is missing.

## How to run a pilot (HLTCOE grid — verified recipe)

WAVE pins `torch==2.6.0` / `transformers==4.51.3`, incompatible with the main MTEB env, so use an
**isolated venv** with the `wave` extra. WAVE uses bf16 → request **A100/H100/L40S** (not V100).

```bash
# --- on a login node: build the isolated env on flash scratch ---
WORK=/expscratch/$USER/wave-mteb; mkdir -p $WORK/beats $WORK/logs
export HF_HOME=$WORK/.cache/huggingface UV_CACHE_DIR=$WORK/.cache/uv
git submodule update --init --recursive external/WAVE
uv venv "$WORK/.venv" --python 3.10 && source "$WORK/.venv/bin/activate"
uv pip install -e ".[wave,audio,video]"        # wave extra pins the torch 2.6 stack + ST<5
# flash-attn: install the prebuilt wheel matching cp310 / torch2.6 / cu12 / abiFALSE
uv pip install --no-build-isolation \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# BEATs backbone checkpoint (WAVE's "BEATs_iter3_plus")
curl -fL -o $WORK/beats/BEATs_iter3_plus_AS2M.pt \
  https://huggingface.co/datasets/Bencr/beats-checkpoints/resolve/main/BEATs_iter3_plus_AS2M.pt
export WAVE_BEATS_PATH=$WORK/beats/BEATs_iter3_plus_AS2M.pt
```

Pilot script (`pilot_wave.py`) and `sbatch` (`--gres=gpu:a100:1 --cpus-per-task=8 --mem=64G`,
`module load cuda/... ffmpeg/6.0.1`, export `HF_HOME`/`WAVE_BEATS_PATH`):

```python
import mteb
model = mteb.get_model("tsinghua-ee/WAVE-7B")             # reads WAVE_BEATS_PATH
tasks = mteb.get_tasks(tasks=["ClothoT2ARetrieval"])       # small audio↔text pilot
results = mteb.evaluate(model, tasks=tasks, encode_kwargs={"batch_size": 1})
```

Progression: `ClothoT2ARetrieval` (audio) → `MSVDT2VRetrieval` (video) →
`AudioCapsAVT2VRetrieval` (audio-visual).

## Verified — pilot run on the grid ✅

Ran on **A100-PCIE-40GB** (`torch 2.6.0+cu124`, `flash_attn 2.7.4.post1`), job COMPLETED in 13:30:

- Model loaded in **219 s**; log shows `Init BEATs Model` + `Classify Type: all_layer` — the BEATs
  dual encoder and all-layer fusion (the faithful path) are active.
- `ClothoT2ARetrieval` (text→audio, 1045-doc corpus) finished in **548 s**, `exceptions=[]`:

  | metric | value |
  | :-- | :-- |
  | **hit_rate@5 (main_score)** | **0.317** |
  | ndcg@10 | 0.260 |
  | recall@5 / @100 / @1000 | 0.317 / 0.727 / 0.994 |
  | map@10 | 0.213 |

  Non-trivial scores over ~1k candidates confirm the wrapper produces meaningful cross-modal
  embeddings (not random) via WAVE's real `--pred_embeds` path.

Also verified: `wave_models` imports without WAVE deps; `mteb.get_model_meta(...)` resolves;
`ruff`/`pyproject` clean.

### Env gotchas hit (already encoded in the `wave` extra)

- `torchaudio`/`torchcodec` must match torch 2.6 (`2.6.0` / `0.3.0`); the loose `audio`/`video`
  extras otherwise pull torch-2.11-built wheels (`libcudart.so.13` / `undefined symbol`).
- `sentence-transformers>=5` hard-imports `torchcodec` (built for a newer torch ABI) → pin `<5`.
- `triton` needs `setuptools` present. `module load ffmpeg/6.0.1` for the decode backends.
- BEATs ships as `BEATs_iter3_plus_AS2M.pt` (WAVE's "iter3_plus"); config is read from its `cfg` key.

## Open assumptions (still worth confirming for video/AV)

- **Video frame layout** — MTEB's collator yields torchcodec frames `(F, C, H, W)`; WAVE's decord
  path expects `(F, H, W, C)`. The wrapper permutes accordingly; the audio pilot didn't exercise
  this, so confirm on a video task (`MSVDT2VRetrieval`) against WAVE's own numbers.
- **Prompts** — defaults to WAVE's per-modality "Please describe the {modality}." prompts; an
  explicit task `prompt` overrides. Prompt-awareness means prompt choice affects scores.
