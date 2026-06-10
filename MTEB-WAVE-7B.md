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

## Verified — pilot runs on the grid ✅

All runs on **A100-PCIE-40GB** (`flash_attn 2.7.4.post1`), `exceptions=[]`:

| task | modality | metric | score | eval time |
| :-- | :-- | :-- | :-- | :-- |
| `ClothoT2ARetrieval` | text→audio (1045 docs) | hit_rate@5 | **0.42** | ~450 s |
| `MSVDT2VRetrieval` | text→video (660 videos) | ndcg@10 | **0.80** (R@1 0.63, R@5 0.90) | ~413 s |

- Load log shows `Init BEATs Model` + `Classify Type: all_layer` — BEATs dual encoder and
  all-layer fusion active. Cold model load ≈ 219 s (≈ 9 s warm).
- The MSVD score is in the SOTA band for that dataset, validating the video decode → frame
  permute → visual tower path end-to-end (a wrong frame layout would score near-random).
- Clotho 0.42 is identical on torch 2.6.0/datasets 3.6 and torch 2.7.1/datasets 4.0 stacks.

### Faithfulness audit findings (fixed)

1. **Text inputs must use WAVE's label path.** In WAVE's forward, only *media* goes through
   all-layer fusion + `classify_linear`; *text* (labels/candidates) is embedded as bare
   ``text + <|im_end|>`` → last token of the FINAL layer, **no** chat template, **no** head
   (`label_ids` and `all_ids` branches). The wrapper originally sent text through the media
   branch; fixing this raised Clotho hit_rate@5 **0.32 → 0.42**.
2. **Synchronized audio-visual wiring**: `use_audio_in_video` + `seconds_per_chunk` now mirror
   `data_qwen._get_item` exactly (audio tokens interleaved inside the `<video>` expansion), and
   the flag is forwarded to the model call. Not yet exercised on an AV task.
3. **Checkpoint-load warnings are benign**: `beats.encoder.pos_conv` parametrization names
   mismatch on `from_pretrained`, then the separate BEATs `load_state_dict` (strict) repairs
   them — identical to WAVE's own eval flow. `beats.predictor.*` unused is expected.
4. **Metadata corrections**: `n_parameters=9_410_651_007` (from the checkpoint index; the 7B
   name undercounts — BEATs + head included), `memory_usage_mb=17949`, citation fixed to the
   real author list (Changli Tang et al.).

### Env gotchas hit (already encoded in the `wave` extra)

- **torch 2.7.1 stack required for video**: `datasets>=4` (Video → torchcodec `VideoDecoder`,
  what MTEB's `VideoCollator` expects) needs `torchcodec>=0.4` ⇒ torch 2.7.1. datasets 3.x
  decodes video via torchvision+PyAV instead and breaks MTEB's collator; datasets 4.8+ needs
  torch≥2.8; datasets 5 changed config discovery. Upstream WAVE pins torch 2.6.0 (training);
  inference on 2.7.1 reproduces identical scores (verified on Clotho).
- `sentence-transformers>=5` hard-imports `torchcodec` at module top → pin `<5`.
- `triton` needs `setuptools`. `module load ffmpeg/6.0.1` for torchcodec's libavutil.
- flash-attn: use the prebuilt wheel matching py/torch/cuda/ABI (torch 2.7 wheels are
  `cxx11abiTRUE`; torch 2.6 wheels were `abiFALSE`).
- BEATs ships as `BEATs_iter3_plus_AS2M.pt` (WAVE's "iter3_plus"); config read from its `cfg` key.
- After editing the `wave` extra, re-run `uv pip install -e . --no-deps` — `ModelMeta`'s
  requirement check reads the *installed* metadata, not the live pyproject.

## Remaining work (honest list)

- **AV-joint validation**: run an audio-visual task (e.g. `AudioCapsAVVA2TRetrieval`) to exercise
  `use_audio_in_video=True` end-to-end.
- **Benchmarks**: `MVEB(beta)` (23 AV tasks) and `MAEB(beta)` exist in-repo — running WAVE on them
  is compute only, no code. Leaderboard listing would additionally need a results-repo submission.
- **Paper-parity datasets (optional)**: WAVE's moment-retrieval evals (Charades-STA, QVHighlights,
  MomentSeeker) are not in MTEB and correspond to WAVE's unported `seg_video` branch (left
  half-finished upstream — it contains a literal `breakpoint()`). Porting would mean a new task
  subtype; only needed to reproduce that slice of the paper. The MMEB-v2-video-style retrieval
  members (MSRVTT, MSVD, DiDeMo, VATEX, YouCook2, VALOR32K) are already present.
- **Prompt parity for paper numbers**: WAVE's task-specific eval prompts (their `ret_*.json`)
  should be matched per-task when comparing to published scores; the wrapper currently uses an
  item's own text, else the task prompt, else WAVE's "Please describe the {modality}." default.
- **Upstreaming**: the `external/WAVE` submodule + `sys.path` import is fine for this fork; an
  upstream `embeddings-benchmark/mteb` PR would likely want the qwenvl code installable (e.g.
  `wave @ git+...`) instead of a submodule, and CI can't run the model (needs GPU + BEATs ckpt).
