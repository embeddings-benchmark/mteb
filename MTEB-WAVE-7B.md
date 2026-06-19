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

## Setting up on any (internet-connected) cluster or login node

One script reconstructs everything — venv, the matching flash-attn wheel, the BEATs checkpoint,
and optional prefetches (proven by a clean-room run on a fresh workspace: same Clotho score):

```bash
git clone -b wave-7b-integration https://github.com/debashishc/mteb.git && cd mteb
bash scripts/setup_wave_env.sh /path/to/fast/scratch/wave-mteb \
    [--prefetch-model] [--prefetch-data ClothoT2ARetrieval,MSVDT2VRetrieval]
```

It prints the exports for job scripts (`source .venv/bin/activate`, `HF_HOME`,
`WAVE_BEATS_PATH`). WAVE uses bf16 → request **A100/L40S/H100** (not V100); video tasks need
FFmpeg 4–7 libs at runtime (HLTCOE: `module load ffmpeg/6.0.1`).

### Artifact manifest (what lives where)

| Artifact | Size | Source | Obtained by |
| :-- | :-- | :-- | :-- |
| repo branch `wave-7b-integration` + submodule `external/WAVE` | ~0.1 GB | GitHub | `git clone` + script |
| venv (py3.10, torch 2.7.1 cu126 stack) | ~9.5 GB | PyPI | script (never copy venvs across arch/python) |
| WAVE-7B weights @ `6d42651d` | 18 GB | HF Hub `tsinghua-ee/WAVE-7B` | auto on first `get_model`, or `--prefetch-model` |
| BEATs `BEATs_iter3_plus_AS2M.pt` | 345 MB | HF dataset `Bencr/beats-checkpoints` | **script (NOT auto)** — `WAVE_BEATS_PATH` |
| flash-attn prebuilt wheel | 179 MB | Dao-AILab GitHub releases | script (auto-detects py/torch/cuda/ABI) |
| datasets (Clotho 2.1 GB, MSVD 0.6 GB, others as pulled; ~30 GB budget for the faithfulness set) | ~30 GB | HF Hub | auto on first run, or `--prefetch-data` |
| result JSONs | <1 MB | `~/.cache/mteb/results/` | produced by runs; rsync to continue elsewhere |

Disk budget on target scratch: **~60–80 GB**. Two shortcuts:
- **New login node, same grid**: nothing to do — `/expscratch` + `/home` are shared; just
  `source $WORK/.venv/bin/activate`.
- **Skip re-downloads on a new cluster**: rsync `$WORK/.cache/huggingface/hub/models--tsinghua-ee--WAVE-7B`
  (18 GB) and `datasets--mteb--*` dirs into the new `HF_HOME` (snapshots are content-addressed).

### Running an evaluation

```python
import mteb
model = mteb.get_model("tsinghua-ee/WAVE-7B")             # reads WAVE_BEATS_PATH
tasks = mteb.get_tasks(tasks=["ClothoT2ARetrieval"])       # small audio↔text pilot
results = mteb.evaluate(model, tasks=tasks, encode_kwargs={"batch_size": 1})
```

Progression: `ClothoT2ARetrieval` (audio) → `MSVDT2VRetrieval` (video) →
`AudioCapsAVVA2TRetrieval` (audio-visual). sbatch templates: `run_pilot.sbatch` in the
workspace (`--gres=gpu:a100:1 --cpus-per-task=8 --mem=64G`, modules cuda + ffmpeg). Note:
multi-task lists must be exported via the environment (`export WAVE_PILOT_TASKS=A,B,C` before
`sbatch`), not `--export=ALL,VAR=A,B,C` — sbatch splits `--export` on commas.

## Verified — pilot runs on the grid ✅

All runs on **A100-PCIE-40GB** (`flash_attn 2.7.4.post1`), `exceptions=[]`. Every modality path
of the wrapper has now been exercised end-to-end:

| task | path exercised | metric | score | eval time |
| :-- | :-- | :-- | :-- | :-- |
| `ClothoT2ARetrieval` | audio media-branch + text label-path | hit_rate@5 | **0.42** (R@1 0.19) | ~450 s |
| `MSVDT2VRetrieval` | video decode → permute → visual tower | ndcg@10 | **0.80** (R@1 0.63) | ~414 s |
| `MSRVTTT2V` | video (1k-candidate pool) | ndcg@10 | **0.68** (R@1 0.51) | ~456 s |
| `DiDeMoT2VRetrieval` | long videos (fps-fix path) | ndcg@10 | **0.71** (R@1 0.55) | ~2989 s |
| `AudioCapsAVVA2TRetrieval` | **synchronized AV** (`use_audio_in_video=True` interleave + BEATs doubling) | ndcg@10 | **0.54** (R@1 0.31) | ~656 s |

- Load log shows `Init BEATs Model` + `Classify Type: all_layer` — BEATs dual encoder and
  all-layer fusion active. Cold model load ≈ 219 s (≈ 9–70 s warm).
- Clotho 0.42 is identical on torch 2.6.0/datasets 3.6 and torch 2.7.1/datasets 4.0 stacks, and
  reproduces exactly from a clean-room environment built by `scripts/setup_wave_env.sh`.
- Regressions after the fps/audio fidelity fixes: Clotho 0.42 and MSVD 0.80 unchanged
  (the fixes are no-ops for short clips, as expected).

### Comparison with the WAVE paper (arXiv:2509.21990, R@1)

| dataset (t2v / t2a) | MTEB (ours) | paper (MMEB-v2 harness) | Δ |
| :-- | :-- | :-- | :-- |
| MSVD | **63.5** | 56.3 | +7.2 |
| MSR-VTT | **50.9** | 54.7 | −3.8 |
| DiDeMo | **54.8** | 69.3 | −14.5 |
| Clotho | **19.4** | 25.6 | −6.2 |

Read this as *faithful-in-kind, protocol-divergent*: the embedding recipe matches WAVE exactly
(verified at source level; deltas go in both directions), but MTEB task protocols differ from
the MMEB-v2 harness — candidate-set construction (e.g. MTEB MSVD's 660-pair pool), query
construction (MMEB DiDeMo uses paragraph-style queries vs MTEB's per-caption), and per-task
prompts. The same divergence applies to any model evaluated on both harnesses, so relative
comparisons within MTEB remain valid. (Paper also reports AudioCaps 44.2, VGGSound-AV 25.0 R@1
— `AudioCapsA2T/T2A` and `VGGSoundAV*` tasks exist in MTEB for follow-up.)

### Faithfulness audit findings (fixed)

1. **Text inputs must use WAVE's label path.** In WAVE's forward, only *media* goes through
   all-layer fusion + `classify_linear`; *text* (labels/candidates) is embedded as bare
   ``text + <|im_end|>`` → last token of the FINAL layer, **no** chat template, **no** head
   (`label_ids` and `all_ids` branches). The wrapper originally sent text through the media
   branch; fixing this raised Clotho hit_rate@5 **0.32 → 0.42**.
2. **Synchronized audio-visual wiring**: `use_audio_in_video` + `seconds_per_chunk` now mirror
   `data_qwen._get_item` exactly (audio tokens interleaved inside the `<video>` expansion), and
   the flag is forwarded to the model call. Validated on `AudioCapsAVVA2TRetrieval` (0.54).
3. **Video fps timing**: WAVE derives `video_second_per_grid` from the *actual* sampled rate
   (`fps = sampled_frames / duration`), which differs from nominal when the 128-frame cap binds
   or short videos keep all frames. The wrapper now records each video's duration in the
   collator (`_DurationVideoCollator`) and replicates this; falls back to nominal fps when
   duration metadata is missing.
4. **Audio >300 s**: WAVE never truncates — `process_audio` chunks long audio into 300 s
   segments. Truncation is now opt-in (`max_audio_length_seconds`, default off).
5. **Checkpoint-load warnings are benign**: `beats.encoder.pos_conv` parametrization names
   mismatch on `from_pretrained`, then the separate BEATs `load_state_dict` (strict) repairs
   them — identical to WAVE's own eval flow. `beats.predictor.*` unused is expected.
6. **Metadata corrections**: `n_parameters=9_410_651_007` (from the checkpoint index; the 7B
   name undercounts — BEATs + head included), `memory_usage_mb=17949`, citation fixed to the
   real author list (Changli Tang et al.).

### Numerical parity (single-input) — measured ✅

Findings #1–#3 were originally verified by source inspection; they are now also verified
**numerically** by `scripts/validate_wave_faithfulness.py` (run via
`scripts/validate_wave_faithfulness.sbatch`; ~2 min on one H200/A100). The harness loads
WAVE-7B once via `Wave7BWrapper` and, per modality, compares the wrapper's embedding to an
upstream-WAVE reference built from `qwenvl`'s own `_get_item` + `DataCollatorForOmniDataset`
(WAVE's *training* collator — it sets `use_audio_in_video` for synchronized AV, which WAVE's
lightweight eval `collate_fn` does not, so the eval collator would be an unfair AV reference).
Both sides reuse the same `wrapper.model`/`processor`, isolating *preprocessing*, not weight
loading. Media are synthesized deterministically (ffmpeg + soundfile), so the check is a
committable, network-free regression guard.

Two levels per media modality:
- **L1 (construction parity)** — wrapper fed the SAME frames WAVE selects (`np.linspace`),
  isolating input construction from frame selection. **Hard gate.**
- **L2 (pipeline parity)** — wrapper fed frames from the REAL MTEB collator
  (`_DurationVideoCollator` → stride sampling), exactly as `encode` runs in production.
  **Diagnostic only** — MTEB deliberately keeps its default (stride) frame sampler.

Measured (H200, bf16, elapsed 1:55):

| modality | level | cosine | max_abs | note |
| :-- | :-- | --: | --: | :-- |
| text (×2) | L1==L2 | **1.00000** | 0 | WAVE label path == `_encode_text` |
| audio | L1==L2 | **1.00000** | 0 | |
| video-short | L1 / L2 | **1.00000** / 1.00000 | 0 | samplers converge (N=T=6) |
| video-mid | L1 / L2 | **1.00000** / 0.99726 | 4.8e-3 | stride last frame 195 vs linspace 199 |
| video-cap | L1 / L2 | **1.00000** / 0.99735 | 4.8e-3 | 128-frame cap; stride last 635 vs 699 |
| av (sync) | L1 / L2 | **1.00000** / 0.99790 | 4.7e-3 | `use_audio_in_video` interleave + BEATs doubling |

**Every L1 gate is bit-identical** (cosine 1.00000, max-abs exactly 0) across text, audio,
video, and synchronized AV: the wrapper reproduces WAVE's forward / all-layer head / label
path / L2-normalization exactly. The only deltas are the L2 video/AV diagnostics (~0.997),
which are **by design** — MTEB's default stride sampler selects different frames than WAVE's
`linspace` on longer clips (drops the tail; off-by-one target count). We keep MTEB's default
sampler; this gap is documented, not patched.

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
