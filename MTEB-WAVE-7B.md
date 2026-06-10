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

## How to run a pilot (GPU required)

WAVE's deps conflict with the main MTEB env (the `transformers==4.51.3` pin), so install them in
an **isolated** venv with the `wave` extra. On the HLTCOE grid, request a GPU (≈24 GB+) via
`sbatch`/`srun`; flash-attn install may need the `flash-attn-install` skill.

```bash
# isolated env with WAVE deps
uv venv .venv-wave && source .venv-wave/bin/activate
uv pip install -e ".[wave,audio,video]"
git submodule update --init --recursive external/WAVE
export WAVE_BEATS_PATH=/path/to/BEATs_iter3_plus.pt
export HF_HOME="$PWD/.hf-wave-pilot"
```

```python
import mteb

model = mteb.get_model("tsinghua-ee/WAVE-7B")            # uses WAVE_BEATS_PATH
tasks = mteb.get_tasks(tasks=["ClothoT2ARetrieval"])      # small audio↔text pilot
results = mteb.evaluate(model, tasks=tasks, encode_kwargs={"batch_size": 1})
```

Record GPU name/memory, batch size, wall time, the per-task `evaluation_time`, and cache growth
(`du -sh "$HF_HOME" ~/.cache/mteb results`). Suggested progression: `ClothoT2ARetrieval` (audio) →
`MSVDT2VRetrieval` (video) → `AudioCapsAVT2VRetrieval` (audio-visual).

## Verified in this environment (no GPU)

- `wave_models` imports without WAVE deps (heavy imports are deferred to `__init__`).
- `mteb.get_model_meta("tsinghua-ee/WAVE-7B")` resolves; `embed_dim=3584`, modalities correct.
- `ruff check` / `ruff format` clean; `pyproject.toml` parses; `wave` extra + conflict registered.
- Candidate eval tasks resolve.

## Open risks / assumptions (validate during the GPU pilot)

- **Not run end-to-end here** — no GPU on the node and WAVE deps are intentionally isolated from
  the main env. The embedding recipe is reproduced from WAVE's source but unverified at runtime.
- **Video frame layout** — MTEB's collator yields torchcodec frames `(F, C, H, W)`; WAVE's decord
  path expects `(F, H, W, C)`. The wrapper permutes accordingly; confirm against WAVE's own output.
- **Prompts** — defaults to WAVE's per-modality "Please describe the {modality}." prompts; an
  explicit task `prompt` overrides. Prompt-awareness means prompt choice affects scores.
- **BEATs checkpoint** — must match the one WAVE trained with (`BEATs_iter3_plus.pt`).
