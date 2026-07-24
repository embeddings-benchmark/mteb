#!/usr/bin/env python3
"""Numerical faithfulness parity harness for the MTEB WAVE-7B wrapper.

Proves that ``mteb.models.model_implementations.wave_models.Wave7BWrapper`` reproduces
the *upstream* WAVE reference embedding path on identical inputs, per modality. The
model is loaded ONCE via ``Wave7BWrapper`` and ``wrapper.model`` / ``wrapper.processor``
are reused for both sides, so the comparison isolates *preprocessing / input
construction*, not weight loading.

Reference (upstream WAVE) path
------------------------------
A one-entry dataset in WAVE's own JSON format is fed through
``qwenvl.data.data_qwen.LazySupervisedDataset._get_item`` + the eval ``collate_fn`` +
``model(**inputs, pred_embeds=True)`` (mirroring ``qwenvl/train/train_qwen.py`` run_test
loop). A single retrieval item yields BOTH ``mllm_embeds`` (the media/query side, via
the all-layer head) and ``text_embeds`` (the label side, final-layer last token).

Wrapper path
------------
The wrapper's own building blocks, which is exactly what ``Wave7BWrapper.encode`` calls
per item:
  * text  -> ``wrapper._encode_text(s)``
  * media -> ``wrapper._build_inputs(...)`` -> ``wrapper._to_model_kwargs(...)`` ->
             ``wrapper.model(**kwargs).mllm_embeds``

Two levels per *media* modality
-------------------------------
  * Level 1 (construction parity): the wrapper is fed the SAME frames the reference
    used (WAVE's ``np.linspace`` sampling, reproduced with decord). This isolates input
    construction from frame selection and is a HARD gate -- it proves the wrapper's
    ``_build_inputs`` / ``_process_video`` / ``_to_model_kwargs`` are byte-faithful.
  * Level 2 (pipeline parity): the wrapper is fed frames produced by the REAL MTEB
    collator (``_DurationVideoCollator`` -> stride sampling), exactly as
    ``Wave7BWrapper.encode`` does in production. This is a DIAGNOSTIC, not a gate: MTEB
    deliberately keeps its default (stride) frame sampler, which selects different
    frames than WAVE's ``linspace`` for longer clips. The harness prints both frame-index
    arrays so the gap is explained, not hidden.

Text and audio have no frame sampling, so Level 1 == Level 2 (a single hard gate).

Media are synthesized deterministically with ffmpeg + soundfile (no network, no large
downloads), so the harness is a committable, rerunnable regression guard. Faithfulness
is about wrapper-vs-reference agreement on the *same* input, for which synthetic media
is sufficient.

Run (HLTCOE A100; env from scripts/setup_wave_env.sh):
    export WAVE_BEATS_PATH=$WORK/beats/BEATs_iter3_plus_AS2M.pt
    python scripts/validate_wave_faithfulness.py

Exit code is 0 iff every HARD gate passes.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

logger = logging.getLogger("validate_wave_faithfulness")

_SR = 16000


# --------------------------------------------------------------------------------------
# Deterministic synthetic media
# --------------------------------------------------------------------------------------
def synth_audio(path: Path, *, seconds: float = 3.0, sr: int = _SR, seed: int = 0) -> None:
    """Write a deterministic 16 kHz mono wav (tones + light noise)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, seconds, int(seconds * sr), endpoint=False)
    sig = (
        0.3 * np.sin(2 * np.pi * 220.0 * t)
        + 0.2 * np.sin(2 * np.pi * 440.0 * t)
        + 0.05 * rng.standard_normal(t.shape)
    )
    sf.write(str(path), sig.astype(np.float32), sr)


def synth_video(path: Path, *, seconds: float, fps: float, size: int = 64) -> None:
    """Encode a deterministic mp4 by piping raw RGB frames to ffmpeg (libx264).

    Each frame has a distinct spatial gradient + per-frame tint so the clip has genuine
    temporal variation.
    """
    n = max(1, int(round(seconds * fps)))
    w = h = size
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps}",
        "-i", "-",
        "-pix_fmt", "yuv420p", "-c:v", "libx264", str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    xs = np.linspace(0, 255, w, dtype=np.uint8)
    ys = np.arange(h, dtype=np.int64)
    for i in range(n):
        frame = np.empty((h, w, 3), dtype=np.uint8)
        frame[:, :, 0] = xs[None, :]
        frame[:, :, 1] = ((ys[:, None] + i * 7) % 256).astype(np.uint8)
        frame[:, :, 2] = np.uint8((i * 13) % 256)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed encoding {path}")


# --------------------------------------------------------------------------------------
# Frame sampling (reference linspace vs MTEB-default stride)
# --------------------------------------------------------------------------------------
def linspace_frames(path: Path, wrapper) -> tuple[torch.Tensor, float, np.ndarray, int]:
    """Reproduce WAVE ``video_decord`` frame selection (np.linspace) with decord.

    Returns (frames[F,H,W,C] uint8, video_length_s, selected_indices, n_source).
    """
    import decord  # noqa: PLC0415 (deferred; provided by the wave env)

    vr = decord.VideoReader(str(path), num_threads=1)
    n_source = len(vr)
    video_length = n_source / vr.get_avg_fps()
    interval = 1.0 / wrapper.fps  # WAVE base_interval (0.5 == 2 fps)
    target = min(max(round(video_length / interval), 1), wrapper.max_frames)
    idx = np.linspace(0, n_source - 1, target, dtype=int)
    frames = torch.from_numpy(vr.get_batch(idx).asnumpy())  # (F, H, W, C)
    return frames, video_length, idx, n_source


def stride_frames(path: Path, wrapper) -> tuple[torch.Tensor, float, np.ndarray, int]:
    """Reproduce MTEB's default collator frame selection, exactly as ``encode`` does.

    Returns (frames[F,C,H,W], duration_s, selected_indices, n_source).
    """
    from torchcodec.decoders import VideoDecoder  # noqa: PLC0415

    from mteb.models.model_implementations.wave_models import _DurationVideoCollator

    vd = VideoDecoder(str(path))
    collator = _DurationVideoCollator(
        target_sampling_rate=_SR, fps=wrapper.fps, max_frames=wrapper.max_frames
    )
    batch = collator([{"video": vd}])
    frames = batch["video"][0]
    duration = batch["video_duration"][0]
    # Re-derive the indices the collator used, for the diagnostic print.
    n_source = vd.metadata.num_frames
    target = min(max(1, int((duration or 0.0) * wrapper.fps)), wrapper.max_frames)
    step = max(1, n_source // max(1, target))
    idx = np.array(list(range(0, n_source, step))[:target])
    return frames, float(duration), idx, n_source


# --------------------------------------------------------------------------------------
# Upstream WAVE reference embeddings
# --------------------------------------------------------------------------------------
def _make_wave_dataset(wrapper, item: dict):
    """Build a one-entry ``LazySupervisedDataset`` mirroring the wrapper's data_args."""
    from qwenvl.data.data_qwen import LazySupervisedDataset  # noqa: PLC0415
    from qwenvl.train.argument import DataArguments  # noqa: PLC0415

    fp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump([item], fp)
    fp.flush()
    fp.close()

    da = DataArguments()
    da.dataset_use = fp.name
    da.omni_processor = wrapper.processor
    da.video_max_frames = wrapper.max_frames
    da.video_min_frames = 1
    da.base_interval = 1.0 / wrapper.fps
    da.use_beats = True
    da.beats_only = False
    da.train_classify = True
    da.run_test = False
    return LazySupervisedDataset(tokenizer=wrapper.processor.tokenizer, data_args=da)


@torch.no_grad()
def wave_reference(wrapper, item: dict) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Run the upstream WAVE reference path; return (mllm_embeds, text_embeds), normalized.

    Uses ``_get_item`` + WAVE's canonical ``DataCollatorForOmniDataset`` (the same collator
    WAVE trains with). That collator is what sets ``use_audio_in_video`` for synchronized AV
    (line 707: ``video_grid_thw is not None and audios is not None``), matching the wrapper.
    WAVE's lightweight *eval* ``collate_fn`` omits that flag, so it is NOT a fair AV reference.
    """
    from qwenvl.data.data_qwen import DataCollatorForOmniDataset  # noqa: PLC0415

    ds = _make_wave_dataset(wrapper, item)
    data = ds._get_item(0)
    if data is None:
        raise RuntimeError(f"WAVE _get_item returned None for item: {item}")
    batch = DataCollatorForOmniDataset()([data])

    device = wrapper.device
    inputs: dict = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
            inputs[k] = [it.to(device) for it in v]
        elif v is None:
            continue
        else:  # bools (use_audio_in_video), the `types` list, etc.
            inputs[k] = v
    inputs["pred_embeds"] = True

    out = wrapper.model(**inputs)
    return _norm(out.mllm_embeds), _norm(out.text_embeds)


# --------------------------------------------------------------------------------------
# Wrapper embeddings
# --------------------------------------------------------------------------------------
@torch.no_grad()
def wrapper_text(wrapper, s: str) -> torch.Tensor:
    return _norm(wrapper._encode_text(s))


@torch.no_grad()
def wrapper_media(
    wrapper, *, text=None, audio=None, video=None, video_duration=None
) -> torch.Tensor:
    dd = wrapper._build_inputs(
        text=text,
        image=None,
        audio=audio,
        video=video,
        instruction=None,
        video_duration=video_duration,
    )
    kw = wrapper._to_model_kwargs(dd)
    return _norm(wrapper.model(**kw).mllm_embeds)


# --------------------------------------------------------------------------------------
# Comparison bookkeeping
# --------------------------------------------------------------------------------------
def _norm(x: torch.Tensor | None) -> torch.Tensor | None:
    if x is None:
        return None
    return F.normalize(x.float(), p=2, dim=-1)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a, b, dim=-1).mean().item()


def _maxabs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


@dataclass
class Row:
    modality: str
    level: str
    cosine: float
    max_abs: float
    hard: bool
    threshold: float
    note: str = ""
    passed: bool = field(init=False)

    def __post_init__(self) -> None:
        self.passed = (not self.hard) or (self.cosine >= self.threshold)


def compare(rows, modality, level, ref, got, *, hard, threshold, note=""):
    rows.append(
        Row(
            modality=modality,
            level=level,
            cosine=_cos(ref, got),
            max_abs=_maxabs(ref, got),
            hard=hard,
            threshold=threshold,
            note=note,
        )
    )


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-name", default="tsinghua-ee/WAVE-7B")
    p.add_argument("--revision", default="6d42651d34bf1a7d83d5779397d6ce0316a4cf4f")
    p.add_argument("--beats-path", default=os.environ.get("WAVE_BEATS_PATH"))
    p.add_argument("--work-dir", default=None, help="Where to write synthetic media (default: a tempdir).")
    p.add_argument("--device", default=None)
    p.add_argument("--cos-hard", type=float, default=0.999, help="Cosine gate for text/video Level-1.")
    p.add_argument("--cos-audio", type=float, default=0.999, help="Cosine gate for audio.")
    p.add_argument("--cos-soft", type=float, default=0.99, help="Reporting threshold for diagnostics.")
    return p


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_arg_parser().parse_args()

    if not args.beats_path:
        logger.error("BEATs checkpoint required: pass --beats-path or set WAVE_BEATS_PATH.")
        return 2
    if not torch.cuda.is_available():
        logger.warning("CUDA not available; WAVE-7B is bf16 and effectively needs an A100/L40S/H100.")

    work_dir = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="wave_parity_"))
    work_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Work dir: %s", work_dir)

    # --- synth media -------------------------------------------------------------------
    audio_wav = work_dir / "audio.wav"
    vid_short = work_dir / "video_short.mp4"
    vid_mid = work_dir / "video_mid.mp4"
    vid_cap = work_dir / "video_cap.mp4"
    logger.info("Synthesizing media ...")
    synth_audio(audio_wav, seconds=3.0)
    synth_video(vid_short, seconds=3.0, fps=2.0)   # frames <= target -> samplers converge
    synth_video(vid_mid, seconds=20.0, fps=10.0)   # diverge, below 128 cap
    synth_video(vid_cap, seconds=70.0, fps=10.0)   # diverge + 128-frame cap binds

    # --- load model once ---------------------------------------------------------------
    from mteb.models.model_implementations.wave_models import Wave7BWrapper  # noqa: PLC0415

    logger.info("Loading WAVE-7B (once) ...")
    wrapper = Wave7BWrapper(
        model_name=args.model_name,
        revision=args.revision,
        device=args.device,
        beats_path=args.beats_path,
    )

    rows: list[Row] = []

    def human(media_tag: str) -> str:
        prompt = {"video": "Please describe the video.", "audio": "Please describe the audio."}[media_tag]
        return f"<{media_tag}>\n{prompt}"

    def conv(human_value: str, gpt_value: str) -> list[dict]:
        return [{"from": "human", "value": human_value}, {"from": "gpt", "value": gpt_value}]

    # --- TEXT --------------------------------------------------------------------------
    logger.info("[text] ...")
    for s in ("A dog barking in a quiet room.", "Orchestral music with strings."):
        item = {"conversations": conv("placeholder query", s), "type": "retrieval"}
        _, ref_text = wave_reference(wrapper, item)
        got = wrapper_text(wrapper, s)
        compare(rows, "text", "L1==L2", ref_text, got, hard=True, threshold=args.cos_hard,
                note=s[:24])

    # --- AUDIO -------------------------------------------------------------------------
    logger.info("[audio] ...")
    arr, sr = sf.read(str(audio_wav))
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    arr = arr.astype(np.float32)
    assert sr == _SR
    a_item = {"conversations": conv(human("audio"), ""), "audio": str(audio_wav), "type": "retrieval"}
    ref_audio, _ = wave_reference(wrapper, a_item)
    got = wrapper_media(wrapper, audio=arr)
    compare(rows, "audio", "L1==L2", ref_audio, got, hard=True, threshold=args.cos_audio)

    # --- VIDEO -------------------------------------------------------------------------
    for tag, path in (("video-short", vid_short), ("video-mid", vid_mid), ("video-cap", vid_cap)):
        logger.info("[%s] ...", tag)
        v_item = {"conversations": conv(human("video"), ""), "video": str(path), "type": "retrieval"}
        ref_v, _ = wave_reference(wrapper, v_item)

        f1, dur1, idx1, n1 = linspace_frames(path, wrapper)
        got1 = wrapper_media(wrapper, video=f1, video_duration=dur1)
        compare(rows, tag, "L1(linspace)", ref_v, got1, hard=True, threshold=args.cos_hard,
                note=f"N={n1} T={len(idx1)} idx[:3]={idx1[:3].tolist()} idx[-1]={int(idx1[-1])}")

        f2, dur2, idx2, n2 = stride_frames(path, wrapper)
        got2 = wrapper_media(wrapper, video=f2, video_duration=dur2)
        compare(rows, tag, "L2(stride)", ref_v, got2, hard=False, threshold=args.cos_soft,
                note=f"N={n2} T={len(idx2)} idx[:3]={idx2[:3].tolist()} idx[-1]={int(idx2[-1])}")

    # --- AUDIO-VISUAL ------------------------------------------------------------------
    logger.info("[av] ...")
    av_item = {
        "conversations": conv(human("video"), ""),
        "video": str(vid_mid),
        "audio": str(audio_wav),
        "type": "retrieval",
    }
    ref_av, _ = wave_reference(wrapper, av_item)
    f1, dur1, idx1, n1 = linspace_frames(vid_mid, wrapper)
    got1 = wrapper_media(wrapper, audio=arr, video=f1, video_duration=dur1)
    compare(rows, "av", "L1(linspace)", ref_av, got1, hard=True, threshold=args.cos_hard)
    f2, dur2, idx2, n2 = stride_frames(vid_mid, wrapper)
    got2 = wrapper_media(wrapper, audio=arr, video=f2, video_duration=dur2)
    compare(rows, "av", "L2(stride)", ref_av, got2, hard=False, threshold=args.cos_soft)

    # --- report ------------------------------------------------------------------------
    print("\n" + "=" * 92)
    print("WAVE-7B faithfulness parity (wrapper vs upstream WAVE reference)")
    print("=" * 92)
    print(f"{'modality':<12}{'level':<14}{'cosine':>10}{'max_abs':>11}{'gate':>7}{'result':>9}  note")
    print("-" * 92)
    for r in rows:
        gate = "HARD" if r.hard else "diag"
        result = "PASS" if r.passed else ("FAIL" if r.hard else "----")
        print(
            f"{r.modality:<12}{r.level:<14}{r.cosine:>10.5f}{r.max_abs:>11.2e}"
            f"{gate:>7}{result:>9}  {r.note}"
        )
    print("=" * 92)

    hard_fail = [r for r in rows if r.hard and not r.passed]
    if hard_fail:
        print(f"FAILED: {len(hard_fail)} hard gate(s) below threshold.")
        return 1
    print("All hard gates passed. (Video/AV L2 deltas are the expected effect of MTEB's "
          "default stride frame sampler -- see frame indices above.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
