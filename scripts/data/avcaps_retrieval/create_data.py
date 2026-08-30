#!/usr/bin/env python3
"""Build the AVCaps audio-visual retrieval tasks for MTEB.

AVCaps (derived from VidOR) captions each clip three separate ways - from the audio
alone, from the visuals alone, and from both together. That is what makes it useful
here: the audio-only, video-only and combined directions can be scored independently on
identical clips, instead of being inferred from one caption set that mixes the modalities.

Only the official test split is used, so the evaluation set matches the split the
authors held out; train and validation are left untouched.

The published clips are muxed mp4s. To keep an audio-caption task from being solvable
off the video track, the audio is demuxed to a separate mono 16 kHz stream and each task
exposes only the media columns its captions actually describe.

Examples:
  # Build the dataset locally from the downloaded archive.
  uv run python scripts/data/avcaps_retrieval/create_data.py --stage build

  # Build and publish.
  uv run python scripts/data/avcaps_retrieval/create_data.py --stage all --push
"""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path

import av
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Audio, Dataset, Video
from huggingface_hub import HfApi, hf_hub_download

_SOURCE_REPO = "TUT-ARG/AVCaps"
_PAPER = "https://ieeexplore.ieee.org/document/11029114/"
_TARGET_REPO = "vnahata/avcaps-retrieval"
_LICENSE = "cc-by-nc-4.0"
_SPLIT = "test"

# caption field in the source -> config name published here
_CAPTION_FIELDS = {
    "audio_captions": "audio_captions",
    "visual_captions": "visual_captions",
    "audio_visual_captions": "av_captions",
}


def _extract_audio(mp4: Path, wav: Path) -> bool:
    """Demux the audio track to mono 16 kHz wav; False when the clip has none."""
    try:
        with av.open(str(mp4)) as inp:
            if not inp.streams.audio:
                return False
            stream = inp.streams.audio[0]
            with av.open(str(wav), "w") as out:
                out_stream = out.add_stream("pcm_s16le", rate=16000, layout="mono")
                resampler = av.audio.resampler.AudioResampler(
                    format="s16", layout="mono", rate=16000
                )
                for frame in inp.decode(stream):
                    for resampled in resampler.resample(frame):
                        out.mux(out_stream.encode(resampled))
                out.mux(out_stream.encode(None))
        return wav.exists() and wav.stat().st_size > 44  # larger than a bare wav header
    except Exception as e:  # noqa: BLE001
        print(f"  audio demux failed for {mp4.name}: {type(e).__name__} {str(e)[:70]}")
        return False


def stage_build(work: Path) -> dict[str, int]:
    """Extract the archive, demux audio, and write media + caption tables."""
    media_dir = work / "clips"
    out_dir = work / "dataset"
    media_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    caps_path = hf_hub_download(_SOURCE_REPO, "test_captions.json", repo_type="dataset")
    caps = json.loads(Path(caps_path).read_text(encoding="utf-8"))
    zip_path = hf_hub_download(_SOURCE_REPO, "test_videos.zip", repo_type="dataset")

    if not any(media_dir.glob("*.mp4")):
        with zipfile.ZipFile(zip_path) as z:
            for name in z.namelist():
                if name.lower().endswith(".mp4"):
                    with (
                        z.open(name) as src,
                        open(media_dir / os.path.basename(name), "wb") as dst,
                    ):
                        dst.write(src.read())

    media, skipped = [], 0
    for clip_id in sorted(caps):
        mp4 = media_dir / f"{clip_id}.mp4"
        wav = media_dir / f"{clip_id}.wav"
        if not mp4.exists() or (not wav.exists() and not _extract_audio(mp4, wav)):
            skipped += 1
            continue
        media.append(
            {
                "id": clip_id,
                "video": {"bytes": mp4.read_bytes(), "path": mp4.name},
                "audio": {"bytes": wav.read_bytes(), "path": wav.name},
            }
        )
    pq.write_table(pa.Table.from_pylist(media), out_dir / "media.parquet")
    print(f"media: {len(media)} clips ({skipped} skipped)")

    have = {m["id"] for m in media}
    stats = {"media": len(media)}
    for field, cfg in _CAPTION_FIELDS.items():
        # Drop repeated caption text. Annotators sometimes produce the same short
        # sentence for different clips ("A man is speaking"), which would make the
        # qrels ambiguous: the identical query text is marked relevant to only one
        # of the clips it actually describes.
        rows, seen = [], set()
        for cid in sorted(have):
            for i, text in enumerate(caps[cid].get(field) or []):
                text = (text or "").strip()
                if not text or text in seen:
                    continue
                seen.add(text)
                rows.append({"id": f"{cid}-{cfg}-{i}", "text": text, "media_id": cid})
        pq.write_table(pa.Table.from_pylist(rows), out_dir / f"{cfg}.parquet")
        stats[cfg] = len(rows)
        print(f"{cfg}: {len(rows)} captions")

    (work / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def stage_push(work: Path) -> None:
    out_dir = work / "dataset"
    api = HfApi()
    api.create_repo(_TARGET_REPO, repo_type="dataset", exist_ok=True)

    media = (
        Dataset.from_parquet(str(out_dir / "media.parquet"))
        .cast_column("video", Video())
        .cast_column("audio", Audio(sampling_rate=16000))
    )
    media.push_to_hub(
        _TARGET_REPO, config_name="media", split=_SPLIT, max_shard_size="400MB"
    )

    for cfg in _CAPTION_FIELDS.values():
        ds = Dataset.from_parquet(str(out_dir / f"{cfg}.parquet"))
        ds.push_to_hub(_TARGET_REPO, config_name=cfg, split=_SPLIT)
        print(f"  pushed {cfg}: {len(ds)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["build", "push", "all"], default="all")
    parser.add_argument("--work-dir", type=Path, default=Path("avcaps_work"))
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    print(f"source: {_PAPER} (license {_LICENSE})")

    if args.stage in ("build", "all"):
        stage_build(args.work_dir)
    if args.stage == "push" or (args.push and args.stage == "all"):
        stage_push(args.work_dir)


if __name__ == "__main__":
    main()
