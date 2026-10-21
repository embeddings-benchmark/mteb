"""Create VATEX v2: cut each clip of mteb/VATEX_test_1k to its official 10 s window.

The v1 files are the full YouTube videos, while each VATEX caption describes one ~10 s clip.
The official test annotations name every clip "<youtube id>_<start>_<end>"; each v1 caption
matches exactly one of them, which gives its window.
https://eric-xw.github.io/vatex-website/download.html

- Files longer than one window are cut to [start, end] (or to the end of the file if the
  source stops early); shorter files already are the clip and are kept as is.
- Rows whose audio does not cover the clip (it stops before the window) or is silent
  throughout are dropped.
- `video_id`, `start` and `end` columns record each row's official clip.

usage: HF_TOKEN=... python create_data.py OUT_DIR [--push]
"""

import io
import json
import subprocess
import sys
import tempfile
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import soundfile as sf
from datasets import Audio, Dataset, Value, Video, concatenate_datasets, load_dataset

REPO, REVISION = "mteb/VATEX_test_1k", "0d2e86e6d36927f4676ee6127c4e38e3867ce0ce"
ANNOTATIONS = (
    "https://eric-xw.github.io/vatex-website/data/vatex_public_test_english_v1.1.json"
)
WINDOW = 10


def duration(path: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "csv=p=0",
        path,
    ]
    return float(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)


def has_sound(audio: bytes) -> bool:
    return bool(abs(sf.read(io.BytesIO(audio), dtype="float32")[0]).max() > 0)


def cut_clip(row: tuple[bytes, bytes, int, int]) -> tuple[bytes, bytes, float, float]:
    """Return (video, audio, video seconds, audio seconds) for one row."""
    video, audio, start, end = row
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = f"{tmp}/in.mp4", f"{tmp}/out.mp4"
        Path(src).write_bytes(video)
        length = duration(src)
        if length <= WINDOW + 0.5:  # already the clip
            return video, audio, length, sf.info(io.BytesIO(audio)).duration
        stop = min(end, length)
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-ss",
                str(start),
                "-to",
                str(stop),
                "-i",
                src,
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-movflags",
                "+faststart",
                dst,
            ],
            check=True,
        )
        video = Path(dst).read_bytes()
        length = duration(dst)
    samples, sr = sf.read(io.BytesIO(audio), dtype="float32")
    buffer = io.BytesIO()
    sf.write(buffer, samples[int(start * sr) : int(stop * sr)], sr, format="WAV")
    return (
        video,
        buffer.getvalue(),
        length,
        len(samples[int(start * sr) : int(stop * sr)]) / sr,
    )


def main(out_dir: Path, push: bool) -> None:
    clip_of = {}
    for clip in json.load(urllib.request.urlopen(ANNOTATIONS)):
        for caption in clip["enCap"]:
            clip_of[caption.strip().lower()] = clip["videoID"]

    ds = load_dataset(REPO, split="test", revision=REVISION)
    ds = ds.cast_column("video", Video(decode=False)).cast_column(
        "audio", Audio(decode=False)
    )
    video_ids = [clip_of[c.strip().lower()] for c in ds["caption"]]
    starts = [int(v[-13:-7]) for v in video_ids]
    ends = [int(v[-6:]) for v in video_ids]

    rows = [
        (ds[i]["video"]["bytes"], ds[i]["audio"]["bytes"], starts[i], ends[i])
        for i in range(len(ds))
    ]
    with ProcessPoolExecutor(8) as pool:
        results = list(pool.map(cut_clip, rows, chunksize=4))

    keep = [
        i
        for i, (_, audio, v, a) in enumerate(results)
        if abs(v - a) <= 0.5 and has_sound(audio)
    ]
    print(
        f"kept {len(keep)} of {len(ds)}; dropped (audio missing or silent): {[video_ids[i] for i in range(len(ds)) if i not in keep]}"
    )

    shards = []
    for chunk in [
        keep[k : k + 50] for k in range(0, len(keep), 50)
    ]:  # small shards: one Arrow column can't hold 2 GB
        shards.append(
            Dataset.from_dict(
                {
                    "video": [{"bytes": results[i][0], "path": None} for i in chunk],
                    "audio": [{"bytes": results[i][1], "path": None} for i in chunk],
                    "caption": [ds[i]["caption"] for i in chunk],
                    "video_id": [video_ids[i] for i in chunk],
                    "start": [starts[i] for i in chunk],
                    "end": [ends[i] for i in chunk],
                }
            )
            .cast_column("video", Video())
            .cast_column("audio", Audio(sampling_rate=16000))
            .cast_column("start", Value("int32"))
            .cast_column("end", Value("int32"))
        )
    out = concatenate_datasets(shards)
    out.save_to_disk(str(out_dir))
    if push:
        out.push_to_hub(
            REPO,
            split="test",
            commit_message="VATEX v2: cut each clip to its official 10 s window",
        )


if __name__ == "__main__":
    main(Path(sys.argv[1]), "--push" in sys.argv)
