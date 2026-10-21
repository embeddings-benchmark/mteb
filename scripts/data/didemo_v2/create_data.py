"""Create DiDeMo v2: cut each video of mteb/DiDeMo to its first 30 s.

DiDeMo's descriptions only cover the first 30 s of each video: "Some videos are longer than
30 seconds. These videos were truncated to 30 seconds during annotation."
https://github.com/LisaAnne/LocalizingMoments#dataset
CLIP4Clip's paragraph retrieval likewise uses seconds 0-31.

- Videos longer than 30 s are cut to [0, 30]; shorter ones are kept as is.
- Rows whose audio does not cover the clip or is silent throughout are dropped.

usage: HF_TOKEN=... python create_data.py OUT_DIR [--push]
"""

import io
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import soundfile as sf
from datasets import Audio, Dataset, Video, concatenate_datasets, load_dataset

REPO, REVISION = "mteb/DiDeMo", "746689f644b66022540a9a39136e842bee164e6b"
LIMIT = 30


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


def cut_clip(row: tuple[bytes, bytes]) -> tuple[bytes, bytes, float, float]:
    """Return (video, audio, video seconds, audio seconds) for one row."""
    video, audio = row
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = f"{tmp}/in.mp4", f"{tmp}/out.mp4"
        Path(src).write_bytes(video)
        length = duration(src)
        if length <= LIMIT + 0.5:
            return video, audio, length, sf.info(io.BytesIO(audio)).duration
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-ss",
                "0",
                "-to",
                str(LIMIT),
                "-i",
                src,
                "-map",
                "0:v:0",
                "-map",
                "0:a:0?",
                "-vf",
                "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # libx264 needs even width and height
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
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
    sf.write(buffer, samples[: LIMIT * sr], sr, format="WAV")
    return video, buffer.getvalue(), length, len(samples[: LIMIT * sr]) / sr


def main(out_dir: Path, push: bool) -> None:
    ds = load_dataset(REPO, split="test", revision=REVISION)
    ds = ds.cast_column("video", Video(decode=False)).cast_column(
        "audio", Audio(decode=False)
    )
    rows = [(ds[i]["video"]["bytes"], ds[i]["audio"]["bytes"]) for i in range(len(ds))]
    with ProcessPoolExecutor(8) as pool:
        results = list(pool.map(cut_clip, rows, chunksize=4))

    keep = [
        i
        for i, (_, audio, v, a) in enumerate(results)
        if abs(v - a) <= 0.5 and has_sound(audio)
    ]
    print(
        f"kept {len(keep)} of {len(ds)}; dropped (audio missing or silent): {[i for i in range(len(ds)) if i not in keep]}"
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
                }
            )
            .cast_column("video", Video())
            .cast_column("audio", Audio(sampling_rate=16000))
        )
    out = concatenate_datasets(shards)
    out.save_to_disk(str(out_dir))
    if push:
        out.push_to_hub(
            REPO,
            split="test",
            commit_message="DiDeMo v2: cut each video to the first 30 s, which the descriptions cover",
        )


if __name__ == "__main__":
    main(Path(sys.argv[1]), "--push" in sys.argv)
