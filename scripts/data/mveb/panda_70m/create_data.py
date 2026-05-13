"""Reference data-preparation script for the mteb/panda-70m dataset.

REFERENCE ONLY. Not runnable end-to-end: it depends on yt-dlp downloads
from YouTube (which require valid cookies) and on the
`multimodalart/panda-70m` metadata table. The
canonical artifact produced by this pipeline is the `mteb/panda-70m`
dataset on the HuggingFace Hub.

Source
------
Chen et al., "Panda-70M: Captioning 70M Videos with Multiple Cross-Modality
Teachers" (CVPR 2024). https://arxiv.org/abs/2402.19479

Upstream artifacts:
    https://huggingface.co/datasets/multimodalart/panda-70m (test split CSV)
    YouTube source videos (downloaded via yt-dlp)

MVEB-specific processing
------------------------
1. For each row of the `multimodalart/panda-70m` `test` split, download
   the source YouTube video with yt-dlp (using a cookies file), then cut
   it into clips at the timestamps recorded in the metadata via
   `ffmpeg -ss/-to -c copy`. Each clip is named `{video_id}_{i}.mp4`.
2. Pair each surviving local clip with its caption, source YouTube id,
   and clip timestamp from the metadata.
3. Extract mono 16 kHz PCM audio with ffmpeg. Drop clips where extraction
   fails.
4. Deduplicate by caption (keep first occurrence) to produce the final
   retrieval pool.
5. Schema: {video, audio, caption, youtube_id, timestamp}. Single `test`
   split.
6. Final size: ~3,395 rows in the single `test` split.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

from datasets import Audio, Dataset, Features, Value, Video, load_dataset

ROOT = Path("/content/drive/MyDrive/Panda70M")
RAW_DIR = ROOT / "raw_videos"
CLIP_DIR = ROOT / "clips"
COOKIES_FILE = "youtube_cookies.txt"

SOURCE_REPO = "multimodalart/panda-70m"
TARGET_REPO = "mteb/panda-70m"


def download_video(url: str, video_id: str) -> Path | None:
    output_path = RAW_DIR / f"{video_id}.mp4"
    if output_path.exists():
        return output_path
    cmd = [
        "yt-dlp", "--cookies", COOKIES_FILE,
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best",
        "--merge-output-format", "mp4",
        "-o", str(output_path), url,
    ]
    result = subprocess.run(cmd)
    return output_path if result.returncode == 0 else None


def extract_clip(video_path: Path, start: str, end: str, output_path: Path) -> bool:
    cmd = [
        "ffmpeg", "-y", "-ss", start, "-to", end,
        "-i", str(video_path), "-c", "copy", str(output_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0


def cut_clips_from_metadata(ds) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    for item in ds:
        video_id = item["videoID"]
        timestamps = ast.literal_eval(item["timestamp"])
        raw = download_video(item["url"], video_id)
        if raw is None:
            continue
        for i, (start, end) in enumerate(timestamps):
            clip_path = CLIP_DIR / f"{video_id}_{i}.mp4"
            if not clip_path.exists():
                extract_clip(raw, start, end, clip_path)


def build_clip_map(ds) -> dict[str, dict]:
    mapping: dict[str, dict] = {}
    for item in ds:
        captions = ast.literal_eval(item["caption"])
        timestamps = ast.literal_eval(item["timestamp"])
        for i, caption in enumerate(captions):
            clip_path = CLIP_DIR / f"{item['videoID']}_{i}.mp4"
            if clip_path.exists():
                mapping[str(clip_path)] = {
                    "caption": caption,
                    "youtube_id": item["videoID"],
                    "timestamp": str(timestamps[i]),
                }
    return mapping


def extract_audio_16k_mono(video_path: Path) -> Path | None:
    wav = video_path.with_suffix(".wav")
    result = subprocess.run(
        [
            "ffmpeg", "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            str(wav), "-y",
        ],
        capture_output=True,
    )
    return wav if result.returncode == 0 else None


def generator(video_list: list[str], mapping: dict[str, dict]):
    for vp in video_list:
        wav = extract_audio_16k_mono(Path(vp))
        if wav is None:
            continue
        meta = mapping[vp]
        yield {
            "video": vp,
            "audio": str(wav),
            "caption": meta["caption"],
            "youtube_id": meta["youtube_id"],
            "timestamp": meta["timestamp"],
        }


def deduplicate_by_caption(ds: Dataset) -> Dataset:
    seen: set[str] = set()
    unique_indices: list[int] = []
    for i, caption in enumerate(ds["caption"]):
        if caption not in seen:
            seen.add(caption)
            unique_indices.append(i)
    return ds.select(unique_indices)


def main() -> None:
    raw = load_dataset(SOURCE_REPO, split="test")
    cut_clips_from_metadata(raw)
    clip_map = build_clip_map(raw)

    features = Features({
        "video": Video(),
        "audio": Audio(sampling_rate=16000),
        "caption": Value("string"),
        "youtube_id": Value("string"),
        "timestamp": Value("string"),
    })

    test_ds = Dataset.from_generator(
        generator,
        gen_kwargs={"video_list": list(clip_map.keys()), "mapping": clip_map},
        features=features,
    )

    ds_unique = deduplicate_by_caption(test_ds)
    ds_unique.push_to_hub(TARGET_REPO, split="test")


if __name__ == "__main__":
    main()
