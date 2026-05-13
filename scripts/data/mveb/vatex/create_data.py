"""Reference data-preparation script for the mteb/VATEX_test_1k dataset.

REFERENCE ONLY. Not runnable end-to-end: the upstream `VLM2Vec/VATEX`
mirror hosts the raw mp4 clips, and the caption metadata comes from
`lmms-lab/vatex_from_url`. The canonical artifact
produced by this pipeline is the `mteb/VATEX_test_1k` dataset on the
HuggingFace Hub.

Source
------
Wang et al., "VATEX: A Large-Scale, High-Quality Multilingual Dataset
for Video-and-Language Research" (ICCV 2019).
    https://arxiv.org/abs/1904.03493

Upstream artifacts:
    https://huggingface.co/datasets/VLM2Vec/VATEX (raw_videos/)
    https://huggingface.co/datasets/lmms-lab/vatex_from_url (vatex_test config)

MVEB-specific processing
------------------------
1. Snapshot-download the `raw_videos/` directory of `VLM2Vec/VATEX`, and
   load the `vatex_test` config / `test` split of `lmms-lab/vatex_from_url`
   for caption metadata.
2. Pair each locally available `{videoID}.mp4` with its first English
   caption (`enCap[0]`).
3. Cap the final pool at 1,000 clips.
4. Extract mono 16 kHz PCM audio with ffmpeg. Drop clips where extraction
   fails.
5. Deduplicate by caption (keep first occurrence).
6. Schema: {video, audio, caption}. Single `test` split.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from datasets import Audio, Dataset, Features, Value, Video, load_dataset
from huggingface_hub import snapshot_download

SOURCE_VIDEO_REPO = "VLM2Vec/VATEX"
SOURCE_CAPTION_REPO = "lmms-lab/vatex_from_url"
SOURCE_CAPTION_CONFIG = "vatex_test"
LOCAL_VIDEO_DIR = Path("vatex_raw_videos")
VIDEO_ROOT = LOCAL_VIDEO_DIR / "raw_videos"
MAX_CLIPS = 1000
TARGET_REPO = "mteb/VATEX_test_1k"


def download_videos() -> None:
    snapshot_download(
        repo_id=SOURCE_VIDEO_REPO,
        repo_type="dataset",
        local_dir=str(LOCAL_VIDEO_DIR),
        allow_patterns=["raw_videos/*"],
    )


def build_caption_map(ds) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in ds:
        video_path = VIDEO_ROOT / f"{item['videoID']}.mp4"
        if video_path.exists():
            mapping[str(video_path)] = item["enCap"][0]
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


def generator(video_list: list[str], mapping: dict[str, str]):
    for vp in video_list:
        wav = extract_audio_16k_mono(Path(vp))
        if wav is None:
            continue
        yield {"video": vp, "audio": str(wav), "caption": mapping[vp]}


def deduplicate_by_caption(ds: Dataset) -> Dataset:
    seen: set[str] = set()
    unique_indices: list[int] = []
    for i, caption in enumerate(ds["caption"]):
        if caption not in seen:
            seen.add(caption)
            unique_indices.append(i)
    return ds.select(unique_indices)


def main() -> None:
    download_videos()
    raw = load_dataset(SOURCE_CAPTION_REPO, name=SOURCE_CAPTION_CONFIG, split="test")

    caption_map = build_caption_map(raw)
    video_list = list(caption_map.keys())[:MAX_CLIPS]

    features = Features({
        "video": Video(),
        "audio": Audio(sampling_rate=16000),
        "caption": Value("string"),
    })

    test_ds = Dataset.from_generator(
        generator,
        gen_kwargs={"video_list": video_list, "mapping": caption_map},
        features=features,
    )

    ds_unique = deduplicate_by_caption(test_ds)
    ds_unique.push_to_hub(TARGET_REPO, split="test")


if __name__ == "__main__":
    main()
