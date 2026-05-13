"""Reference data-preparation script for the mteb/Shot2Story20K_test dataset.

REFERENCE ONLY. Not runnable end-to-end: the upstream video archive is
served from Microsoft OneDrive, and the caption metadata lives in
`mhan/Shot2Story-20K` on the HuggingFace Hub.
The canonical artifact produced by this pipeline is the
`mteb/Shot2Story20K_test` dataset on the HuggingFace Hub.

Source
------
Han et al., "Shot2Story20K: A New Benchmark for Comprehensive Understanding
of Multi-shot Videos" (2023). https://arxiv.org/abs/2312.10300

Upstream artifacts:
    https://huggingface.co/datasets/mhan/Shot2Story-20K (captions)
    collation_final_videos_20k.tar (raw mp4 clips, OneDrive share)

MVEB-specific processing
------------------------
1. Download the `collation_final_videos_20k.tar` archive and selectively
   extract only the mp4 clips referenced by the `test` split of
   `mhan/Shot2Story-20K`.
2. Pair each surviving local clip with its `whole_caption` field.
3. Extract mono 16 kHz PCM audio with ffmpeg. Drop clips where extraction
   fails.
4. Deduplicate by caption (keep the first occurrence) to produce the
   final retrieval pool.
5. Schema: {video, audio, caption}. Single `test` split.
6. Final size: ~4,023 rows.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from datasets import Audio, Dataset, Features, Value, Video, load_dataset

SOURCE_REPO = "mhan/Shot2Story-20K"
VIDEO_ROOT = Path("collation_final_videos_20k")
TARGET_REPO = "mteb/Shot2Story20K_test"


def build_caption_map(ds) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in ds:
        video_path = VIDEO_ROOT / str(item["video"])
        if video_path.exists():
            mapping[str(video_path)] = item["whole_caption"]
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
    raw = load_dataset(SOURCE_REPO, split="test")
    caption_map = build_caption_map(raw)
    video_list = list(caption_map.keys())

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
