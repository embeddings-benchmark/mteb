"""Reference data-preparation script for the mteb/DiDeMo dataset.

REFERENCE ONLY. Not runnable end-to-end: it depends on the
`friedrichor/DiDeMo` HF mirror of the raw DiDeMo videos and the
`VLM2Vec/DiDeMo` HF retrieval split. The canonical artifact produced by
this pipeline is the `mteb/DiDeMo` dataset on the HuggingFace Hub.

Source
------
Anne Hendricks et al., "Localizing Moments in Video with Natural Language"
(ICCV 2017).
    https://arxiv.org/abs/1708.01641

Download infrastructure used (upstream):
    https://huggingface.co/datasets/friedrichor/DiDeMo/resolve/main/DiDeMo_Videos_mp4_test.tar
    HF dataset `VLM2Vec/DiDeMo` (test split) for (video, caption) pairs.

MVEB-specific processing
------------------------
1. Load the `VLM2Vec/DiDeMo` test split: each entry has a `video` filename
   (under `test/`) and a free-text `caption`.
2. Selectively untar the source tarball using a generated `wanted_videos.txt`
   listing only the test-split filenames needed.
3. Join entries to `video/test/<filename>` on disk. Drop entries whose mp4
   is missing.
4. Extract mono 16 kHz PCM audio with ffmpeg. Pre-filter the video list to
   only those for which audio extraction succeeds.
5. Deduplicate by caption (keep first occurrence).
6. Schema: {video, audio, caption}.
7. Final size: ~999 rows in the single `test` split.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from datasets import Audio, Dataset, Features, Value, Video, load_dataset

VIDEO_ROOT = Path("video/test")
SOURCE_REPO = "VLM2Vec/DiDeMo"
TARGET_REPO = "mteb/DiDeMo"


def build_test_map() -> dict[str, str]:
    ds = load_dataset(SOURCE_REPO, split="test")

    test_map: dict[str, str] = {}
    for item in ds:
        rel = item["video"].replace("test/", "")
        video_path = VIDEO_ROOT / rel
        if video_path.exists():
            test_map[str(video_path)] = item["caption"]
    return test_map


def extract_audio_16k_mono(video_path: str) -> str | None:
    wav = str(Path(video_path).with_suffix(".wav"))
    result = subprocess.run(
        [
            "ffmpeg", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            wav, "-y",
        ],
        capture_output=True,
    )
    return wav if result.returncode == 0 else None


def main() -> None:
    test_map = build_test_map()
    test_videos = list(test_map.keys())

    valid_videos = [vp for vp in test_videos if extract_audio_16k_mono(vp) is not None]

    features = Features({
        "video": Video(),
        "audio": Audio(sampling_rate=16000),
        "caption": Value("string"),
    })

    def gen():
        for vp in valid_videos:
            audio = extract_audio_16k_mono(vp)
            if audio is None:
                continue
            yield {"video": vp, "audio": audio, "caption": test_map[vp]}

    test_ds = Dataset.from_generator(gen, features=features, num_proc=2)

    seen: set[str] = set()
    unique_indices: list[int] = []
    for i, caption in enumerate(test_ds["caption"]):
        if caption not in seen:
            seen.add(caption)
            unique_indices.append(i)

    ds_unique = test_ds.select(unique_indices)
    ds_unique.push_to_hub(TARGET_REPO, split="test")


if __name__ == "__main__":
    main()
