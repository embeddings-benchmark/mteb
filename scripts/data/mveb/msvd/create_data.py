"""Reference data-preparation script for the mteb/MSVD dataset.

REFERENCE ONLY. Not runnable end-to-end: it depends on the UT Austin MSVD
mirror plus the `VLM2Vec/MSVD` HF dataset.
The canonical artifact produced by this pipeline is the `mteb/MSVD` dataset
on the HuggingFace Hub.

Source
------
Chen and Dolan, "Collecting Highly Parallel Data for Paraphrase Evaluation"
(ACL 2011) - the YouTubeClips corpus that came to be known as MSVD.
    https://aclanthology.org/P11-1020/

Download infrastructure used (upstream):
    https://www.cs.utexas.edu/~ml/clamp/videoDescription/YouTubeClips.tar
    HF dataset `VLM2Vec/MSVD` (test split) for the (video_id, captions) pairs.

MVEB-specific processing
------------------------
1. Decode `YouTubeClips.tar` to `YouTubeClips/<video_id>.avi`.
2. Load `VLM2Vec/MSVD` test split. Each row carries one `video` filename
   and a list of paraphrase `caption`s; we take the first caption only.
3. Join rows to `YouTubeClips/<video>` on disk. Drop rows whose .avi is
   missing.
4. Extract mono 16 kHz PCM audio with ffmpeg. Pre-filter the video list to
   only those for which audio extraction succeeds. Audio extraction is used
   as a quality gate but not included in the published schema.
5. Deduplicate by caption (keep first occurrence).
6. Schema: {video, caption}.
7. Final size: ~660 rows in the single `test` split.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from datasets import Dataset, Features, Value, Video, load_dataset

VIDEO_ROOT = Path("YouTubeClips")
SOURCE_REPO = "VLM2Vec/MSVD"
TARGET_REPO = "mteb/MSVD"


def build_test_map() -> dict[str, str]:
    ds = load_dataset(SOURCE_REPO, split="test")

    test_map: dict[str, str] = {}
    for item in ds:
        first_caption = item["caption"][0]
        video_path = VIDEO_ROOT / item["video"]
        if video_path.exists():
            test_map[str(video_path)] = first_caption
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
        "caption": Value("string"),
    })

    def gen():
        for vp in valid_videos:
            yield {"video": vp, "caption": test_map[vp]}

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
