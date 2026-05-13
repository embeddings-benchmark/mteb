"""Reference data-preparation script for the mteb/MSR-VTT dataset.

REFERENCE ONLY. Not runnable end-to-end: it depends on a Kaggle mirror of
MSR-VTT and on the standard `msrvtt_ret_test1k.json` annotation file.
The canonical artifact produced by this pipeline
is the `mteb/MSR-VTT` dataset on the HuggingFace Hub.

Source
------
Xu et al., "MSR-VTT: A Large Video Description Dataset for Bridging Video
and Language" (CVPR 2016).
    https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/cvpr16.msr-vtt.tmei_-1.pdf

Download infrastructure used (upstream):
    Kaggle dataset `khoahunhtngng/msrvtt` (mirrors `MSR-VTT/TestVideo/*.mp4`)
    and the JSF retrieval test split file `msrvtt_ret_test1k.json`.

MVEB-specific processing
------------------------
1. Load `msrvtt_ret_test1k.json`: a 1k video-caption retrieval test split.
   Each entry has a `video` filename and a `caption` string.
2. Join entries to `MSR-VTT/TestVideo/<video>.mp4`. Drop entries whose mp4
   is missing locally.
3. Extract mono 16 kHz PCM audio with ffmpeg. Pre-filter the video list to
   only those for which audio extraction succeeds.
4. Deduplicate by caption (keep first occurrence) so the retrieval set has
   one unique caption per row.
5. Schema: {video, audio, caption}.
6. Final size: 879 rows in the single `test` split.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from datasets import Audio, Dataset, Features, Value, Video

VIDEO_ROOT = Path("msr_dataset/MSR-VTT/TestVideo")
RET_TEST_JSON = "msrvtt_ret_test1k.json"
TARGET_REPO = "mteb/MSR-VTT"


def build_test_map() -> dict[str, str]:
    with open(RET_TEST_JSON) as f:
        ret_test = json.load(f)

    test_map: dict[str, str] = {}
    for item in ret_test:
        video_path = VIDEO_ROOT / item["video"]
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
