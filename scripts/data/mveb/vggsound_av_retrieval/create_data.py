"""Reference data-preparation script for the mteb/VGGSound_AV_RETRIEVAL dataset.

REFERENCE ONLY. The canonical artifact is `mteb/VGGSound_AV_RETRIEVAL`
on the HuggingFace Hub.

Source
------
VGGSound: Chen et al., "VGGSound: A Large-Scale Audio-Visual Dataset"
(ICASSP 2020). https://www.robots.ox.ac.uk/~vgg/data/vggsound/

Mirrors used:
    Captions:  https://huggingface.co/datasets/jianzongwu/VGGSound-T2AV
               (split=test; `prompt_v`, `prompt_a`, `video_path` fields)
    Videos:    https://huggingface.co/datasets/11hu83/vggsound
               (snapshot_download with allow_patterns=["video/*"])

MVEB-specific processing
------------------------
1. Snapshot-download the `video/*` tree of `11hu83/vggsound` (each clip
   is stored as `<video_root>/<youtube_id>_<start>/video.mp4`).
2. Load `jianzongwu/VGGSound-T2AV` test split as the caption source.
3. For each row, resolve the local clip path as
   `<video_root>/<youtube_id>_<start>/video.mp4` from the T2AV
   `video_path` field.
4. Extract mono 16 kHz PCM audio with ffmpeg.
5. Carry both prompts through as `video_caption` (`prompt_v`) and
   `audio_caption` (`prompt_a`).
6. Deduplicate by `video_caption` so a single clip yields one row.
7. Schema: {video, audio, video_caption, audio_caption}. Final size:
   ~696 rows in the single `test` split.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from datasets import Audio, Dataset, Features, Value, Video, load_dataset

VIDEO_ROOT = Path("vggsound_raw_videos/video")
TARGET_REPO = "mteb/VGGSound_AV_RETRIEVAL"


def derive_video_path(video_path_field: str) -> Path:
    # T2AV stores `<split>/<youtube_id>_<start>.mp4`; the 11hu83 mirror
    # exposes clips as `<video_root>/<youtube_id>_<start>/video.mp4`.
    folder = video_path_field.split("/")[1].replace(".mp4", "")
    return VIDEO_ROOT / folder / "video.mp4"


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


def main() -> None:
    ds = load_dataset("jianzongwu/VGGSound-T2AV", split="test")

    features = Features({
        "video": Video(),
        "audio": Audio(sampling_rate=16000),
        "video_caption": Value("string"),
        "audio_caption": Value("string"),
    })

    def gen():
        seen: set[str] = set()
        for item in ds:
            video_path = derive_video_path(item["video_path"])
            if not video_path.exists():
                continue
            video_caption = item["prompt_v"]
            if video_caption in seen:
                continue
            seen.add(video_caption)
            wav = extract_audio_16k_mono(video_path)
            if wav is None:
                continue
            yield {
                "video": str(video_path),
                "audio": str(wav),
                "video_caption": video_caption,
                "audio_caption": item["prompt_a"],
            }

    test_ds = Dataset.from_generator(gen, features=features)
    test_ds.push_to_hub(TARGET_REPO, split="test")


if __name__ == "__main__":
    main()
