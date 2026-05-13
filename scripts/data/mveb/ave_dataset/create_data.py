"""Reference data-preparation script for the mteb/AVE-Dataset.

REFERENCE ONLY — RECONSTRUCTED. Reconstructed from the surviving source URL and the published HF schema; not a faithful replay of the original Colab.

Source
------
AVE: Tian et al., "Audio-Visual Event Localization in Unconstrained
Videos" (ECCV 2018). https://sites.google.com/view/audiovisualresearch

Mirror used:
    Videos + annotations:
      https://huggingface.co/datasets/UnFaZeD07/AVE-Dataset
      (videos.zip containing ~10-second clips, plus an annotation CSV
       with youtube_id, start_seconds, label, split)

MVEB-specific processing
------------------------
1. Download and unzip `videos.zip` from the UnFaZeD07 AVE mirror.
2. Parse the AVE annotation CSV. AVE officially defines 28 audio-visual
   event classes spanning roughly 10s YouTube clips.
3. Filter rows to clips that resolve to a local mp4 file.
4. Extract mono 16 kHz PCM audio from each clip via ffmpeg.
5. Schema: {video, audio, label (ClassLabel over 28 classes), youtube_id,
   start_seconds}.
6. Final sizes: ~3,312 train / ~402 test.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
from datasets import (
    Audio,
    ClassLabel,
    Dataset,
    DatasetDict,
    Features,
    Value,
    Video,
)

VIDEO_ROOT = Path("videos")
ANNOTATION_CSV = "AVE_annotations.csv"
TARGET_REPO = "mteb/AVE-Dataset"


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


def build_split(df: pd.DataFrame, features: Features) -> Dataset:
    def gen():
        for row in df.itertuples():
            video_path = VIDEO_ROOT / f"{row.youtube_id}.mp4"
            if not video_path.exists():
                continue
            wav = extract_audio_16k_mono(video_path)
            if wav is None:
                continue
            yield {
                "video": str(video_path),
                "audio": str(wav),
                "label": row.label,
                "youtube_id": row.youtube_id,
                "start_seconds": int(row.start_seconds),
            }

    return Dataset.from_generator(gen, features=features)


def main() -> None:
    df = pd.read_csv(ANNOTATION_CSV)
    class_names = sorted(df["label"].unique().tolist())

    features = Features({
        "video": Video(),
        "audio": Audio(sampling_rate=16000),
        "label": ClassLabel(names=class_names),
        "youtube_id": Value("string"),
        "start_seconds": Value("int32"),
    })

    train_ds = build_split(df[df["split"] == "train"], features)
    test_ds = build_split(df[df["split"] == "test"], features)

    DatasetDict({"train": train_ds, "test": test_ds}).push_to_hub(TARGET_REPO)


if __name__ == "__main__":
    main()
