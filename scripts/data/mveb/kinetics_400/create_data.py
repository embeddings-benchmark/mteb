"""Reference data-preparation script for the mteb/kinetics-400 dataset.

REFERENCE ONLY. Not runnable end-to-end: it depends on the official
Kinetics-400 download infrastructure (S3 path lists + helper shell scripts).
The canonical artifact produced by this
pipeline is the `mteb/kinetics-400` dataset on the HuggingFace Hub.

Source
------
Kay et al., "The Kinetics Human Action Video Dataset" (2017).
    https://arxiv.org/abs/1705.06950

Download infrastructure used (upstream):
    https://github.com/cvdfoundation/kinetics-dataset
    https://s3.amazonaws.com/kinetics/400/{train,test}/k400_{train,test}_path.txt
    https://s3.amazonaws.com/kinetics/400/annotations/{train,test}.csv

MVEB-specific processing
------------------------
1. Take 25 random shard files from each of the official train and test
   path lists, then download + extract via the upstream `download.sh` /
   `extract.sh`.
2. Join filenames to action labels via the official annotation CSVs
   (`youtube_id`, `time_start`, `time_end` -> `label`).
3. Keep classes that have at least 10 examples in BOTH train and test
   (all 400 classes pass this threshold).
4. Cap at 10 examples per class per split.
5. Extract mono 16 kHz PCM audio with ffmpeg. Drop clips where extraction
   fails (~30 per split). Final sizes: ~3,970 train / 4,000 test.
6. Schema: {video, audio, label}; ClassLabel over the 400 sorted action class names.
"""

from __future__ import annotations

import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from datasets import Audio, ClassLabel, Dataset, DatasetDict, Features, Video

VIDEO_ROOT = Path("kinetics-dataset")
TRAIN_CSV = "train.csv"
TEST_CSV = "test.csv"

CLASS_THRESHOLD = 10
MAX_PER_CLASS = 10
TARGET_REPO = "mteb/kinetics-400"


def build_filename_to_label(csv_path: str) -> dict[str, str]:
    df = pd.read_csv(csv_path)
    return {
        f"{r.youtube_id}_{int(r.time_start):06d}_{int(r.time_end):06d}.mp4": r.label
        for r in df.itertuples()
    }


def select_balanced_subset(
    video_paths: list[Path],
    label_map: dict[str, str],
    valid_classes: set[str],
) -> list[Path]:
    selected: list[Path] = []
    per_class: dict[str, int] = defaultdict(int)
    for vp in video_paths:
        cls = label_map.get(vp.name)
        if cls in valid_classes and per_class[cls] < MAX_PER_CLASS:
            selected.append(vp)
            per_class[cls] += 1
    return selected


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


def build_split(
    videos: list[Path],
    label_map: dict[str, str],
    features: Features,
) -> Dataset:
    def gen():
        for vp in videos:
            wav = extract_audio_16k_mono(vp)
            if wav is None:
                continue
            yield {"video": str(vp), "audio": str(wav), "label": label_map[vp.name]}

    return Dataset.from_generator(gen, features=features)


def main() -> None:
    train_map = build_filename_to_label(TRAIN_CSV)
    test_map = build_filename_to_label(TEST_CSV)

    video_paths = list(VIDEO_ROOT.rglob("*.mp4"))

    train_counts = Counter(train_map[p.name] for p in video_paths if p.name in train_map)
    test_counts = Counter(test_map[p.name] for p in video_paths if p.name in test_map)

    valid_classes = {
        c for c in train_counts
        if train_counts[c] >= CLASS_THRESHOLD
        and test_counts.get(c, 0) >= CLASS_THRESHOLD
    }

    train_videos = select_balanced_subset(video_paths, train_map, valid_classes)
    test_videos = select_balanced_subset(video_paths, test_map, valid_classes)

    class_names = sorted(set(train_map.values()) | set(test_map.values()))
    features = Features({
        "video": Video(),
        "audio": Audio(sampling_rate=16000),
        "label": ClassLabel(names=class_names),
    })

    train_ds = build_split(train_videos, train_map, features)
    test_ds = build_split(test_videos, test_map, features)

    DatasetDict({"train": train_ds, "test": test_ds}).push_to_hub(TARGET_REPO)


if __name__ == "__main__":
    main()
