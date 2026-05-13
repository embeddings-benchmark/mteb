"""Reference data-preparation script for the mteb/VGGSound dataset.

REFERENCE ONLY. Not runnable end-to-end: it depends on a third-party HF
mirror of the raw VGGSound clips. The
canonical artifact produced by this pipeline is the `mteb/VGGSound`
dataset on the HuggingFace Hub.

Source
------
Chen et al., "VGG-Sound: A Large-scale Audio-Visual Dataset" (ICASSP 2020).
    https://arxiv.org/abs/2004.14368

Download infrastructure used (upstream):
    HF dataset `11hu83/vggsound` (the `video/` subtree only), plus the
    official `vggsound.csv` annotation file with columns
    `youtube_id,start_seconds,label,split`.

MVEB-specific processing
------------------------
1. Filter `vggsound.csv` to rows where `split == "test"`. Each test row maps
   to `video/<youtube_id>_<start_seconds:06d>/video.mp4`. Drop any rows
   whose mp4 is missing locally.
2. Keep all classes with at least 1 test example.
3. Cap at 32 examples per class.
4. Extract mono 16 kHz PCM audio with ffmpeg. Drop clips where extraction
   fails.
5. Schema: {video, audio, label}. `label` is a ClassLabel over the surviving
   VGGSound class names (309 classes, sorted alphabetically).
6. Final size: ~9,888 rows in the single `test` split.
"""

from __future__ import annotations

import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from datasets import Audio, ClassLabel, Dataset, Features, Video

VIDEO_ROOT = Path("vggsound_raw_videos/video")
VGGSOUND_CSV = "vggsound.csv"

CLASS_THRESHOLD = 1
MAX_PER_CLASS = 32
TARGET_REPO = "mteb/VGGSound"


def build_test_map() -> dict[str, str]:
    df = pd.read_csv(VGGSOUND_CSV, header=None)
    df.columns = ["youtube_id", "start_seconds", "label", "split"]
    df_test = df[df["split"] == "test"]

    test_map: dict[str, str] = {}
    for row in df_test.itertuples():
        folder = f"{row.youtube_id}_{int(row.start_seconds):06d}"
        video_path = VIDEO_ROOT / folder / "video.mp4"
        if video_path.exists():
            test_map[str(video_path)] = row.label
    return test_map


def select_balanced(
    test_map: dict[str, str], valid_classes: set[str],
) -> list[str]:
    selected: list[str] = []
    per_class: dict[str, int] = defaultdict(int)
    for path, cls in test_map.items():
        if cls in valid_classes and per_class[cls] < MAX_PER_CLASS:
            selected.append(path)
            per_class[cls] += 1
    return selected


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

    test_counts = Counter(test_map.values())
    valid_classes = {c for c, n in test_counts.items() if n >= CLASS_THRESHOLD}

    test_videos = select_balanced(test_map, valid_classes)

    class_names = sorted(set(test_map.values()))
    features = Features({
        "video": Video(),
        "audio": Audio(sampling_rate=16000),
        "label": ClassLabel(names=class_names),
    })

    def gen():
        for vp in test_videos:
            audio = extract_audio_16k_mono(vp)
            if audio is None:
                continue
            yield {"video": vp, "audio": audio, "label": test_map[vp]}

    test_ds = Dataset.from_generator(gen, features=features, num_proc=2)
    test_ds.push_to_hub(TARGET_REPO, split="test")


if __name__ == "__main__":
    main()
