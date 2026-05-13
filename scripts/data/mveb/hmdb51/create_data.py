"""Reference data-preparation script for the mteb/HMDB51 dataset.

REFERENCE ONLY. Not runnable end-to-end: it depends on the original
HMDB51 RAR archives mirrored on Google Drive plus the official
testTrainMulti_7030_splits archive. The
canonical artifact produced by this pipeline is the `mteb/HMDB51`
dataset on the HuggingFace Hub.

Source
------
Kuehne, Jhuang, Garrote, Poggio, Serre, "HMDB: A Large Video Database
for Human Motion Recognition" (ICCV 2011).
    https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/

Upstream archives used (Google Drive mirrors):
    hmdb51_org.rar                  (videos, organized by class folder)
    test_train_splits.rar           (official 70/30 split lists)
        -> testTrainMulti_7030_splits/{class}_test_split{N}.txt

MVEB-specific processing
------------------------
1. Unrar the `hmdb51_org` archive (51 class folders of `.avi` clips) and
   the split-list archive. Use split number 1 of the official splits.
2. Parse each `{class}_test_split1.txt`: flag `1` -> train, flag `2` ->
   test, flag `0` -> ignored.
3. Keep classes with at least 13 examples in BOTH train and test (all 51
   classes pass this threshold).
4. Cap at 70 examples per class per split.
5. No audio (HMDB51 clips do not carry meaningful audio).
6. Schema: {video, label}. `label` is a ClassLabel over the 51 action
   classes (sorted alphabetically).
7. Push train and test splits to `mteb/HMDB51`.
8. Final sizes: ~3,570 train / ~1,530 test.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from datasets import ClassLabel, Dataset, Features, Video

VIDEO_ROOT = Path("hmdb51_org_extracted")
SPLIT_ROOT = Path("splits/testTrainMulti_7030_splits")
SPLIT_NUM = 1

CLASS_THRESHOLD = 13
MAX_PER_CLASS = 70
TARGET_REPO = "mteb/HMDB51"


def parse_splits(
    classes: list[str],
) -> tuple[dict[Path, str], dict[Path, str]]:
    train_map: dict[Path, str] = {}
    test_map: dict[Path, str] = {}
    for cls in classes:
        split_file = SPLIT_ROOT / f"{cls}_test_split{SPLIT_NUM}.txt"
        if not split_file.exists():
            continue
        for line in split_file.read_text().splitlines():
            video_name, flag = line.strip().split()
            video_path = VIDEO_ROOT / cls / video_name
            if flag == "1":
                train_map[video_path] = cls
            elif flag == "2":
                test_map[video_path] = cls
    return train_map, test_map


def select_balanced_subset(
    label_map: dict[Path, str],
    valid_classes: set[str],
) -> list[Path]:
    selected: list[Path] = []
    per_class: dict[str, int] = defaultdict(int)
    for path, cls in label_map.items():
        if (
            path.exists()
            and cls in valid_classes
            and per_class[cls] < MAX_PER_CLASS
        ):
            selected.append(path)
            per_class[cls] += 1
    return selected


def build_split(
    videos: list[Path],
    label_map: dict[Path, str],
    features: Features,
) -> Dataset:
    def gen():
        for vp in videos:
            yield {"video": str(vp), "label": label_map[vp]}

    return Dataset.from_generator(gen, features=features)


def main() -> None:
    classes = sorted(p.name for p in VIDEO_ROOT.iterdir() if p.is_dir())

    train_map, test_map = parse_splits(classes)

    train_counts = Counter(train_map.values())
    test_counts = Counter(test_map.values())
    valid_classes = {
        c for c in train_counts
        if train_counts[c] >= CLASS_THRESHOLD
        and test_counts.get(c, 0) >= CLASS_THRESHOLD
    }

    train_videos = select_balanced_subset(train_map, valid_classes)
    test_videos = select_balanced_subset(test_map, valid_classes)

    features = Features({
        "video": Video(),
        "label": ClassLabel(names=classes),
    })

    train_ds = build_split(train_videos, train_map, features)
    test_ds = build_split(test_videos, test_map, features)

    train_ds.push_to_hub(TARGET_REPO, split="train")
    test_ds.push_to_hub(TARGET_REPO, split="test")


if __name__ == "__main__":
    main()
