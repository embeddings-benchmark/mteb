"""Reference data-preparation script for the mteb/Breakfast dataset.

REFERENCE ONLY. Not runnable end-to-end: the upstream Breakfast Actions
archive is mirrored on a Google Drive link that may be retired. The canonical artifact produced by this pipeline is the
`mteb/Breakfast` dataset on the HuggingFace Hub.

Source
------
Kuehne, Arslan, Serre, "The Language of Actions: Recovering the Syntax
and Semantics of Goal-Directed Human Activities" (CVPR 2014).
    https://serre-lab.clps.brown.edu/resource/breakfast-actions-dataset/

Upstream archive used (Google Drive mirror):
    BreakfastII_15fps_qvga_sync.tar.gz
    (file id 1jgSoof1AatiDRpGY091qd4TEKF-BUt6I)

MVEB-specific processing
------------------------
1. Download and extract the QVGA, 15 fps "BreakfastII" archive. Use only
   videos from camera 01 (`*/cam01/*.avi`).
2. Derive the meal class label from each filename: `P{subject}_{meal}.avi`
   -> `{meal}` (e.g. `P47_tea.avi` -> `tea`). This yields 10 meal classes.
3. Keep classes with at least 13 test examples and cap at 50 examples per
   class (all 10 classes pass the threshold).
4. No audio extraction (this Breakfast distribution is silent / sync-only).
   Schema: {video, label}. `label` is a ClassLabel over the 10 meal types
   (sorted alphabetically).
5. Push a single `test` split to `mteb/Breakfast`.
6. Final size: ~433 rows in the single `test` split.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from datasets import ClassLabel, Dataset, Features, Video

VIDEO_ROOT = Path("BreakfastII_15fps_qvga_sync")
VIDEO_GLOB = "*/cam01/*.avi"

CLASS_THRESHOLD = 13
MAX_PER_CLASS = 50
TARGET_REPO = "mteb/Breakfast"


def meal_from_filename(video_path: Path) -> str:
    # Filenames look like "P47_tea.avi" -> "tea".
    return video_path.stem.split("_", 1)[1]


def select_balanced_subset(
    video_paths: list[Path],
    label_map: dict[Path, str],
    valid_classes: set[str],
) -> list[Path]:
    selected: list[Path] = []
    per_class: dict[str, int] = defaultdict(int)
    for vp in video_paths:
        cls = label_map[vp]
        if cls in valid_classes and per_class[cls] < MAX_PER_CLASS:
            selected.append(vp)
            per_class[cls] += 1
    return selected


def main() -> None:
    video_paths = sorted(VIDEO_ROOT.glob(VIDEO_GLOB))
    label_map = {vp: meal_from_filename(vp) for vp in video_paths}

    counts = Counter(label_map.values())
    valid_classes = {c for c, n in counts.items() if n >= CLASS_THRESHOLD}

    test_videos = select_balanced_subset(video_paths, label_map, valid_classes)

    class_names = sorted(set(label_map.values()))
    features = Features({
        "video": Video(),
        "label": ClassLabel(names=class_names),
    })

    def gen():
        for vp in test_videos:
            yield {"video": str(vp), "label": label_map[vp]}

    test_ds = Dataset.from_generator(gen, features=features)
    test_ds.push_to_hub(TARGET_REPO, split="test")


if __name__ == "__main__":
    main()
