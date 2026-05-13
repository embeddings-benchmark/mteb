"""Reference data-preparation script for the mteb/SomethingSomethingV2 dataset.

REFERENCE ONLY. Not runnable end-to-end: it depends on the Qualcomm AI Dataset
download mirror for Something-Something-V2.
The canonical artifact produced by this pipeline is the
`mteb/SomethingSomethingV2` dataset on the HuggingFace Hub.

Source
------
Goyal et al., "The `something something` video database for learning and
evaluating visual common sense" (ICCV 2017).
    https://arxiv.org/abs/1706.04261

Download infrastructure used (upstream):
    https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/dataset/AIDataset/Something-Something-V2/
        20bn-something-something-v2-00
        20bn-something-something-v2-01
    The numbered shards are concatenated and untarred to yield
    `20bn-something-something-v2/*.webm` plus the `labels.json` and
    `validation.json` annotation files.

MVEB-specific processing
------------------------
1. Use the official `validation.json` annotations as the test split. Map each
   clip's `template` (with the `[]` placeholders removed) to the integer class
   id from `labels.json`. There are 174 templated action classes.
2. Keep classes with at least 13 examples in the validation split.
3. Cap at 32 examples per class.
4. Video-only schema: {video, label}. Audio is intentionally not extracted -
   SSv2 clips are silent.
5. `label` is a ClassLabel over the 174 SSv2 templates (order from
   `labels.json`).
6. Final size: ~5,444 rows in the single `test` split.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from datasets import ClassLabel, Dataset, Features, Video

VIDEO_ROOT = Path("20bn-something-something-v2")
LABELS_JSON = "labels.json"
VALIDATION_JSON = "validation.json"

CLASS_THRESHOLD = 13
MAX_PER_CLASS = 32
TARGET_REPO = "mteb/SomethingSomethingV2"


def load_class_to_idx() -> dict[str, int]:
    with open(LABELS_JSON) as f:
        raw = json.load(f)
    return {k: int(v) for k, v in raw.items()}


def build_test_map(class_to_idx: dict[str, int]) -> dict[str, int]:
    with open(VALIDATION_JSON) as f:
        validation_data = json.load(f)

    test_map: dict[str, int] = {}
    for item in validation_data:
        template = item["template"].replace("[", "").replace("]", "")
        video_path = VIDEO_ROOT / f"{item['id']}.webm"
        if video_path.exists():
            test_map[str(video_path)] = class_to_idx[template]
    return test_map


def select_balanced(test_map: dict[str, int], valid_classes: set[int]) -> list[str]:
    selected: list[str] = []
    per_class: dict[int, int] = defaultdict(int)
    for path, cls in test_map.items():
        if cls in valid_classes and per_class[cls] < MAX_PER_CLASS:
            selected.append(path)
            per_class[cls] += 1
    return selected


def main() -> None:
    class_to_idx = load_class_to_idx()
    test_map = build_test_map(class_to_idx)

    test_counts = Counter(test_map.values())
    valid_classes = {c for c, n in test_counts.items() if n >= CLASS_THRESHOLD}

    test_videos = select_balanced(test_map, valid_classes)

    class_names = list(class_to_idx.keys())
    features = Features({
        "video": Video(),
        "label": ClassLabel(names=class_names),
    })

    def gen():
        for vp in test_videos:
            yield {"video": vp, "label": test_map[vp]}

    test_ds = Dataset.from_generator(gen, features=features, num_proc=2)
    test_ds.push_to_hub(TARGET_REPO, split="test")


if __name__ == "__main__":
    main()
