"""Build the EPICKitchens100Clustering video clustering task from EPIC-KITCHENS-100.

Source: lightly-ai/epic-kitchens-100-clips, a pre-trimmed mirror of the
official EPIC-KITCHENS-100 action clips (the full official release is
740GB-1.1TB, impractical to mirror in full; this mirror is CC-BY-NC-4.0
and not gated). The repo ships 37,455 individual clip files
(clips/{participant_id}/{narration_id}.mp4) plus a metadata table with
one row per narration annotation -- video isn't a packaged dataset
column, so the clip file list and the metadata table are cross-referenced
by narration_id (every clip file has a matching metadata row; not every
metadata row has an extracted clip file).

Each metadata row has a `verb` (the raw narration verb string, e.g.
"pour-out") and a `verb_class` (an integer id grouping semantically
equivalent verb strings, e.g. both "pour-out" and "pour-from" share one
verb_class). Cluster ground truth is verb_class, not the raw `verb`
string -- using the raw string would fragment one real action category
into many labels. Since a verb_class's narrations don't all share one
surface form, the label used is each verb_class's most frequent `verb`
string.

25 verb classes with at least CLIPS_PER_CLASS available clips are
selected (by descending clip count, ties broken by class id) and
CLIPS_PER_CLASS clips are sampled from each, for 500 clips total. Only
the ~500 selected clips are downloaded, not the full 37,455-clip mirror.

Usage:
    python scripts/data/epic_kitchens100_clustering/create_data.py \\
        --push-to-hub yaswanth169/EPIC-KITCHENS100-VideoClustering
"""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict

from datasets import Dataset, DatasetDict, Video, load_dataset
from huggingface_hub import hf_hub_download, list_repo_files

SOURCE_DATASET = "lightly-ai/epic-kitchens-100-clips"
SOURCE_SPLIT = "train"
NUM_CLASSES = 25
CLIPS_PER_CLASS = 20
SEED = 42


def _available_clips() -> dict[str, str]:
    """Map narration_id -> clip repo path, for narrations with an extracted clip."""
    files = list_repo_files(SOURCE_DATASET, repo_type="dataset")
    clips = [f for f in files if f.startswith("clips/") and f.endswith(".mp4")]
    return {path.rsplit("/", 1)[-1][:-4]: path for path in clips}


def build_split(seed: int = SEED) -> Dataset:
    clip_paths = _available_clips()

    metadata = load_dataset(SOURCE_DATASET, split=SOURCE_SPLIT)
    metadata = metadata.select_columns(["narration_id", "verb", "verb_class"])

    by_class: dict[int, list[str]] = defaultdict(list)
    verb_counts: dict[int, Counter] = defaultdict(Counter)
    for row in metadata:
        narration_id = row["narration_id"]
        if narration_id not in clip_paths:
            continue
        by_class[row["verb_class"]].append(narration_id)
        verb_counts[row["verb_class"]][row["verb"]] += 1

    eligible = sorted(
        (c for c, ids in by_class.items() if len(ids) >= CLIPS_PER_CLASS),
        key=lambda c: (-len(by_class[c]), c),
    )[:NUM_CLASSES]
    assert len(eligible) == NUM_CLASSES, (
        f"only {len(eligible)} classes have >= {CLIPS_PER_CLASS} available clips"
    )

    rng = random.Random(seed)
    rows: list[dict] = []
    for verb_class in eligible:
        label = verb_counts[verb_class].most_common(1)[0][0]
        narration_ids = list(by_class[verb_class])
        rng.shuffle(narration_ids)
        for narration_id in narration_ids[:CLIPS_PER_CLASS]:
            local_path = hf_hub_download(
                repo_id=SOURCE_DATASET,
                repo_type="dataset",
                filename=clip_paths[narration_id],
            )
            rows.append(
                {
                    "narration_id": narration_id,
                    "video": local_path,
                    "label": label,
                }
            )

    dataset = Dataset.from_list(rows)
    return dataset.cast_column("video", Video())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--push-to-hub", default=None, help="HF repo id to push to")
    args = parser.parse_args()

    dataset = build_split()

    print(f"clips: {len(dataset)}, classes: {len(set(dataset['label']))}")
    assert len(dataset) == NUM_CLASSES * CLIPS_PER_CLASS
    assert len(set(dataset["label"])) == NUM_CLASSES
    assert len(set(dataset["narration_id"])) == len(dataset)

    if args.push_to_hub:
        DatasetDict({"test": dataset}).push_to_hub(args.push_to_hub)


if __name__ == "__main__":
    main()
