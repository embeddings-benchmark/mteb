"""Reference data-preparation script for the mteb/EgoSchema_subset dataset.

REFERENCE ONLY — RECONSTRUCTED. Reconstructed from the published HF
schema; original pipeline unavailable.

Source
------
EgoSchema: Mangalam et al., "EgoSchema: A Diagnostic Benchmark for Very
Long-form Video Language Understanding" (NeurIPS 2023).
    https://egoschema.github.io/

Mirror used:
    Videos + QA:  https://huggingface.co/datasets/VLM2Vec/EgoSchema
                  (videos_chunked_01..05.zip; `Subset` config, split=test)

MVEB-specific processing
------------------------
1. Build a wanted-list of video filenames from the `Subset` test split.
2. Selectively extract those files from each of the five
   `videos_chunked_0{1..5}.zip` archives.
3. Load the `Subset` config of `VLM2Vec/EgoSchema` (test split) and pair
   each QA row with its local mp4 by `video_id` (UUID-style filename).
4. Lift the answer string from the 5-option `option` list using the
   integer index recorded by the source dataset.
5. Schema: {question_id, video_id, video, question, candidates, answer}.
   No audio extraction (the published artifact is video-only).
6. Final size: 500 rows in the single `test` split.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from datasets import (
    Dataset,
    Features,
    Sequence,
    Value,
    Video,
    load_dataset,
)

VIDEO_ROOT = Path("egoschema_videos")
ZIP_ARCHIVES = [Path(f"videos_chunked_{i:02d}.zip") for i in range(1, 6)]
TARGET_REPO = "mteb/EgoSchema_subset"


def selectively_extract(wanted: set[str]) -> None:
    for archive in ZIP_ARCHIVES:
        if not archive.exists():
            continue
        with zipfile.ZipFile(archive) as zf:
            for name in set(zf.namelist()) & wanted:
                zf.extract(name, VIDEO_ROOT)


def index_videos(root: Path) -> dict[str, Path]:
    return {p.stem: p for p in root.rglob("*.mp4") if p.is_file()}


def main() -> None:
    ds = load_dataset("VLM2Vec/EgoSchema", name="Subset", split="test")

    wanted = {f"{row['video_id']}.mp4" for row in ds}
    selectively_extract(wanted)
    video_index = index_videos(VIDEO_ROOT)

    features = Features({
        "question_id": Value("string"),
        "video_id": Value("string"),
        "video": Video(),
        "question": Value("string"),
        "candidates": Sequence(Value("string")),
        "answer": Value("string"),
    })

    def gen():
        for item in ds:
            video_path = video_index.get(item["video_id"])
            if video_path is None:
                continue
            candidates = item["option"]
            answer = candidates[int(item["answer"])]
            yield {
                "question_id": str(item["question_idx"]),
                "video_id": item["video_id"],
                "video": str(video_path),
                "question": item["question"],
                "candidates": candidates,
                "answer": answer,
            }

    test_ds = Dataset.from_generator(gen, features=features)
    test_ds.push_to_hub(TARGET_REPO, split="test")


if __name__ == "__main__":
    main()
