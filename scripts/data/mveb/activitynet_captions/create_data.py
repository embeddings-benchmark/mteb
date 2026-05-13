"""Reference data-preparation script for the mteb/ActivityNet_Captions_val2 dataset.

REFERENCE ONLY. Not runnable end-to-end: the upstream multi-part tarball
is hosted on a third-party HuggingFace mirror. The
canonical artifact produced by this pipeline is the
`mteb/ActivityNet_Captions_val2` dataset on the HuggingFace Hub.

Source
------
Krishna et al., "Dense-Captioning Events in Videos" (ICCV 2017).
    https://arxiv.org/abs/1705.00754

Upstream artifacts:
    https://huggingface.co/datasets/friedrichor/ActivityNet_Captions
    ActivityNet_Videos.tar.part-000 ... part-007 (concatenated to a single tar)

MVEB-specific processing
------------------------
1. Concatenate the 8 ActivityNet_Videos tar parts and selectively extract
   only the mp4 clips referenced by the `val2` split of
   `friedrichor/ActivityNet_Captions`.
2. Pair each available local clip with its dense `caption` field.
3. Deduplicate by caption (keep first occurrence) to produce the final
   retrieval pool.
4. Schema: {video, caption}. No audio (video-only retrieval). Single
   `test` split (sourced from upstream `val2`).
5. Final size: ~4,884 rows in the single `test` split.
"""

from __future__ import annotations

from pathlib import Path

from datasets import Dataset, Features, Value, Video, load_dataset

SOURCE_REPO = "friedrichor/ActivityNet_Captions"
SOURCE_SPLIT = "val2"
VIDEO_ROOT = Path("Activity_Videos")
TARGET_REPO = "mteb/ActivityNet_Captions_val2"


def write_extract_list(ds, output_file: str) -> None:
    with open(output_file, "w") as f:
        for item in ds:
            f.write(f"{VIDEO_ROOT}/{item['video']}\n")


def build_caption_map(ds) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in ds:
        video_path = VIDEO_ROOT / item["video"]
        if video_path.exists():
            mapping[str(video_path)] = item["caption"]
    return mapping


def generator(video_list: list[str], mapping: dict[str, str]):
    for vp in video_list:
        yield {"video": vp, "caption": mapping[vp]}


def deduplicate_by_caption(ds: Dataset) -> Dataset:
    seen: set[str] = set()
    unique_indices: list[int] = []
    for i, caption in enumerate(ds["caption"]):
        if caption not in seen:
            seen.add(caption)
            unique_indices.append(i)
    return ds.select(unique_indices)


def main() -> None:
    raw = load_dataset(SOURCE_REPO, split=SOURCE_SPLIT)
    write_extract_list(raw, "wanted_videos.txt")
    # Upstream: cat ActivityNet_Videos.tar.part-* > ActivityNet_Videos.tar
    # Upstream: tar -xf ActivityNet_Videos.tar -T wanted_videos.txt

    caption_map = build_caption_map(raw)

    features = Features({
        "video": Video(),
        "caption": Value("string"),
    })

    test_ds = Dataset.from_generator(
        generator,
        gen_kwargs={"video_list": list(caption_map.keys()), "mapping": caption_map},
        features=features,
    )

    ds_unique = deduplicate_by_caption(test_ds)
    ds_unique.push_to_hub(TARGET_REPO, split="test")


if __name__ == "__main__":
    main()
