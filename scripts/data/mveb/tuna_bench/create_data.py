"""Reference data-preparation script for the mteb/TUNA-Bench_1K dataset.

REFERENCE ONLY. Not runnable end-to-end: it depends on the upstream
`friedrichor/TUNA-Bench` repo on the HuggingFace Hub (videos.zip and
metadata). The canonical artifact produced by this pipeline is the
`mteb/TUNA-Bench_1K` dataset on the HuggingFace Hub.

Source
------
Kong et al., "TUNA: Comprehensive Fine-grained Temporal Understanding
Evaluation on Dense Dynamic Videos" (CVPR 2025).
    https://arxiv.org/abs/2505.20124

Upstream artifacts:
    https://huggingface.co/datasets/friedrichor/TUNA-Bench
    https://huggingface.co/datasets/friedrichor/TUNA-Bench/resolve/main/videos.zip

MVEB-specific processing
------------------------
1. Download and unzip `videos.zip` from `friedrichor/TUNA-Bench` to obtain
   the raw mp4 clips, and load the `TUNA-1K` config / `test` split for
   metadata.
2. Pair each locally available `{video_id}.mp4` with its dense `caption`.
3. Deduplicate by caption (keep first occurrence) to produce the final
   retrieval pool of ~1K rows.
4. Schema: {video, caption}. No audio (video-only retrieval). Single
   `test` split.
"""

from __future__ import annotations

from pathlib import Path

from datasets import Dataset, Features, Value, Video, load_dataset

SOURCE_REPO = "friedrichor/TUNA-Bench"
SOURCE_CONFIG = "TUNA-1K"
VIDEO_ROOT = Path("video")
TARGET_REPO = "mteb/TUNA-Bench_1K"


def build_caption_map(ds) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in ds:
        video_path = VIDEO_ROOT / f"{item['video']}.mp4"
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
    raw = load_dataset(SOURCE_REPO, name=SOURCE_CONFIG, split="test")
    caption_map = build_caption_map(raw)
    video_list = list(caption_map.keys())

    features = Features({
        "video": Video(),
        "caption": Value("string"),
    })

    test_ds = Dataset.from_generator(
        generator,
        gen_kwargs={"video_list": video_list, "mapping": caption_map},
        features=features,
    )

    ds_unique = deduplicate_by_caption(test_ds)
    ds_unique.push_to_hub(TARGET_REPO, split="test")


if __name__ == "__main__":
    main()
