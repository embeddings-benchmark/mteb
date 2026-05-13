"""Reference data-preparation script for the mteb/VALOR-32K dataset.

REFERENCE ONLY. Not runnable end-to-end: the upstream video tarball is
hosted at a third-party HuggingFace mirror. The
canonical artifact produced by this pipeline is the `mteb/VALOR-32K`
dataset on the HuggingFace Hub.

Source
------
Liu et al., "VALOR: Vision-Audio-Language Omni-Perception Pretraining
Model and Dataset" (TPAMI 2024). https://arxiv.org/abs/2304.08345

Upstream artifacts:
    https://huggingface.co/datasets/UnFaZeD07/VALOR-32K (mirror with mp4s)
    desc_test.json (official VALOR-32K test split descriptions)

MVEB-specific processing
------------------------
1. Build a file list of `{video_id}.mp4` paths for every entry in the
   official `desc_test.json` test split, then selectively extract them
   from `VALOR-32K-videos.tar`.
2. Load the `UnFaZeD07/VALOR-32K` `train` config (which contains the test
   set captions as the `desc` field) and pair each available local clip
   with its description.
3. Extract mono 16 kHz PCM audio with ffmpeg. Drop clips where extraction
   fails.
4. Deduplicate by description (keep first occurrence) to produce the
   final audio-visual retrieval pool.
5. Schema: {video, audio, description}. Single `test` split.
6. Final size: ~3,491 rows in the single `test` split.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from datasets import Audio, Dataset, Features, Value, Video, load_dataset

SOURCE_REPO = "UnFaZeD07/VALOR-32K"
DESC_JSON = "desc_test.json"
VIDEO_ROOT = Path("raid/datasets/audioset/valor_videos")
TARGET_REPO = "mteb/VALOR-32K"


def write_extract_list(output_file: str) -> None:
    with open(DESC_JSON) as f:
        data = json.load(f)
    allowed_ids = {item["video_id"] for item in data}
    with open(output_file, "w") as f:
        for video_id in allowed_ids:
            f.write(f"{VIDEO_ROOT}/{video_id}.mp4\n")


def build_description_map(ds) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in ds:
        video_path = VIDEO_ROOT / f"{item['video_id']}.mp4"
        if video_path.exists():
            mapping[str(video_path)] = item["desc"]
    return mapping


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


def generator(video_list: list[str], mapping: dict[str, str]):
    for vp in video_list:
        wav = extract_audio_16k_mono(Path(vp))
        if wav is None:
            continue
        yield {"video": vp, "audio": str(wav), "description": mapping[vp]}


def deduplicate_by_description(ds: Dataset) -> Dataset:
    seen: set[str] = set()
    unique_indices: list[int] = []
    for i, desc in enumerate(ds["description"]):
        if desc not in seen:
            seen.add(desc)
            unique_indices.append(i)
    return ds.select(unique_indices)


def main() -> None:
    write_extract_list("wanted_videos.txt")
    # Upstream: tar -xf VALOR-32K-videos.tar -T wanted_videos.txt

    raw = load_dataset(SOURCE_REPO, split="train")
    desc_map = build_description_map(raw)

    features = Features({
        "video": Video(),
        "audio": Audio(sampling_rate=16000),
        "description": Value("string"),
    })

    test_ds = Dataset.from_generator(
        generator,
        gen_kwargs={"video_list": list(desc_map.keys()), "mapping": desc_map},
        features=features,
    )

    ds_unique = deduplicate_by_description(test_ds)
    ds_unique.push_to_hub(TARGET_REPO, split="test")


if __name__ == "__main__":
    main()
