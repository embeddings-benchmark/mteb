"""Reference data-preparation script for the mteb/MVBench dataset.

REFERENCE ONLY. Not runnable end-to-end: it depends on the upstream
`VLM2Vec/MVBench` repo on the HuggingFace Hub for both the QA metadata
and a collection of per-source video zips, plus auxiliary mirrors for
`tvqa` and `nturgbd` clips that are distributed separately.

The MTEB task module for MVBench is in-progress. The canonical artifact
is the `mteb/MVBench` dataset on the HuggingFace Hub, which publishes
all 20 MVBench task-family configs (~3,899 rows total) under the `test`
split. Each config is produced by re-running this script with a
different `SOURCE_CONFIG` (e.g. `unexpected_action`, `action_sequence`,
`object_existence`, ...).

Source
------
Li et al., "MVBench: A Comprehensive Multi-modal Video Understanding
Benchmark" (CVPR 2024). https://arxiv.org/abs/2311.17005

Upstream artifacts:
    https://huggingface.co/datasets/VLM2Vec/MVBench
        video/{FunQA_test,Moments_in_Time_Raw,clevrer,data0613,perception,
               scene_qa,ssv2_video,sta,star,tvqa,vlnqa}.zip
    https://huggingface.co/datasets/mengdeerer/tvqa     (tvqa.tar.gz)
    https://huggingface.co/datasets/VictorChen42/nturgbd (nturgbd.zip)

MVEB-specific processing
------------------------
1. Download every per-source zip referenced by MVBench from
   `VLM2Vec/MVBench` plus the `tvqa` and `nturgbd` auxiliary archives,
   then unpack them all into a single `videos/` tree.
2. Load the `unexpected_action` config / `train` split (MVBench exposes
   its evaluation rows under the `train` split name) from
   `VLM2Vec/MVBench`.
3. Index extracted files by basename, derive each QA item's filename
   from the last path component of its `video` field, and pair the two.
4. Extract mono 16 kHz PCM audio with ffmpeg for configs whose source
   clips carry an audio track.
5. Schema varies by config: {video_id, video, audio, question,
   candidates, answer} for audio-bearing configs (~half of the 20),
   {video_id, video, question, candidates, answer} for the video-only
   ones (e.g. synthetic clips from CLEVRER). Audio-bearing configs
   include the `audio` column; drop it for video-only configs.
6. Pushed under one config per task family / split `test`.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from datasets import (
    Audio,
    Dataset,
    Features,
    Sequence,
    Value,
    Video,
    load_dataset,
)

SOURCE_REPO = "VLM2Vec/MVBench"
SOURCE_CONFIG = "unexpected_action"
TARGET_CONFIG = "unexpected_actions"
VIDEO_ROOT = Path("videos")
TARGET_REPO = "mteb/MVBench"


def index_videos() -> dict[str, Path]:
    return {p.name: p for p in VIDEO_ROOT.rglob("*") if p.is_file()}


def build_question_map(ds, video_index: dict[str, Path]) -> dict[str, dict]:
    mapping: dict[str, dict] = {}
    for item in ds:
        filename = Path(item["video"]).name
        if filename not in video_index:
            continue
        mapping[str(video_index[filename])] = {
            "video_id": item["video"],
            "question": item["question"],
            "candidates": item["candidates"],
            "answer": item["answer"],
        }
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


def generator(video_list: list[str], mapping: dict[str, dict]):
    for vp in video_list:
        wav = extract_audio_16k_mono(Path(vp))
        if wav is None:
            continue
        meta = mapping[vp]
        yield {
            "video_id": meta["video_id"],
            "video": vp,
            "audio": str(wav),
            "question": meta["question"],
            "candidates": meta["candidates"],
            "answer": meta["answer"],
        }


def main() -> None:
    raw = load_dataset(SOURCE_REPO, name=SOURCE_CONFIG, split="train")
    video_index = index_videos()
    question_map = build_question_map(raw, video_index)

    features = Features({
        "video_id": Value("string"),
        "video": Video(),
        "audio": Audio(sampling_rate=16000),
        "question": Value("string"),
        "candidates": Sequence(Value("string")),
        "answer": Value("string"),
    })

    test_ds = Dataset.from_generator(
        generator,
        gen_kwargs={"video_list": list(question_map.keys()), "mapping": question_map},
        features=features,
    )

    test_ds.push_to_hub(TARGET_REPO, config_name=TARGET_CONFIG, split="test")


if __name__ == "__main__":
    main()
