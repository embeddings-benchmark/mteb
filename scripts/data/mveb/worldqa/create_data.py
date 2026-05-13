"""Reference data-preparation script for the mteb/worldqa dataset.

REFERENCE ONLY. Not runnable end-to-end: the upstream `lmms-lab/worldqa`
repo on the HuggingFace Hub bundles the videos as a single `videos.zip`
snapshot. The canonical artifact produced by this pipeline is the
`mteb/worldqa` dataset on the HuggingFace Hub.

Source
------
Zhang et al., "WorldQA: Multimodal World Knowledge in Videos through
Long-Chain Reasoning" (2024). https://arxiv.org/abs/2405.03272

Upstream artifacts:
    https://huggingface.co/datasets/lmms-lab/worldqa
    https://huggingface.co/datasets/lmms-lab/worldqa/resolve/main/videos.zip

MVEB-specific processing
------------------------
1. Snapshot-download `videos.zip` from `lmms-lab/worldqa` and unzip to
   obtain the raw mp4 clips. Load the `MC` (multiple-choice) config /
   `test` split for the question metadata.
2. Index all extracted mp4 files by basename, then pair each question with
   its `{video_idx}.mp4` clip when present locally. Key questions by
   `(video_path, question_idx)` because a single video can underpin
   multiple questions.
3. Extract mono 16 kHz PCM audio with ffmpeg. Drop questions whose audio
   extraction fails.
4. Schema: {question_id, video_id, video, audio, question, candidates,
   answer}. Single `test` split.
5. Final size: ~3,284 rows in the single `test` split.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from datasets import Audio, Dataset, Features, Sequence, Value, Video, load_dataset

SOURCE_REPO = "lmms-lab/worldqa"
SOURCE_CONFIG = "MC"
VIDEO_ROOT = Path("videos")
TARGET_REPO = "mteb/worldqa"


def index_videos() -> dict[str, Path]:
    return {p.name: p for p in VIDEO_ROOT.rglob("*") if p.is_file()}


def build_question_map(ds, video_index: dict[str, Path]) -> dict[tuple[str, str], dict]:
    mapping: dict[tuple[str, str], dict] = {}
    for item in ds:
        filename = f"{item['video_idx']}.mp4"
        if filename not in video_index:
            continue
        video_path = str(video_index[filename])
        key = (video_path, str(item["question_idx"]))
        mapping[key] = {
            "question_id": item["question_idx"],
            "video_id": item["video_idx"],
            "question": item["question"],
            "candidates": item["option"],
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


def generator(keys, mapping: dict[tuple[str, str], dict]):
    for video_path, question_idx in keys:
        wav = extract_audio_16k_mono(Path(video_path))
        if wav is None:
            continue
        meta = mapping[(video_path, question_idx)]
        yield {
            "question_id": str(meta["question_id"]),
            "video_id": str(meta["video_id"]),
            "video": video_path,
            "audio": str(wav),
            "question": meta["question"],
            "candidates": meta["candidates"],
            "answer": meta["answer"],
        }


def main() -> None:
    raw = load_dataset(SOURCE_REPO, name=SOURCE_CONFIG, split="test")
    video_index = index_videos()
    question_map = build_question_map(raw, video_index)

    features = Features({
        "question_id": Value("string"),
        "video_id": Value("string"),
        "video": Video(),
        "audio": Audio(sampling_rate=16000),
        "question": Value("string"),
        "candidates": Sequence(Value("string")),
        "answer": Value("string"),
    })

    test_ds = Dataset.from_generator(
        generator,
        gen_kwargs={"keys": list(question_map.keys()), "mapping": question_map},
        features=features,
    )

    test_ds.push_to_hub(TARGET_REPO, split="test")


if __name__ == "__main__":
    main()
