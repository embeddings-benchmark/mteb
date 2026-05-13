"""Reference data-preparation script for the mteb/Daily-Omni dataset.

REFERENCE ONLY. Not runnable end-to-end: it depends on the upstream
`liarliar/Daily-Omni` repo on the HuggingFace Hub for both the video
archive (`Videos.tar`) and the QA metadata. The canonical artifact
produced by this pipeline is the `mteb/Daily-Omni` dataset on the
HuggingFace Hub.

Source
------
Zhou et al., "Daily-Omni: Towards Audio-Visual Reasoning with Temporal
Alignment across Modalities" (2025). https://arxiv.org/abs/2505.17862

Upstream artifacts:
    https://huggingface.co/datasets/liarliar/Daily-Omni
    https://huggingface.co/datasets/liarliar/Daily-Omni/resolve/main/Videos.tar

MVEB-specific processing
------------------------
1. Download and untar `Videos.tar` from `liarliar/Daily-Omni`, then load
   the default `train` split (the upstream release exposes the evaluation
   set under the `train` split name).
2. Index extracted mp4 files by basename and pair each QA item with its
   `{video_id}_video.mp4` clip. Resolve the letter answer (A/B/C/D) to
   the corresponding `Choice` string. Key by `(video_path, question)` so
   multiple questions on the same clip are preserved.
3. Extract mono 16 kHz PCM audio with ffmpeg. Drop questions whose audio
   extraction fails.
4. Schema: {video_id, video, audio, question, candidates, answer}.
   Single `test` split.
5. Final size: ~1,200 rows in the single `test` split.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from datasets import Audio, Dataset, Features, Sequence, Value, Video, load_dataset

SOURCE_REPO = "liarliar/Daily-Omni"
VIDEO_ROOT = Path("Videos")
TARGET_REPO = "mteb/Daily-Omni"

LETTER_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}


def index_videos() -> dict[str, Path]:
    return {p.name: p for p in VIDEO_ROOT.rglob("*.mp4") if p.is_file()}


def build_question_map(ds, video_index: dict[str, Path]) -> dict[tuple[str, str], dict]:
    mapping: dict[tuple[str, str], dict] = {}
    for item in ds:
        filename = f"{item['video_id']}_video.mp4"
        if filename not in video_index:
            continue
        video_path = str(video_index[filename])
        candidates = item["Choice"]
        answer = candidates[LETTER_TO_INDEX[item["Answer"]]]
        mapping[(video_path, item["Question"])] = {
            "video_id": item["video_id"],
            "question": item["Question"],
            "candidates": candidates,
            "answer": answer,
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
    for video_path, question in keys:
        wav = extract_audio_16k_mono(Path(video_path))
        if wav is None:
            continue
        meta = mapping[(video_path, question)]
        yield {
            "video_id": str(meta["video_id"]),
            "video": video_path,
            "audio": str(wav),
            "question": meta["question"],
            "candidates": meta["candidates"],
            "answer": meta["answer"],
        }


def main() -> None:
    raw = load_dataset(SOURCE_REPO, split="train")
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
        gen_kwargs={"keys": list(question_map.keys()), "mapping": question_map},
        features=features,
    )

    test_ds.push_to_hub(TARGET_REPO, split="test")


if __name__ == "__main__":
    main()
