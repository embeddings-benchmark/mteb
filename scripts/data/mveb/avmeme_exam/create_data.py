"""Reference data-preparation script for the mteb/AVMeme-Exam dataset.

REFERENCE ONLY. Not runnable end-to-end: it depends on the upstream
`naplab/AVMeme-Exam` HuggingFace dataset (mp4 clips shipped as a `clips.zip`
archive) and helper binaries (ffmpeg). The canonical
artifact produced by this pipeline is the `mteb/AVMeme-Exam` dataset on the
HuggingFace Hub.

Source
------
Mei et al., "AVMeme Exam: A Multimodal Multilingual Multicultural
Benchmark for LLMs' Contextual and Cultural Knowledge and Thinking" (2026).
    https://arxiv.org/abs/2601.17645
    https://huggingface.co/datasets/naplab/AVMeme-Exam

MVEB-specific processing
------------------------
1. Download `clips.zip` from `naplab/AVMeme-Exam` via `snapshot_download`,
   unzip locally, and load the `full`/`test` split of QA annotations.
2. Resolve each row's `video_path` to a local mp4 under `clips/video/`.
   The annotation's `solution` field is taken verbatim as the answer
   string. The choice list comes from `choices`.
3. Preserve metadata for filtering / analysis:
   `summary`, `usage`, the first entry of the `emotion` array,
   `sensitivity` list, `category`, `question_type`, and `language`. These
   four label fields are wrapped as ClassLabels.
4. Extract mono 16 kHz PCM audio with ffmpeg. Skip rows where extraction
   fails.
5. Schema: {video_id, video, audio, language, summary, emotion, question,
   candidates, answer, usage, category, question_type, sensitivity}.
6. Push as a single `test` split to `mteb/AVMeme-Exam`.
7. Final size: ~900 rows in the single `test` split.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from datasets import (
    Audio,
    ClassLabel,
    Dataset,
    Features,
    Sequence,
    Value,
    Video,
    load_dataset,
)
from huggingface_hub import snapshot_download

SOURCE_REPO = "naplab/AVMeme-Exam"
VIDEO_ROOT = Path("clips/video")
TARGET_REPO = "mteb/AVMeme-Exam"


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


def build_qa_map(ds, video_index: dict[str, Path]) -> dict[str, dict]:
    qa_map: dict[str, dict] = {}
    for item in ds:
        filename = str(item["video_path"]).split("/")[-1]
        if filename not in video_index:
            continue
        vp = video_index[filename]
        qa_map[str(vp)] = {
            "video_id": filename,
            "question": item["question"],
            "candidates": item["choices"],
            "answer": item["solution"],
            "language": item["language"],
            "summary": item["summary"],
            "usage": item["usage"],
            "emotion": item["emotion"][0],
            "sensitivity": item["sensitivity"],
            "category": item["category"],
            "question_type": item["question_type"],
        }
    return qa_map


def main() -> None:
    snapshot_download(
        repo_id=SOURCE_REPO,
        repo_type="dataset",
        local_dir=".",
        allow_patterns="*.zip",
    )
    subprocess.run(["unzip", "-o", "clips.zip"], check=True)

    ds = load_dataset(SOURCE_REPO, "full", split="test")

    video_index = {p.name: p for p in VIDEO_ROOT.rglob("*.mp4") if p.is_file()}
    qa_map = build_qa_map(ds, video_index)

    categories = sorted({r["category"] for r in qa_map.values()})
    question_types = sorted({r["question_type"] for r in qa_map.values()})
    emotions = sorted({r["emotion"] for r in qa_map.values()})
    languages = sorted({r["language"] for r in qa_map.values()})

    features = Features({
        "video_id": Value("string"),
        "video": Video(),
        "audio": Audio(sampling_rate=16000),
        "language": ClassLabel(names=languages),
        "summary": Value("string"),
        "emotion": ClassLabel(names=emotions),
        "question": Value("string"),
        "candidates": Sequence(Value("string")),
        "answer": Value("string"),
        "usage": Value("string"),
        "category": ClassLabel(names=categories),
        "question_type": ClassLabel(names=question_types),
        "sensitivity": Sequence(Value("string")),
    })

    def gen():
        for vp, row in qa_map.items():
            wav = extract_audio_16k_mono(Path(vp))
            if wav is None:
                continue
            yield {
                "video_id": row["video_id"],
                "video": vp,
                "audio": str(wav),
                "language": row["language"],
                "summary": row["summary"],
                "emotion": row["emotion"],
                "question": row["question"],
                "candidates": row["candidates"],
                "answer": row["answer"],
                "usage": row["usage"],
                "sensitivity": row["sensitivity"],
                "category": row["category"],
                "question_type": row["question_type"],
            }

    test_ds = Dataset.from_generator(gen, features=features)
    test_ds.push_to_hub(TARGET_REPO, split="test")


if __name__ == "__main__":
    main()
