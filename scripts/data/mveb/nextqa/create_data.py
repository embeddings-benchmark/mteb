"""Reference data-preparation script for the mteb/NExT-QA dataset.

REFERENCE ONLY. Not runnable end-to-end: it depends on the upstream
`VLM2Vec/NExTQA` videos archive and the `Shuaiii/NExT-QA` annotation mirror
on HuggingFace, plus ffmpeg. The canonical
artifact produced by this pipeline is the `mteb/NExT-QA` dataset on the
HuggingFace Hub.

Source
------
Xiao, Shang, Yao, Chua, "NExT-QA: Next Phase of Question-Answering to
Explaining Temporal Actions" (CVPR 2021).
    https://arxiv.org/abs/2105.08276
    https://github.com/doc-doc/NExT-QA

Upstream mirrors used:
    https://huggingface.co/datasets/VLM2Vec/NExTQA (videos.zip)
    https://huggingface.co/datasets/Shuaiii/NExT-QA (QA annotations)

MVEB-specific processing
------------------------
1. Download `videos.zip` from `VLM2Vec/NExTQA` and unzip the
   `NExTVideo/` tree of mp4 clips.
2. Load the `test` split of `Shuaiii/NExT-QA` for the multiple-choice
   annotations (`qid`, `video`, `question`, `options`, `answer`).
3. Resolve each row's `video` id to a local `<video>.mp4` path. The
   answer is the option indexed by the integer `answer` field.
4. Extract mono 16 kHz PCM audio with ffmpeg. Skip rows where extraction
   fails.
5. Schema: {question_id, video_id, video, audio, question, candidates,
   answer}.
6. Push as a single `test` split to `mteb/NExT-QA`.
7. Final size: ~993 rows in the single `test` split.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from datasets import Audio, Dataset, Features, Sequence, Value, Video, load_dataset

ANNOTATION_REPO = "Shuaiii/NExT-QA"
VIDEO_ARCHIVE_URL = (
    "https://huggingface.co/datasets/VLM2Vec/NExTQA/resolve/main/videos.zip"
)
VIDEO_ROOT = Path("NExTVideo")
TARGET_REPO = "mteb/NExT-QA"


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
        filename = f"{item['video']}.mp4"
        if filename not in video_index:
            continue
        candidates = item["options"]
        answer = candidates[int(item["answer"])]
        vp = video_index[filename]
        qa_map[str(vp)] = {
            "question_id": item["qid"],
            "video_id": item["video"],
            "question": item["question"],
            "candidates": candidates,
            "answer": answer,
        }
    return qa_map


def main() -> None:
    subprocess.run(["wget", VIDEO_ARCHIVE_URL], check=True)
    subprocess.run(["unzip", "-o", "videos.zip"], check=True)

    ds = load_dataset(ANNOTATION_REPO, split="test")

    video_index = {p.name: p for p in VIDEO_ROOT.rglob("*") if p.is_file()}
    qa_map = build_qa_map(ds, video_index)

    features = Features({
        "question_id": Value("string"),
        "video_id": Value("string"),
        "video": Video(),
        "audio": Audio(sampling_rate=16000),
        "question": Value("string"),
        "candidates": Sequence(Value("string")),
        "answer": Value("string"),
    })

    def gen():
        for vp, row in qa_map.items():
            wav = extract_audio_16k_mono(Path(vp))
            if wav is None:
                continue
            yield {
                "question_id": row["question_id"],
                "video_id": row["video_id"],
                "video": vp,
                "audio": str(wav),
                "question": row["question"],
                "candidates": row["candidates"],
                "answer": row["answer"],
            }

    test_ds = Dataset.from_generator(gen, features=features)
    test_ds.push_to_hub(TARGET_REPO, split="test")


if __name__ == "__main__":
    main()
