"""Reference data-preparation script for the mteb/PerceptionTest_val dataset.

REFERENCE ONLY. Not runnable end-to-end: it depends on the
`lmms-lab/PerceptionTest_Val` videos archives on HuggingFace and ffmpeg. The canonical
artifact produced by this pipeline is the `mteb/PerceptionTest_val`
dataset on the HuggingFace Hub.

Source
------
Patraucean et al., "Perception Test: A Diagnostic Benchmark for
Multimodal Video Models" (NeurIPS 2023).
    https://arxiv.org/abs/2305.13786
    https://github.com/google-deepmind/perception_test

Upstream mirrors used:
    https://huggingface.co/datasets/lmms-lab/PerceptionTest_Val
    https://huggingface.co/datasets/lmms-lab/PerceptionTest (mc_question_val)

MVEB-specific processing
------------------------
1. Download `videos_chunked_02.zip` from `lmms-lab/PerceptionTest_Val`
   and load the `mc_question_val` validation split from
   `lmms-lab/PerceptionTest` for multiple-choice annotations.
2. Build the list of `videos/<video_name>.mp4` paths that the annotations
   reference, and selectively extract only those entries from the chunked
   zip (avoids unpacking the full archive).
3. For each surviving QA item, resolve `<video_name>.mp4` to a local
   path. The answer is the option indexed by `answer_id`.
4. Extract mono 16 kHz PCM audio with ffmpeg. Skip rows where extraction
   fails.
5. Schema: {question_id, video_id, video, audio, question, candidates,
   answer}.
6. Push as a single `test` split to `mteb/PerceptionTest_val`.
7. Final size: ~938 rows in the single `test` split.
"""

from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

from datasets import Audio, Dataset, Features, Sequence, Value, Video, load_dataset

ANNOTATION_REPO = "lmms-lab/PerceptionTest"
VIDEO_ARCHIVE_URL = (
    "https://huggingface.co/datasets/lmms-lab/PerceptionTest_Val/resolve/main/"
    "videos_chunked_02.zip"
)
VIDEO_ARCHIVE = "videos_chunked_02.zip"
VIDEO_ROOT = Path("videos")
TARGET_REPO = "mteb/PerceptionTest_val"


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


def extract_wanted_videos(ds, archive: str, output_dir: str) -> None:
    wanted = {f"videos/{item['video_name']}.mp4" for item in ds}
    with zipfile.ZipFile(archive) as z:
        present = wanted & set(z.namelist())
        for file_path in present:
            z.extract(file_path, output_dir)


def build_qa_map(ds, video_index: dict[str, Path]) -> dict[tuple[str, str], dict]:
    qa_map: dict[tuple[str, str], dict] = {}
    for item in ds:
        filename = f"{item['video_name']}.mp4"
        if filename not in video_index:
            continue
        candidates = item["options"]
        answer = candidates[int(item["answer_id"])]
        vp = video_index[filename]
        qa_map[(str(vp), item["question_id"])] = {
            "question_id": item["question_id"],
            "video_id": item["video_name"],
            "question": item["question"],
            "candidates": candidates,
            "answer": answer,
        }
    return qa_map


def main() -> None:
    subprocess.run(["wget", VIDEO_ARCHIVE_URL], check=True)

    ds = load_dataset(ANNOTATION_REPO, name="mc_question_val", split="validation")
    extract_wanted_videos(ds, VIDEO_ARCHIVE, ".")

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
        for (vp, _qid), row in qa_map.items():
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
