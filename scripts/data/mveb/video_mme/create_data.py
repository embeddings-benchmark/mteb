"""Reference data-preparation script for the mteb/Video-MME_short dataset.

REFERENCE ONLY. Not runnable end-to-end: it depends on the
`lmms-lab/Video-MME` chunked video archives on HuggingFace (20 zip files)
and ffmpeg. The canonical artifact
produced by this pipeline is the `mteb/Video-MME_short` dataset on the
HuggingFace Hub.

Source
------
Fu et al., "Video-MME: The First-Ever Comprehensive Evaluation Benchmark
of Multi-modal LLMs in Video Analysis" (CVPR 2025).
    https://arxiv.org/abs/2405.21075
    https://github.com/MME-Benchmarks/Video-MME

Upstream mirror used:
    https://huggingface.co/datasets/lmms-lab/Video-MME

MVEB-specific processing
------------------------
1. Download every `videos_chunked_*.zip` shard from `lmms-lab/Video-MME`
   (the upstream split videos across 20 chunks).
2. Load the `test` split of QA annotations and keep only items whose
   `duration` field equals `"short"` (the 11s-2min bucket). Long and
   medium clips are dropped. After a deterministic shuffle (seed=42)
   take up to 2000 short clips.
3. Build the list of `data/<videoID>.mp4` paths the annotations need and
   selectively extract those entries from each chunked zip.
4. For each QA item resolve `<videoID>.mp4` and decode the letter answer
   ("A".."D") into the chosen option from `options`.
5. Extract mono 16 kHz PCM audio with ffmpeg. Skip rows where extraction
   fails.
6. Schema: {question_id, video_id, video, audio, question, candidates,
   answer}.
7. Push as a single `test` split to `mteb/Video-MME_short`.
8. Final size: ~900 rows.
"""

from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

from datasets import Audio, Dataset, Features, Sequence, Value, Video, load_dataset

SOURCE_REPO = "lmms-lab/Video-MME"
VIDEO_ARCHIVE_URL = (
    "https://huggingface.co/datasets/lmms-lab/Video-MME/resolve/main/"
    "videos_chunked_{idx:02d}.zip"
)
NUM_CHUNKS = 20
VIDEO_ROOT = Path("data")
TARGET_TOTAL = 2000
SHUFFLE_SEED = 42
LETTER_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}
TARGET_REPO = "mteb/Video-MME_short"


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


def select_short_subset(ds) -> "Dataset":
    shuffled = ds.shuffle(seed=SHUFFLE_SEED)
    filtered = shuffled.filter(lambda x: x["duration"] == "short")
    keep = min(TARGET_TOTAL, len(filtered))
    return filtered.select(range(keep))


def extract_wanted_videos(ds, output_dir: str) -> None:
    wanted = {f"data/{item['videoID']}.mp4" for item in ds}
    for idx in range(1, NUM_CHUNKS + 1):
        archive = f"videos_chunked_{idx:02d}.zip"
        if not Path(archive).exists():
            continue
        with zipfile.ZipFile(archive) as z:
            present = wanted & set(z.namelist())
            for file_path in present:
                z.extract(file_path, output_dir)


def build_qa_map(ds, video_index: dict[str, Path]) -> dict[str, dict]:
    qa_map: dict[str, dict] = {}
    for item in ds:
        filename = f"{item['videoID']}.mp4"
        if filename not in video_index:
            continue
        candidates = item["options"]
        answer = candidates[LETTER_TO_INDEX[item["answer"]]]
        qa_map[item["question_id"]] = {
            "question_id": item["question_id"],
            "video_id": item["videoID"],
            "video_path": str(video_index[filename]),
            "question": item["question"],
            "candidates": candidates,
            "answer": answer,
        }
    return qa_map


def main() -> None:
    for idx in range(1, NUM_CHUNKS + 1):
        subprocess.run(["wget", VIDEO_ARCHIVE_URL.format(idx=idx)], check=True)

    ds = load_dataset(SOURCE_REPO, split="test")
    ds = select_short_subset(ds)
    extract_wanted_videos(ds, ".")

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
        for row in qa_map.values():
            wav = extract_audio_16k_mono(Path(row["video_path"]))
            if wav is None:
                continue
            yield {
                "question_id": row["question_id"],
                "video_id": row["video_id"],
                "video": row["video_path"],
                "audio": str(wav),
                "question": row["question"],
                "candidates": row["candidates"],
                "answer": row["answer"],
            }

    test_ds = Dataset.from_generator(gen, features=features)
    test_ds.push_to_hub(TARGET_REPO, split="test")


if __name__ == "__main__":
    main()
