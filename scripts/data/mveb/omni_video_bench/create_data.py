"""Reference data-preparation script for the mteb/OmniVideoBench_subset dataset.

REFERENCE ONLY. Not runnable end-to-end: it depends on the upstream
`NJU-LINK/OmniVideoBench` HuggingFace dataset (videos shipped as ~30-minute
mp4 assets) and helper binaries (ffmpeg). The
canonical artifact produced by this pipeline is the
`mteb/OmniVideoBench_subset` dataset on the HuggingFace Hub.

Source
------
Ye et al., "OmniVideoBench: Towards Audio-Visual Understanding Evaluation
for Omni MLLMs" (2025).
    https://arxiv.org/abs/2510.10689
    https://huggingface.co/datasets/NJU-LINK/OmniVideoBench

MVEB-specific processing
------------------------
1. Pull every mp4 in `NJU-LINK/OmniVideoBench` via `snapshot_download` and
   load the `test` split of QA annotations.
2. Keep only items whose `duration` field's leading two characters parse as
   minutes < 5 (clips under 5 minutes).
3. Resolve each QA item's `video` field to a local mp4 path, decode the
   letter-indexed `correct_option` ("A".."D") into the chosen string from
   `options`, and emit one row per surviving (video, question) pair.
4. Extract mono 16 kHz PCM audio with ffmpeg. Skip rows where extraction
   fails.
5. Schema: {video_id, video, audio, question, candidates, answer}.
6. Push as a single `test` split to `mteb/OmniVideoBench_subset`.
7. Final size: ~509 rows in the single `test` split.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from datasets import Audio, Dataset, Features, Sequence, Value, Video, load_dataset
from huggingface_hub import snapshot_download

SOURCE_REPO = "NJU-LINK/OmniVideoBench"
VIDEO_ROOT = Path("videos")
MAX_DURATION_MIN = 5
LETTER_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}
TARGET_REPO = "mteb/OmniVideoBench_subset"


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


def build_qa_map(ds, video_index: dict[str, Path]) -> dict[tuple[str, str], dict]:
    qa_map: dict[tuple[str, str], dict] = {}
    for item in ds:
        filename = str(item["video"]).split("/")[-1]
        if filename not in video_index:
            continue
        candidates = item["options"]
        answer = candidates[LETTER_TO_INDEX[item["correct_option"]]]
        vp = video_index[filename]
        qa_map[(str(vp), item["question"])] = {
            "video_id": item["video"],
            "question": item["question"],
            "candidates": candidates,
            "answer": answer,
        }
    return qa_map


def main() -> None:
    snapshot_download(
        repo_id=SOURCE_REPO,
        repo_type="dataset",
        local_dir=str(VIDEO_ROOT),
        allow_patterns="*.mp4",
    )

    ds = load_dataset(SOURCE_REPO, split="test")
    ds = ds.filter(lambda x: int(x["duration"][:2]) < MAX_DURATION_MIN)

    video_index = {p.name: p for p in VIDEO_ROOT.rglob("*.mp4") if p.is_file()}
    qa_map = build_qa_map(ds, video_index)

    features = Features({
        "video_id": Value("string"),
        "video": Video(),
        "audio": Audio(sampling_rate=16000),
        "question": Value("string"),
        "candidates": Sequence(Value("string")),
        "answer": Value("string"),
    })

    def gen():
        for (vp, _question), row in qa_map.items():
            wav = extract_audio_16k_mono(Path(vp))
            if wav is None:
                continue
            yield {
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
