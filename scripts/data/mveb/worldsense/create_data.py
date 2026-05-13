"""Reference data-preparation script for the mteb/WorldSense_1min dataset.

REFERENCE ONLY. Not runnable end-to-end: it depends on the
`lmms-lab/WorldSense` chunked video archives on HuggingFace (four zip
files) and ffmpeg. The canonical artifact produced
by this pipeline is the `mteb/WorldSense_1min` dataset on the
HuggingFace Hub.

Source
------
Hong et al., "WorldSense: Evaluating Real-world Omnimodal Understanding
for Multimodal LLMs" (2025).
    https://arxiv.org/abs/2502.04326
    https://github.com/JaaackHongggg/WorldSense

Upstream mirror used:
    https://huggingface.co/datasets/lmms-lab/WorldSense

MVEB-specific processing
------------------------
1. Download and unpack all four `videos_chunk_00{1..4}.zip` shards from
   `lmms-lab/WorldSense`.
2. Load the `test` split of QA annotations and keep only items whose
   `duration` field equals `"<1min"` (the sub-minute bucket).
3. For each surviving QA item resolve `<video>.mp4` to a local path and
   decode the letter answer ("A".."D") into the chosen option from
   `candidates`. Keep `index`, `video_caption` (= video id), and `domain`.
4. Extract mono 16 kHz PCM audio with ffmpeg. Skip rows where extraction
   fails.
5. Schema: {index, video_id, video, audio, video_caption, domain,
   question, candidates, answer}. `domain` is a ClassLabel over `domain`
   values in the subset.
6. Push as a single `test` split to `mteb/WorldSense_1min`.
7. Final size: ~1,047 rows in the single `test` split.
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

SOURCE_REPO = "lmms-lab/WorldSense"
VIDEO_ARCHIVE_URL = (
    "https://huggingface.co/datasets/lmms-lab/WorldSense/resolve/main/"
    "videos_chunk_{idx:03d}.zip"
)
NUM_CHUNKS = 4
VIDEO_ROOT = Path("videos")
DURATION_KEEP = "<1min"
LETTER_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}
TARGET_REPO = "mteb/WorldSense_1min"


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
        filename = f"{item['video']}.mp4"
        if filename not in video_index:
            continue
        candidates = item["candidates"]
        answer = candidates[LETTER_TO_INDEX[item["answer"]]]
        vp = video_index[filename]
        qa_map[(str(vp), str(item["index"]))] = {
            "index": item["index"],
            "video_id": item["video"],
            "video_caption": item["video"],
            "domain": item["domain"],
            "question": item["question"],
            "candidates": candidates,
            "answer": answer,
        }
    return qa_map


def main() -> None:
    for idx in range(1, NUM_CHUNKS + 1):
        url = VIDEO_ARCHIVE_URL.format(idx=idx)
        subprocess.run(["wget", url], check=True)
        subprocess.run(["unzip", "-o", f"videos_chunk_{idx:03d}.zip"], check=True)

    ds = load_dataset(SOURCE_REPO, split="test")
    ds = ds.filter(lambda x: x["duration"] == DURATION_KEEP)

    video_index = {p.name: p for p in VIDEO_ROOT.rglob("*") if p.is_file()}
    qa_map = build_qa_map(ds, video_index)

    domains = sorted({r["domain"] for r in qa_map.values()})

    features = Features({
        "index": Value("string"),
        "video_id": Value("string"),
        "video": Video(),
        "audio": Audio(sampling_rate=16000),
        "video_caption": Value("string"),
        "domain": ClassLabel(names=domains),
        "question": Value("string"),
        "candidates": Sequence(Value("string")),
        "answer": Value("string"),
    })

    def gen():
        for (vp, _idx), row in qa_map.items():
            wav = extract_audio_16k_mono(Path(vp))
            if wav is None:
                continue
            yield {
                "index": row["index"],
                "video_id": row["video_id"],
                "video": vp,
                "audio": str(wav),
                "video_caption": row["video_caption"],
                "domain": row["domain"],
                "question": row["question"],
                "candidates": row["candidates"],
                "answer": row["answer"],
            }

    test_ds = Dataset.from_generator(gen, features=features)
    test_ds.push_to_hub(TARGET_REPO, split="test")


if __name__ == "__main__":
    main()
