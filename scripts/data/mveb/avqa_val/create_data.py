"""Reference data-preparation script for the mteb/AVQA_val dataset.

REFERENCE ONLY. Not runnable end-to-end. The canonical artifact is
`mteb/AVQA_val` on the HuggingFace Hub.

Source
------
AVQA: Yang et al., "AVQA: A Dataset for Audio-Visual Question Answering
on Videos" (ACM MM 2022). https://mn.cs.tsinghua.edu.cn/avqa/

Mirrors used:
    Videos:    https://huggingface.co/datasets/Jayson236/AVQA_ddd
               (video_part01.zip)
    Test QA:   https://huggingface.co/datasets/gwkrsrch2/avqa_2025
               (split=test)

MVEB-specific processing
------------------------
1. Download `video_part01.zip` from the Jayson236 AVQA mirror; unzip into
   the local `Videos/` tree.
2. Load `gwkrsrch2/avqa_2025` test split as the QA source.
3. For each QA row, locate `{video_id}_video.mp4` under the unzipped tree
   and key the resulting record by (video_path, question) since a single
   video may have multiple questions.
4. Decode the letter answer (`A`/`B`/`C`/`D`) into the corresponding
   string in `Choice`.
5. Extract mono 16 kHz PCM audio from each clip via ffmpeg.
6. Schema: {video_id, video, audio, question, candidates, answer}.
   Final size: ~921 rows in the single `test` split.
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

VIDEO_ROOT = Path("Videos")
LETTER_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}
TARGET_REPO = "mteb/AVQA_val"


def index_videos(root: Path) -> dict[str, Path]:
    return {p.name: p for p in root.rglob("*.mp4") if p.is_file()}


def build_qa_map(ds, video_index: dict[str, Path]) -> dict[tuple[str, str], dict]:
    qa_map: dict[tuple[str, str], dict] = {}
    for item in ds:
        fname = f"{item['video_id']}_video.mp4"
        path = video_index.get(fname)
        if path is None:
            continue
        candidates = item["Choice"]
        answer = candidates[LETTER_TO_INDEX[item["Answer"]]]
        qa_map[(str(path), item["Question"])] = {
            "video_id": item["video_id"],
            "question": item["Question"],
            "candidates": candidates,
            "answer": answer,
        }
    return qa_map


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


def main() -> None:
    ds = load_dataset("gwkrsrch2/avqa_2025", split="test")
    video_index = index_videos(VIDEO_ROOT)
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
        for (video_path, _question), record in qa_map.items():
            wav = extract_audio_16k_mono(Path(video_path))
            if wav is None:
                continue
            yield {
                "video_id": record["video_id"],
                "video": video_path,
                "audio": str(wav),
                "question": record["question"],
                "candidates": record["candidates"],
                "answer": record["answer"],
            }

    test_ds = Dataset.from_generator(gen, features=features)
    test_ds.push_to_hub(TARGET_REPO, split="test")


if __name__ == "__main__":
    main()
