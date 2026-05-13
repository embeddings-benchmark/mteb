"""Reference data-preparation script for the mteb/star_bench_val dataset.

REFERENCE ONLY. Not runnable end-to-end: it depends on the AI2 Charades
public S3 bucket (`Charades_v1_480.zip`) for the video clips and on the
`zechen-nlp/star_bench` repo on the HuggingFace Hub for the QA
metadata.

The MTEB task module for STAR is in-progress. The canonical artifact is
the `mteb/star_bench_val` dataset on the HuggingFace Hub, which
publishes all four STAR question-type configs (`feasibility`,
`interaction`, `prediction`, `sequence`; ~7,098 rows total) under the
`test` split. Each config is produced by re-running this script with a
different `SOURCE_CONFIG`.

Source
------
Wu et al., "STAR: A Benchmark for Situated Reasoning in Real-World
Videos" (NeurIPS 2021). https://arxiv.org/abs/2405.09711

Upstream artifacts:
    https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip
    https://huggingface.co/datasets/zechen-nlp/star_bench

MVEB-specific processing
------------------------
1. Download and unzip the AI2 Charades 480p mp4 archive (the STAR
   benchmark grounds its questions in Charades clips).
2. Load the `sequence` config / `validation` split from
   `zechen-nlp/star_bench`.
3. Each row's `video` field is a path of the form
   `Charades_v1_480/<id>.mp4`; take the basename and pair with the
   matching extracted clip when present locally.
4. Extract mono 16 kHz PCM audio with ffmpeg. Drop questions whose
   audio extraction fails.
5. Schema: {question_id, video_id, video, audio, question, candidates,
   answer}. Pushed under config `sequence` / split `test`.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from datasets import Audio, Dataset, Features, Sequence, Value, Video, load_dataset

SOURCE_REPO = "zechen-nlp/star_bench"
SOURCE_CONFIG = "sequence"
VIDEO_ROOT = Path("Charades_v1_480")
TARGET_REPO = "mteb/star_bench_val"


def index_videos() -> dict[str, Path]:
    return {p.name: p for p in VIDEO_ROOT.rglob("*") if p.is_file()}


def build_question_map(ds, video_index: dict[str, Path]) -> dict[str, dict]:
    mapping: dict[str, dict] = {}
    for item in ds:
        filename = Path(item["video"]).name
        if filename not in video_index:
            continue
        mapping[str(video_index[filename])] = {
            "question_id": item["question_id"],
            "video_id": item["video_id"],
            "question": item["question"],
            "candidates": item["choices"]["choice"],
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
            "question_id": str(meta["question_id"]),
            "video_id": str(meta["video_id"]),
            "video": vp,
            "audio": str(wav),
            "question": meta["question"],
            "candidates": meta["candidates"],
            "answer": meta["answer"],
        }


def main() -> None:
    raw = load_dataset(SOURCE_REPO, name=SOURCE_CONFIG, split="validation")
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
        gen_kwargs={"video_list": list(question_map.keys()), "mapping": question_map},
        features=features,
    )

    test_ds.push_to_hub(TARGET_REPO, config_name=SOURCE_CONFIG, split="test")


if __name__ == "__main__":
    main()
