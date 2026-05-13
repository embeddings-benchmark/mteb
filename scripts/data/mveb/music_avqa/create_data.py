"""Reference data-preparation script for the mteb/MUSIC-AVQA_cls-preprocessed dataset.

REFERENCE ONLY. Not runnable end-to-end: it depends on the third-party
HF mirror `UnFaZeD07/Music-AVQA` (which packages both the question
annotations and the mp4 clips). The canonical
artifact produced by this pipeline is the
`mteb/MUSIC-AVQA_cls-preprocessed` dataset on the HuggingFace Hub.

Source
------
Li, Hu, Xie, Yang, Xie, "Learning to Answer Questions in Dynamic
Audio-Visual Scenarios" (CVPR 2022).
    https://arxiv.org/abs/2203.14072
    https://gewu-lab.github.io/MUSIC-AVQA/

Upstream HF mirror used:
    https://huggingface.co/datasets/UnFaZeD07/Music-AVQA

MVEB-specific processing
------------------------
1. Pull the `.mp4` clips and the test annotations from
   `UnFaZeD07/Music-AVQA` (`split=test`). The original task is open-ended
   audio-visual QA; the `anser` field carries the free-text answer.
2. Filter to rows whose answer is a single musical instrument from the
   22-class instrument vocabulary defined by the dataset authors. This
   converts MUSIC-AVQA into a classification task over instruments.
3. Rename the `anser` column to `label`.
4. Index the local mp4 clips by `{video_id}.mp4` and keep only rows that
   resolve locally.
5. Extract mono 16 kHz PCM audio with ffmpeg; drop any video whose audio
   extraction fails.
6. Schema: {video_id, video, audio, label}. `label` is a ClassLabel over
   the 22 instrument classes (in the canonical order below).
7. Push a single `test` split to `mteb/MUSIC-AVQA_cls-preprocessed`.
8. Final size: ~1,706 rows in the single `test` split.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from datasets import Audio, ClassLabel, Dataset, Features, Value, Video, load_dataset

VIDEO_ROOT = Path("videos")
SOURCE_REPO = "UnFaZeD07/Music-AVQA"
TARGET_REPO = "mteb/MUSIC-AVQA_cls-preprocessed"

INSTRUMENT_LABELS = [
    "accordion", "acoustic_guitar", "bagpipe", "banjo", "bassoon",
    "cello", "clarinet", "congas", "drum", "electric_bass", "erhu",
    "flute", "guzheng", "piano", "pipa", "saxophone", "suona",
    "trumpet", "tuba", "ukulele", "violin", "xylophone",
]


def index_videos(root: Path) -> dict[str, Path]:
    return {p.name: p for p in root.rglob("*.mp4") if p.is_file()}


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
    instruments = set(INSTRUMENT_LABELS)
    qa = load_dataset(SOURCE_REPO, split="test")
    classification = qa.filter(lambda r: r["anser"] in instruments)
    classification = classification.rename_column("anser", "label")

    video_index = index_videos(VIDEO_ROOT)

    rows = []
    for item in classification:
        filename = f"{item['video_id']}.mp4"
        if filename in video_index:
            rows.append({
                "video_id": item["video_id"],
                "video_path": video_index[filename],
                "label": item["label"],
            })

    features = Features({
        "video_id": Value("string"),
        "video": Video(),
        "audio": Audio(sampling_rate=16000),
        "label": ClassLabel(names=INSTRUMENT_LABELS),
    })

    def gen():
        for r in rows:
            wav = extract_audio_16k_mono(r["video_path"])
            if wav is None:
                continue
            yield {
                "video_id": r["video_id"],
                "video": str(r["video_path"]),
                "audio": str(wav),
                "label": r["label"],
            }

    test_ds = Dataset.from_generator(gen, features=features)
    test_ds.push_to_hub(TARGET_REPO, split="test")


if __name__ == "__main__":
    main()
