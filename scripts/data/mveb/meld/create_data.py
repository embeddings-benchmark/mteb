"""Reference data-preparation script for the mteb/MELD dataset.

REFERENCE ONLY. Not runnable end-to-end: it depends on third-party HF
and Kaggle mirrors of MELD. The canonical artifact
produced by this pipeline is the `mteb/MELD` dataset on the
HuggingFace Hub.

Source
------
Poria, Hazarika, Majumder, Naik, Cambria, Mihalcea, "MELD: A Multimodal
Multi-Party Dataset for Emotion Recognition in Conversations" (ACL 2019).
    https://arxiv.org/abs/1810.02508
    https://affective-meld.github.io/

Upstream mirrors used:
    Metadata: HF dataset `TwinkStart/MELD` (split=test)
    Videos:   Kaggle `daiphuocvo/meld-multimodal-speech-text-sentiment-analysis`
              and HF dataset `seniruk/MELD-emotion-detection-preprocessed`
              (`video_files.zip`).

MVEB-specific processing
------------------------
1. Load the MELD test metadata from `TwinkStart/MELD` and download/unzip
   the matching `.mp4` clips from the upstream mirrors.
2. Index the downloaded clips by filename and keep only metadata rows
   whose `video_path` resolves to a local file.
3. Carry the original fields through unchanged: utterance, speaker,
   emotion, sentiment.
4. Extract mono 16 kHz PCM audio with ffmpeg; drop any video whose audio
   extraction fails.
5. Schema: {video_id, video, audio, utterance, speaker, emotion,
   sentiment}. `speaker`, `emotion`, and `sentiment` are ClassLabels
   over the values observed in the test split (sorted alphabetically);
   `utterance` is a free-form string.
6. Push a single `test` split to `mteb/MELD`.
7. Final size: ~2,610 rows in the single `test` split.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from datasets import Audio, ClassLabel, Dataset, Features, Value, Video, load_dataset

VIDEO_ROOT = Path("meld_videos")
SOURCE_REPO = "TwinkStart/MELD"
TARGET_REPO = "mteb/MELD"


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
    metadata = load_dataset(SOURCE_REPO, split="test")
    video_index = index_videos(VIDEO_ROOT)

    rows = []
    for item in metadata:
        filename = item["video_path"].split("/")[-1]
        if filename in video_index:
            rows.append({
                "video_id": filename,
                "video_path": video_index[filename],
                "utterance": item["Utterance"],
                "speaker": item["Speaker"],
                "emotion": item["Emotion"],
                "sentiment": item["Sentiment"],
            })

    emotions = sorted({r["emotion"] for r in rows})
    speakers = sorted({r["speaker"] for r in rows})
    sentiments = sorted({r["sentiment"] for r in rows})

    features = Features({
        "video_id": Value("string"),
        "video": Video(),
        "audio": Audio(sampling_rate=16000),
        "utterance": Value("string"),
        "speaker": ClassLabel(names=speakers),
        "emotion": ClassLabel(names=emotions),
        "sentiment": ClassLabel(names=sentiments),
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
                "utterance": r["utterance"],
                "speaker": r["speaker"],
                "emotion": r["emotion"],
                "sentiment": r["sentiment"],
            }

    test_ds = Dataset.from_generator(gen, features=features)
    test_ds.push_to_hub(TARGET_REPO, split="test")


if __name__ == "__main__":
    main()
