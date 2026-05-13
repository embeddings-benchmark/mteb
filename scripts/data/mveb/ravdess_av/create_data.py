"""Reference data-preparation script for the mteb/RAVDESS_AV dataset.

REFERENCE ONLY. Not runnable end-to-end: it depends on third-party HF
mirrors of RAVDESS (`PiantoDiGruppo/Ravdess_AML` for the audio-visual
mp4s, `AbstractTTS/RAVDESS` for the per-clip metadata). The canonical artifact produced by this pipeline is
the `mteb/RAVDESS_AV` dataset on the HuggingFace Hub.

Source
------
Livingstone & Russo, "The Ryerson Audio-Visual Database of Emotional
Speech and Song (RAVDESS): A dynamic, multimodal set of facial and
vocal expressions in North American English" (PLoS ONE 2018).
    https://doi.org/10.1371/journal.pone.0196391
    https://zenodo.org/record/1188976

Upstream HF mirrors used:
    Videos:   https://huggingface.co/datasets/PiantoDiGruppo/Ravdess_AML
    Metadata: https://huggingface.co/datasets/AbstractTTS/RAVDESS

MVEB-specific processing
------------------------
1. Download the audio-visual `.mp4` clips from `PiantoDiGruppo/Ravdess_AML`
   and the labelled metadata table from `AbstractTTS/RAVDESS`
   (`split=train`).
2. RAVDESS filenames encode modality in the first 2 characters: `03-...`
   is audio-only, `01-...` is full audio-visual. Each metadata `file`
   field points at the audio-only variant, so swap the modality prefix
   to `01` and append `.mp4` to obtain the corresponding AV clip name.
3. Keep only rows whose remapped video filename resolves locally.
4. Carry the original metadata fields through unchanged: gender, emotion,
   transcription.
5. Extract mono 16 kHz PCM audio with ffmpeg; drop any video whose audio
   extraction fails.
6. Schema: {video_id, video, audio, gender, emotion, transcription}.
   `gender` is a ClassLabel of {male, female}; `emotion` is a ClassLabel
   over the values observed in the metadata (sorted alphabetically);
   `transcription` is a free-form string.
7. Push a single `test` split to `mteb/RAVDESS_AV`.
8. Final size: ~1,440 rows in the single `test` split.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from datasets import Audio, ClassLabel, Dataset, Features, Value, Video, load_dataset

VIDEO_ROOT = Path("videos")
SOURCE_REPO = "AbstractTTS/RAVDESS"
TARGET_REPO = "mteb/RAVDESS_AV"


def audio_filename_to_video(file_field: str) -> str:
    # Metadata `file` looks like ".../03-01-05-01-02-01-12.wav".
    # The audio-visual counterpart has modality prefix "01" and a .mp4
    # extension; everything after the modality bits is shared.
    stem = Path(file_field).stem  # 03-01-05-01-02-01-12
    return "01" + stem[2:] + ".mp4"


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
    metadata = load_dataset(SOURCE_REPO, split="train")
    video_index = index_videos(VIDEO_ROOT)

    rows = []
    for item in metadata:
        video_filename = audio_filename_to_video(item["file"])
        if video_filename in video_index:
            rows.append({
                "video_id": video_filename,
                "video_path": video_index[video_filename],
                "gender": item["gender"],
                "emotion": item["emotion"],
                "transcription": item["transcription"],
            })

    emotions = sorted({r["emotion"] for r in rows})

    features = Features({
        "video_id": Value("string"),
        "video": Video(),
        "audio": Audio(sampling_rate=16000),
        "gender": ClassLabel(names=["male", "female"]),
        "emotion": ClassLabel(names=emotions),
        "transcription": Value("string"),
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
                "gender": r["gender"],
                "emotion": r["emotion"],
                "transcription": r["transcription"],
            }

    test_ds = Dataset.from_generator(gen, features=features)
    test_ds.push_to_hub(TARGET_REPO, split="test")


if __name__ == "__main__":
    main()
