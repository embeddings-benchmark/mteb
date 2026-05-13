"""Reference data-preparation script for the mteb/Human-Animal-Cartoon dataset.

REFERENCE ONLY. Not runnable end-to-end: it depends on third-party HF
mirrors of the HAC (Human-Animal-Cartoon) test-only splits. The canonical artifact
produced by this pipeline is the `mteb/Human-Animal-Cartoon` dataset on
the HuggingFace Hub.

Source
------
Dong, Hong, Yu, Zhao, "SimMMDG: A Simple and Effective Framework for
Multi-modal Domain Generalization" (NeurIPS 2023).
    https://arxiv.org/abs/2310.19795
    https://github.com/donghao51/SimMMDG

Upstream HF mirrors used:
    https://huggingface.co/datasets/hdong51/Human-Animal-Cartoon

MVEB-specific processing
------------------------
1. Download the three HAC test-only CSVs (`HAC_test_only_{human,animal,
   cartoon}.csv`) and the matching `{human,animal,cartoon}.zip` video
   archives. Each CSV has columns `video_file,label`.
2. Concatenate the three CSVs, tagging each row with its `source` domain
   (`human` / `animal` / `cartoon`).
3. Map the integer `label` field to one of 7 action names: sleeping,
   watching tv, eating, drinking, swimming, running, opening door.
4. Index the unzipped videos by filename and keep rows whose `video_file`
   resolves locally.
5. Apply class threshold of 1 and cap at 32 examples per class.
6. Extract mono 16 kHz PCM audio with ffmpeg; drop any video whose audio
   extraction fails.
7. Schema: {video_id, video, audio, action, source}. `action` is a
   ClassLabel over the 7 classes (sorted alphabetically); `source` is a
   free-form string tag.
8. Push a single `test` split to `mteb/Human-Animal-Cartoon`.
9. Final size: ~644 rows in the single `test` split.
"""

from __future__ import annotations

import subprocess
from collections import defaultdict
from pathlib import Path

import pandas as pd
from datasets import Audio, ClassLabel, Dataset, Features, Value, Video

VIDEO_ROOT = Path("HAC")
CSV_PATHS = {
    "human": "HAC_test_only_human.csv",
    "animal": "HAC_test_only_animal.csv",
    "cartoon": "HAC_test_only_cartoon.csv",
}
LABEL_NAMES = {
    0: "sleeping",
    1: "watching tv",
    2: "eating",
    3: "drinking",
    4: "swimming",
    5: "running",
    6: "opening door",
}

CLASS_THRESHOLD = 1
MAX_PER_CLASS = 32
TARGET_REPO = "mteb/Human-Animal-Cartoon"


def load_combined_csv() -> pd.DataFrame:
    frames = []
    for source, csv in CSV_PATHS.items():
        df = pd.read_csv(csv, header=None, names=["video_file", "label"])
        df["source"] = source
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined["action"] = combined["label"].map(LABEL_NAMES)
    return combined


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
    df = load_combined_csv()
    video_index = index_videos(VIDEO_ROOT)

    rows = [r for r in df.itertuples(index=False) if r.video_file in video_index]

    per_class: dict[str, int] = defaultdict(int)
    selected = []
    for r in rows:
        if per_class[r.action] >= MAX_PER_CLASS:
            continue
        per_class[r.action] += 1
        selected.append(r)

    actions = sorted({r.action for r in selected})
    features = Features({
        "video_id": Value("string"),
        "video": Video(),
        "audio": Audio(sampling_rate=16000),
        "action": ClassLabel(names=actions),
        "source": Value("string"),
    })

    def gen():
        for r in selected:
            vp = video_index[r.video_file]
            wav = extract_audio_16k_mono(vp)
            if wav is None:
                continue
            yield {
                "video_id": r.video_file,
                "video": str(vp),
                "audio": str(wav),
                "action": r.action,
                "source": r.source,
            }

    test_ds = Dataset.from_generator(gen, features=features)
    test_ds.push_to_hub(TARGET_REPO, split="test")


if __name__ == "__main__":
    main()
