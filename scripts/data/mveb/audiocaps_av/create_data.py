"""Reference data-preparation script for the mteb/AudioCaps_AV dataset.

REFERENCE ONLY. Not runnable end-to-end: it depends on a Kaggle video
mirror of AudioCaps (`ramaneswaran/audiocaps-videos-part-2`) and the
`d0rj/audiocaps` caption mirror on the HuggingFace Hub. The canonical
artifact produced by this pipeline is the `mteb/AudioCaps_AV` dataset on
the HuggingFace Hub.

Source
------
Kim et al., "AudioCaps: Generating Captions for Audios in The Wild"
(NAACL 2019). https://aclanthology.org/N19-1011/

Upstream artifacts:
    https://huggingface.co/datasets/d0rj/audiocaps           (captions)
    https://www.kaggle.com/datasets/ramaneswaran/audiocaps-videos-part-2
        (video mirror, distributed as a single zip archive)

MVEB-specific processing
------------------------
1. Load the `test` split of `d0rj/audiocaps` for captions and emit a
   `wanted_videos.txt` list of expected `{youtube_id}_{start_time}.mkv`
   filenames.
2. Pull `ramaneswaran/audiocaps-videos-part-2` via `kagglehub`, then
   selectively extract only the wanted mkv files from the archive.
3. Pair each surviving local clip with its caption, youtube_id and
   start_time.
4. Extract mono 16 kHz PCM audio with ffmpeg. Drop clips whose audio
   extraction fails.
5. Deduplicate by caption (keep the first occurrence) to produce the
   final audio-visual retrieval pool.
6. Schema: {video, audio, caption, youtube_id, start_time}. Single
   `test` split.
7. Final size: ~665 rows in the single `test` split.
"""

from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

from datasets import Audio, Dataset, Features, Value, Video, load_dataset

SOURCE_CAPTIONS = "d0rj/audiocaps"
KAGGLE_DATASET = "ramaneswaran/audiocaps-videos-part-2"
ARCHIVE_PATH = Path(
    "/root/.cache/kagglehub/datasets/ramaneswaran/audiocaps-videos-part-2/1.archive"
)
VIDEO_ROOT = Path("videos")
WANTED_LIST = Path("wanted_videos.txt")
TARGET_REPO = "mteb/AudioCaps_AV"


def write_wanted_list(ds) -> None:
    with WANTED_LIST.open("w") as f:
        for item in ds:
            f.write(f"{item['youtube_id']}_{item['start_time']}.mkv\n")


def extract_wanted_from_archive() -> None:
    VIDEO_ROOT.mkdir(exist_ok=True)
    wanted = {line.strip() for line in WANTED_LIST.read_text().splitlines() if line.strip()}
    with zipfile.ZipFile(ARCHIVE_PATH) as z:
        present = set(z.namelist())
        for name in wanted & present:
            z.extract(name, VIDEO_ROOT)


def build_caption_map(ds) -> dict[str, dict]:
    mapping: dict[str, dict] = {}
    for item in ds:
        video_path = VIDEO_ROOT / f"{item['youtube_id']}_{item['start_time']}.mkv"
        if video_path.exists():
            mapping[str(video_path)] = {
                "caption": item["caption"],
                "youtube_id": item["youtube_id"],
                "start_time": item["start_time"],
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
            "video": vp,
            "audio": str(wav),
            "caption": meta["caption"],
            "youtube_id": meta["youtube_id"],
            "start_time": meta["start_time"],
        }


def deduplicate_by_caption(ds: Dataset) -> Dataset:
    seen: set[str] = set()
    unique_indices: list[int] = []
    for i, caption in enumerate(ds["caption"]):
        if caption not in seen:
            seen.add(caption)
            unique_indices.append(i)
    return ds.select(unique_indices)


def main() -> None:
    raw = load_dataset(SOURCE_CAPTIONS, split="test")
    write_wanted_list(raw)
    extract_wanted_from_archive()

    caption_map = build_caption_map(raw)
    video_list = list(caption_map.keys())

    features = Features({
        "video": Video(),
        "audio": Audio(sampling_rate=16000),
        "caption": Value("string"),
        "youtube_id": Value("string"),
        "start_time": Value("int32"),
    })

    test_ds = Dataset.from_generator(
        generator,
        gen_kwargs={"video_list": video_list, "mapping": caption_map},
        features=features,
    )

    ds_unique = deduplicate_by_caption(test_ds)
    ds_unique.push_to_hub(TARGET_REPO, split="test")


if __name__ == "__main__":
    main()
