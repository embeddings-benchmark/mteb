"""Reference data-preparation script for the mteb/UCF101-51VA dataset.

REFERENCE ONLY. Not runnable end-to-end: it depends on the CRCV UCF101
download mirror. The canonical artifact
produced by this pipeline is the `mteb/UCF101-51VA` dataset on the
HuggingFace Hub.

Source
------
Soomro et al., "UCF101: A Dataset of 101 Human Actions Classes From Videos
in The Wild" (CRCV-TR-12-01, 2012).
    https://arxiv.org/abs/1212.0402

Download infrastructure used (upstream):
    https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
    Official train/test split files `trainlist01.txt` / `testlist01.txt`.

MVEB-specific processing
------------------------
1. Decode UCF101.rar to `UCF-101/<class>/<clip>.avi`. Use the official
   `split 1` train/test partition. The class label is the directory name.
2. Restrict the 101 UCF classes to a hand-picked subset of 51 classes that
   carry meaningful audio content (the "51VA" - 51 video+audio subset).
3. Require each kept class to have at least 13 clips in BOTH train and test
   (all 51 selected classes pass).
4. Cap at 1,000 clips per class per split.
5. Extract mono 16 kHz PCM audio with ffmpeg. Drop clips where extraction
   fails.
6. Schema: {video, audio, label}. `label` is a ClassLabel over the 51
   kept UCF class names (the audio-bearing subset).
7. Final sizes: ~4,890 train / ~1,947 test.
"""

from __future__ import annotations

import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from datasets import Audio, ClassLabel, Dataset, Features, Video

VIDEO_ROOT = Path("UCF-101")
SPLIT_ROOT = Path("splits")
SPLIT_NUM = 1

CLASS_THRESHOLD = 13
MAX_PER_CLASS = 1000
TARGET_REPO = "mteb/UCF101-51VA"

ALLOWED_CLASSES = {
    "ApplyEyeMakeup", "ApplyLipstick", "Archery", "BabyCrawling", "BalanceBeam",
    "BandMarching", "BasketballDunk", "BlowDryHair", "BlowingCandles",
    "BodyWeightSquats", "Bowling", "BoxingPunchingBag", "BoxingSpeedBag",
    "BrushingTeeth", "CliffDiving", "CricketBowling", "CricketShot",
    "CuttingInKitchen", "FieldHockeyPenalty", "FloorGymnastics",
    "FrisbeeCatch", "FrontCrawl", "Haircut", "Hammering", "HammerThrow",
    "HandstandPushups", "HandstandWalking", "HeadMassage", "IceDancing",
    "Knitting", "LongJump", "MoppingFloor", "ParallelBars", "PlayingCello",
    "PlayingDaf", "PlayingDhol", "PlayingFlute", "PlayingSitar", "Rafting",
    "ShavingBeard", "Shotput", "SkyDiving", "SoccerPenalty", "StillRings",
    "SumoWrestling", "Surfing", "TableTennisShot", "Typing", "UnevenBars",
    "WallPushups", "WritingOnBoard",
}


def discover_classes() -> dict[str, int]:
    classes = sorted(p.name for p in VIDEO_ROOT.iterdir() if p.is_dir())
    return {c: i for i, c in enumerate(classes)}


def read_split_map(path: Path, class_to_idx: dict[str, int]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    with open(path) as f:
        for line in f:
            rel_path = line.strip().split()[0]
            cls = rel_path.split("/")[0]
            mapping[str(VIDEO_ROOT / rel_path)] = class_to_idx[cls]
    return mapping


def select_balanced(
    split_map: dict[str, int], valid_classes: set[int],
) -> list[str]:
    selected: list[str] = []
    per_class: dict[int, int] = defaultdict(int)
    for path, cls in split_map.items():
        if (
            Path(path).exists()
            and cls in valid_classes
            and per_class[cls] < MAX_PER_CLASS
        ):
            selected.append(path)
            per_class[cls] += 1
    return selected


def extract_audio_16k_mono(video_path: str) -> str | None:
    wav = str(Path(video_path).with_suffix(".wav"))
    result = subprocess.run(
        [
            "ffmpeg", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            wav, "-y",
        ],
        capture_output=True,
    )
    return wav if result.returncode == 0 else None


def main() -> None:
    class_to_idx = discover_classes()
    id_to_class = {i: c for c, i in class_to_idx.items()}

    train_map = read_split_map(SPLIT_ROOT / f"trainlist0{SPLIT_NUM}.txt", class_to_idx)
    test_map = read_split_map(SPLIT_ROOT / f"testlist0{SPLIT_NUM}.txt", class_to_idx)

    train_counts = Counter(train_map.values())
    test_counts = Counter(test_map.values())

    valid_classes = {
        c for c in train_counts
        if train_counts[c] >= CLASS_THRESHOLD
        and test_counts.get(c, 0) >= CLASS_THRESHOLD
        and id_to_class[c] in ALLOWED_CLASSES
    }

    train_videos = select_balanced(train_map, valid_classes)
    test_videos = select_balanced(test_map, valid_classes)

    class_names = list(class_to_idx.keys())
    features = Features({
        "video": Video(),
        "audio": Audio(sampling_rate=16000),
        "label": ClassLabel(names=class_names),
    })

    def gen(video_list: list[str], mapping: dict[str, int]):
        for vp in video_list:
            audio = extract_audio_16k_mono(vp)
            if audio is None:
                continue
            yield {"video": vp, "audio": audio, "label": mapping[vp]}

    train_ds = Dataset.from_generator(
        gen, gen_kwargs={"video_list": train_videos, "mapping": train_map},
        features=features, num_proc=2,
    )
    test_ds = Dataset.from_generator(
        gen, gen_kwargs={"video_list": test_videos, "mapping": test_map},
        features=features, num_proc=2,
    )

    train_ds.push_to_hub(TARGET_REPO, split="train")
    test_ds.push_to_hub(TARGET_REPO, split="test")


if __name__ == "__main__":
    main()
