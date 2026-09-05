#!/usr/bin/env python3
"""Build the COCO Modality Equivalence dataset for MTEB.

This script constructs a unified retrieval dataset where the SAME set of MSCOCO
images is paired with text captions, human-spoken captions (SpokenCOCO), and
TTS-synthesized captions (SpeechCoco). By holding semantic content constant
and varying only the modality, any retrieval score difference is attributable
to modality difficulty rather than dataset content.

Source datasets (all already on HuggingFace):
  - whybe-choi/SpokenCOCOA2IRetrieval  -- human speech + COCO images
  - dukesun99/SpeechCoco-A2I           -- TTS speech + COCO images
  - mteb/mbeir_mscoco_task0            -- text captions + COCO images

We intersect the corpus image IDs across all three sources, then build six
retrieval configs over the shared pool:
  t2i  : text caption  -> image
  a2i_h: human speech  -> image
  a2i_s: TTS speech    -> image
  i2t  : image         -> text caption
  i2a_h: image         -> human speech
  i2a_s: image         -> TTS speech

Examples:
  # Build and inspect locally
  uv run python scripts/data/coco_modality_equivalence/create_data.py --stage build

  # Build and push to Hugging Face (set HF_TOKEN env var first)
  uv run python scripts/data/coco_modality_equivalence/create_data.py --stage all --push --repo YOUR_HF_ORG/coco-modality-equivalence
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Audio, Dataset, Features, Image, Value, load_dataset

_SPOKEN_COCO_A2I = "whybe-choi/SpokenCOCOA2IRetrieval"
_SPOKEN_COCO_I2A = "whybe-choi/SpokenCOCOI2ARetrieval"
_SPEECH_COCO_A2I = "dukesun99/SpeechCoco-A2I"
_SPEECH_COCO_I2A = "dukesun99/SpeechCoco-I2A"
_MSCOCO_T2I = "mteb/mbeir_mscoco_task0"
_SPLIT = "test"

_DEFAULT_REPO = "mteb/coco-modality-equivalence"


def _load_spoken_coco() -> tuple[dict, dict]:
    """Return (image_id -> image_bytes, image_id -> list[audio_bytes]) from SpokenCOCO."""
    print("Loading SpokenCOCO A2I …")
    ds_a2i = load_dataset(_SPOKEN_COCO_A2I, split=_SPLIT)
    ds_i2a = load_dataset(_SPOKEN_COCO_I2A, split=_SPLIT)

    # SpokenCOCO corpus has images keyed by corpus id; queries keyed by spoken-caption id.
    # We need to find the corpus image column and the cross-reference to query audio.
    print("SpokenCOCO A2I corpus columns:", ds_a2i.column_names)
    print("SpokenCOCO I2A corpus columns:", ds_i2a.column_names)

    # Build image_id -> image_bytes from corpus (image side)
    corpus = ds_i2a  # i2a corpus = spoken captions; queries = images
    # Actually for i2a: queries are images, corpus is audio.
    # For a2i: queries are audio, corpus is images.
    # We'll extract from a2i corpus (images) and i2a queries (images) to get all images.

    images: dict[str, bytes] = {}
    audio_by_image: dict[str, list[bytes]] = {}

    # Walk a2i: query=audio, corpus=image
    for row in ds_a2i:
        img_id = row.get("corpus_id") or row.get("image_id") or row.get("id")
        # SpokenCOCO may use nested dicts; inspect and adapt
        if "image" in row and img_id is not None:
            img_bytes = row["image"]
            if isinstance(img_bytes, dict):
                img_bytes = img_bytes.get("bytes") or b""
            if img_bytes:
                images[str(img_id)] = img_bytes
        if "audio" in row:
            aud = row["audio"]
            if isinstance(aud, dict):
                aud = aud.get("bytes") or b""
            q_img_id = row.get("query_image_id") or row.get("image_id")
            if q_img_id is not None and aud:
                audio_by_image.setdefault(str(q_img_id), []).append(aud)

    return images, audio_by_image


def _load_speech_coco() -> dict[str, bytes]:
    """Return image_id -> TTS audio bytes from SpeechCoco."""
    print("Loading SpeechCoco A2I …")
    ds = load_dataset(_SPEECH_COCO_A2I, split=_SPLIT)
    print("SpeechCoco A2I columns:", ds.column_names)

    tts_by_image: dict[str, bytes] = {}
    for row in ds:
        img_id = row.get("corpus_id") or row.get("image_id") or row.get("id")
        aud = row.get("audio")
        if isinstance(aud, dict):
            aud = aud.get("bytes") or b""
        if img_id is not None and aud:
            tts_by_image.setdefault(str(img_id), aud)
    return tts_by_image


def _load_mscoco_text() -> dict[str, str]:
    """Return image_id -> first text caption from MSCOCO MBEIR."""
    print("Loading MSCOCO text captions …")
    ds = load_dataset(_MSCOCO_T2I, split=_SPLIT)
    print("MSCOCO T2I columns:", ds.column_names)

    text_by_image: dict[str, str] = {}
    for row in ds:
        img_id = row.get("corpus_id") or row.get("image_id") or row.get("id")
        text = row.get("text") or row.get("caption") or ""
        if img_id is not None and text:
            text_by_image.setdefault(str(img_id), str(text))
    return text_by_image


def stage_build(work: Path) -> None:
    """Download sources, intersect, and save the unified dataset to work/."""
    work.mkdir(parents=True, exist_ok=True)

    images, audio_human = _load_spoken_coco()
    audio_tts = _load_speech_coco()
    texts = _load_mscoco_text()

    # Intersect: keep only IDs present in ALL four sources
    shared_ids = set(images) & set(audio_human) & set(audio_tts) & set(texts)
    print(f"\nShared image IDs across all sources: {len(shared_ids)}")

    if not shared_ids:
        raise RuntimeError(
            "No shared IDs found. The corpus ID fields may differ across datasets — "
            "inspect the printed column names above and adapt _load_* functions."
        )

    rows = []
    for img_id in sorted(shared_ids):
        rows.append(
            {
                "image_id": img_id,
                "image": images[img_id],
                "text": texts[img_id],
                "audio_human": audio_human[img_id][0],  # first recording
                "audio_tts": audio_tts[img_id],
            }
        )

    print(f"Building dataset with {len(rows)} items …")
    ds = (
        Dataset.from_list(rows)
        .cast_column("image", Image())
        .cast_column("audio_human", Audio(sampling_rate=16_000))
        .cast_column("audio_tts", Audio(sampling_rate=16_000))
    )

    out = work / "dataset"
    ds.save_to_disk(str(out))
    print(f"Saved to {out}")

    # Write a manifest for inspection
    manifest = {
        "n_items": len(rows),
        "image_ids_sample": sorted(shared_ids)[:5],
        "columns": ds.column_names,
    }
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Manifest:", manifest)


def stage_push(work: Path, repo: str) -> None:
    """Push the locally built dataset to Hugging Face Hub."""
    from datasets import load_from_disk

    from huggingface_hub import HfApi

    ds_path = work / "dataset"
    if not ds_path.exists():
        raise FileNotFoundError(f"{ds_path} not found — run --stage build first")

    ds = load_from_disk(str(ds_path))

    api = HfApi()
    api.create_repo(repo_id=repo, repo_type="dataset", exist_ok=True)
    ds.push_to_hub(repo, split="test", private=False)
    info = api.dataset_info(repo)
    print(f"Pushed to {repo} — revision: {info.sha}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["build", "push", "all"],
        default="build",
        help="build: create dataset locally; push: upload; all: both",
    )
    parser.add_argument(
        "--work",
        default="work/coco_modality_equivalence",
        help="Local working directory",
    )
    parser.add_argument(
        "--repo",
        default=_DEFAULT_REPO,
        help="HuggingFace repo ID to push to",
    )
    args = parser.parse_args()

    work = Path(args.work)

    if args.stage in ("build", "all"):
        stage_build(work)
    if args.stage in ("push", "all"):
        stage_push(work, args.repo)


if __name__ == "__main__":
    main()
