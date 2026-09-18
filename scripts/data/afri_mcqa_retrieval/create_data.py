#!/usr/bin/env python3
"""Build the Afri-MCQA multilingual speech-image retrieval tasks for MTEB.

Source is `Atnafu/Afri-MCQA` at a pinned revision, a culturally grounded question
answering benchmark written and recorded by native speakers across 16 African languages
(Tonja et al. 2026). It ships an official `test` split, so nothing here is drawn from
training data.

Each question is spoken aloud in the native language and grounded in one image, which is
the pairing used here: retrieve the photograph a spoken question is asking about. Only
the question audio is kept; the four spoken answer options are dropped because they
describe candidate answers rather than the image.

Images are held per language rather than pooled. The same photograph can carry questions
in more than one language, so a pooled image corpus would mark a correct retrieval wrong
whenever the model returned the twin belonging to another language.

Examples:
  # Build the reshaped tables locally.
  uv run python scripts/data/afri_mcqa_retrieval/create_data.py --stage build

  # Build and publish both directions.
  uv run python scripts/data/afri_mcqa_retrieval/create_data.py --stage all --push
"""

from __future__ import annotations

import argparse
import io
import hashlib
import json
from pathlib import Path

from datasets import Audio, Dataset, Image, load_dataset
from huggingface_hub import HfApi

_SOURCE_REPO = "Atnafu/Afri-MCQA"
_SOURCE_REV = "8b8c53df57b0c2cf9d9798c53515ba1dd14df669"
_TARGET = "vnahata/AfriMCQA-speech-image-retrieval"
_LICENSE = "cc-by-nc-4.0"

# dataset's `Language` column -> the subset name used on the Hub
_LANGS = {
    "Akan/Twi": "twi",
    "Amharic": "amh",
    "Chichewa": "nya",
    "Hausa": "hau",
    "Igbo": "ibo",
    "Kikuyu": "kik",
    "Kinyarwanda": "kin",
    "Lingala": "lin",
    "Luganda": "lug",
    "Oromo": "orm",
    "Sesotho": "sot",
    "Setswana": "tsn",
    "Somali": "som",
    "Tigrinya": "tir",
    "Yoruba": "yor",
    "Zulu": "zul",
}

_KEEP = ["ID", "Language", "image", "native_audio_question"]


def stage_build(work: Path) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(
        _SOURCE_REPO, "all_test", revision=_SOURCE_REV, split="test"
    ).select_columns(_KEEP)

    # Read the two label columns on their own; filtering whole rows would decode media.
    all_langs = ds["Language"]
    all_ids = ds["ID"]

    images_only = ds.select_columns(["image"])
    audio_only = ds.select_columns(["native_audio_question"])

    # Not every question was recorded, and a few images are missing. Read both columns
    # undecoded to find the gaps cheaply, then drop those rows rather than publish
    # entries that fail to load.
    undecoded = ds.select_columns(["image", "native_audio_question"])
    undecoded = undecoded.cast_column("image", Image(decode=False)).cast_column(
        "native_audio_question", Audio(decode=False)
    )
    # Hashed in the same pass. A few photographs are byte-identical to another in the
    # same language, and a few questions share one recording; both would leave an
    # identical query or document relevant to only one of the rows it matches.
    ok_img, ok_aud, img_hash, aud_hash = [], [], [], []
    for rec in undecoded:
        iv, av = rec["image"], rec["native_audio_question"]
        ib = iv.get("bytes") if iv else None
        ab = av.get("bytes") if av else None
        ok_img.append(bool(ib))
        ok_aud.append(bool(ab))
        img_hash.append(hashlib.md5(ib).hexdigest() if ib else None)
        aud_hash.append(hashlib.md5(ab).hexdigest() if ab else None)

    dropped = sum(1 for a, i in zip(ok_aud, ok_img, strict=True) if not (a and i))
    print(f"dropping {dropped} rows with no recording or no image", flush=True)

    counts: dict[str, dict[str, int]] = {}
    for name, code in _LANGS.items():
        rows = [
            i
            for i, lg in enumerate(all_langs)
            if lg == name and ok_aud[i] and ok_img[i]
        ]
        if not rows:
            continue
        img_path, aud_path = (
            work / f"{code}_images.parquet",
            work / f"{code}_audio.parquet",
        )
        if img_path.exists() and aud_path.exists():
            import pyarrow.parquet as pq

            counts[code] = {
                "images": pq.read_metadata(img_path).num_rows,
                "questions": pq.read_metadata(aud_path).num_rows,
            }
            print(f"  {code}: cached {counts[code]}", flush=True)
            continue

        # ID is "<image>_<question index>", so the stem groups questions on one image.
        # Identical photographs collapse onto the first stem carrying that image.
        canon: dict[str, str] = {}
        stem_of_hash: dict[str, str] = {}
        first: dict[str, int] = {}
        keep_rows: list[int] = []
        seen_audio: set[str] = set()
        for i in rows:
            stem = all_ids[i].split("_")[0]
            target = stem_of_hash.setdefault(img_hash[i], stem)
            canon[stem] = target
            first.setdefault(target, i)
            if aud_hash[i] in seen_audio:
                continue
            seen_audio.add(aud_hash[i])
            keep_rows.append(i)

        img = images_only.select(list(first.values()))
        img = img.add_column("id", [f"{code}-img-{s}" for s in first])
        img.cast_column("image", Image()).to_parquet(
            str(work / f"{code}_images.parquet")
        )

        aud = audio_only.select(keep_rows).rename_column(
            "native_audio_question", "audio"
        )
        aud = aud.add_column("id", [f"{code}-aud-{all_ids[i]}" for i in keep_rows])
        aud = aud.add_column(
            "image_id",
            [f"{code}-img-{canon[all_ids[i].split('_')[0]]}" for i in keep_rows],
        )
        aud.cast_column("audio", Audio(sampling_rate=16000)).to_parquet(
            str(work / f"{code}_audio.parquet")
        )

        counts[code] = {"images": img.num_rows, "questions": aud.num_rows}
        print(f"  {code}: {counts[code]}", flush=True)

    (work / "counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print("built", json.dumps(counts))
    return counts


def _card(codes: list[str]) -> str:
    lines = [
        "---",
        f"license: {_LICENSE}",
        "task_categories:",
        "  - image-to-text",
        "  - audio-classification",
        "language:",
        *[f"  - {c}" for c in codes],
        "tags:",
        "  - mteb",
        "  - retrieval",
        "  - multilingual",
        "configs:",
    ]
    for c in codes:
        for suffix, src in (("images", f"{c}_images"), ("audio", f"{c}_audio")):
            lines += [
                f"  - config_name: {c}-{suffix}",
                "    data_files:",
                "      - split: test",
                f"        path: {src}.parquet",
            ]
    lines += [
        "---",
        "",
        "# Afri-MCQA speech-image retrieval (MTEB)",
        "",
        "Afri-MCQA reshaped for retrieval: find the photograph a spoken question is asking",
        "about. Questions are spoken by native speakers in 16 African languages and are",
        "grounded in culturally relevant images.",
        "",
        f"Source: `{_SOURCE_REPO}` at revision `{_SOURCE_REV[:7]}`, {_LICENSE}, official",
        "`test` split. Images are stored per language because one photograph can carry",
        "questions in several languages.",
        "",
        "Built by `scripts/data/afri_mcqa_retrieval/create_data.py` in the MTEB repo.",
    ]
    return "\n".join(lines) + "\n"


def stage_push(work: Path) -> None:
    api = HfApi()
    api.create_repo(_TARGET, repo_type="dataset", exist_ok=True)
    codes = [c for c in _LANGS.values() if (work / f"{c}_images.parquet").exists()]
    for c in codes:
        for kind in ("images", "audio"):
            api.upload_file(
                path_or_fileobj=str(work / f"{c}_{kind}.parquet"),
                path_in_repo=f"{c}_{kind}.parquet",
                repo_id=_TARGET,
                repo_type="dataset",
            )
        print(f"  pushed {c}", flush=True)
    api.upload_file(
        path_or_fileobj=io.BytesIO(_card(codes).encode()),
        path_in_repo="README.md",
        repo_id=_TARGET,
        repo_type="dataset",
    )
    print(f"pushed {_TARGET}: {len(codes)} languages")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["build", "push", "all"], default="all")
    parser.add_argument("--work-dir", type=Path, default=Path("afri_mcqa_work"))
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    print(f"source: {_SOURCE_REPO}@{_SOURCE_REV[:7]} (license {_LICENSE})")
    if args.stage in ("build", "all"):
        stage_build(args.work_dir)
    if args.stage == "push" or (args.push and args.stage == "all"):
        stage_push(args.work_dir)


if __name__ == "__main__":
    main()
