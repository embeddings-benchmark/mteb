#!/usr/bin/env python3
"""Build the Afri-MCQA cross-modal cultural category classification task for MTEB.

Source is `Atnafu/Afri-MCQA` at a pinned revision (Tonja et al. 2026). Each entry pairs a
culturally relevant photograph with a question about it spoken by a native speaker, and
carries the cultural category the entry belongs to. Classifying that category from the
image and the spoken question together is a cross-modal task: mteb currently has no
audio+image classification task at all.

The official `dev` split becomes the training split the evaluator fits on and the official
`test` split is evaluated, so no test material is used for training.

Rows are kept only when both the photograph and the recording are present, and only when
the entry carries a single category; roughly 13% of entries are tagged with several
categories at once and cannot act as a single label. Category strings are casefolded
before mapping because the source writes some of them in two different cases.

A photograph carries several questions and each is kept as its own example. Identical
recordings are dropped, and a photograph appearing in the test split is dropped from train
because the label is a property of the photograph and the official dev and test splits
overlap heavily.

Examples:
  # Build the reshaped tables locally.
  uv run python scripts/data/afri_mcqa_classification/create_data.py --stage build

  # Build and publish.
  uv run python scripts/data/afri_mcqa_classification/create_data.py --stage all --push
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

from datasets import Audio, Dataset, Image, load_dataset
from huggingface_hub import HfApi

_SOURCE_REPO = "Atnafu/Afri-MCQA"
_SOURCE_REV = "8b8c53df57b0c2cf9d9798c53515ba1dd14df669"
_TARGET = "vnahata/AfriMCQA-category-classification"  # same repo, gains text columns
_LICENSE = "cc-by-nc-4.0"

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

# Canonical label order; index into this list is the published `label`. "Tranditions" is
# the source's own spelling and is matched as written.
_CATEGORIES = [
    "geography, building, and landmarks",
    "public figure and pop culture",
    "cooking and food",
    "objects, materials, clothing",
    "tranditions, art, and history",
    "brands, products, and companies",
    "plants and animals",
    "people, and everyday life",
    "vehicles and transportation",
    "sports and recreation",
    "other",
]
_LABEL_OF = {name: i for i, name in enumerate(_CATEGORIES)}

# The source spells five of the categories a second way. Matching only one spelling
# silently dropped 173 single-category rows, so both are mapped to the same label.
_ALIASES = {
    "geography, buildings, and landmarks": "geography, building, and landmarks",
    "tradition, art, and history": "tranditions, art, and history",
    "objects, materials, and clothing": "objects, materials, clothing",
    "brands and products": "brands, products, and companies",
    "people and everyday life": "people, and everyday life",
}


_KEEP = [
    "Language",
    "Category",
    "image",
    "native_audio_question",
    "eng_question",
    "native_question",
]


def _prepare(config: str, split: str):
    ds = load_dataset(
        _SOURCE_REPO, config, revision=_SOURCE_REV, split=split
    ).select_columns(_KEEP)

    undecoded = ds.select_columns(["image", "native_audio_question"])
    undecoded = undecoded.cast_column("image", Image(decode=False)).cast_column(
        "native_audio_question", Audio(decode=False)
    )
    ok, img_hash, aud_hash = [], [], []
    for r in undecoded:
        ib = r["image"].get("bytes") if r["image"] else None
        ab = (
            r["native_audio_question"].get("bytes")
            if r["native_audio_question"]
            else None
        )
        ok.append(bool(ib) and bool(ab))
        img_hash.append(hashlib.md5(ib).hexdigest() if ib else None)
        aud_hash.append(hashlib.md5(ab).hexdigest() if ab else None)

    labels, unmapped = [], set()
    for cat in ds["Category"]:
        text = (cat or "").strip()
        if not text or "\n" in text:
            labels.append(None)
            continue
        key = text.casefold()
        key = _ALIASES.get(key, key)
        if key not in _LABEL_OF:
            unmapped.add(text)
            labels.append(None)
            continue
        labels.append(_LABEL_OF[key])
    if unmapped:
        print(f"  unmapped categories in {config}: {sorted(unmapped)[:5]}", flush=True)
    return ds, ok, labels, img_hash, aud_hash


def stage_build(work: Path) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    # test is prepared first so its photographs can be withheld from train
    prepared = {
        "test": _prepare("all_test", "test"),
        "train": _prepare("all_dev", "dev"),
    }

    counts: dict[str, dict[str, int]] = {}
    for name, code in _LANGS.items():
        counts[code] = {}
        test_images: set[str] = set()
        for split, (ds, ok, labels, img_hash, aud_hash) in prepared.items():
            langs = ds["Language"]
            # A photograph carries several questions and each is a separate example, so
            # repeated photographs are kept. Identical recordings are not: they would
            # make the same example appear twice. A photograph in the test split is
            # dropped from train, since the label is a property of the photograph and
            # the official dev and test splits overlap heavily.
            seen_img, seen_aud, rows = set(), set(), []
            for i, lg in enumerate(langs):
                if lg != name or not ok[i] or labels[i] is None:
                    continue
                if aud_hash[i] in seen_aud:
                    continue
                if split == "train" and img_hash[i] in test_images:
                    continue
                seen_img.add(img_hash[i])
                seen_aud.add(aud_hash[i])
                rows.append(i)
            if split == "test":
                test_images = set(seen_img)
            if not rows:
                continue
            sub = ds.select_columns(
                ["image", "native_audio_question", "eng_question", "native_question"]
            ).select(rows)
            sub = sub.rename_column("native_audio_question", "audio")
            sub = sub.add_column("label", [labels[i] for i in rows])
            sub.cast_column("image", Image()).cast_column(
                "audio", Audio(sampling_rate=16000)
            ).to_parquet(str(work / f"{code}_{split}.parquet"))
            counts[code][split] = len(rows)
        print(f"  {code}: {counts[code]}", flush=True)

    (work / "counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print("built", json.dumps(counts))
    return counts


def _card(codes: list[str]) -> str:
    lines = [
        "---",
        f"license: {_LICENSE}",
        "task_categories:",
        "  - audio-classification",
        "  - text-classification",
        "  - image-classification",
        "language:",
        *[f"  - {c}" for c in codes],
        "tags:",
        "  - mteb",
        "  - classification",
        "  - multilingual",
        "configs:",
    ]
    for c in codes:
        lines += [f"  - config_name: {c}", "    data_files:"]
        for split in ("train", "test"):
            lines += [
                f"      - split: {split}",
                f"        path: {c}_{split}.parquet",
            ]
    lines += [
        "---",
        "",
        "# Afri-MCQA cross-modal cultural category classification (MTEB)",
        "",
        "Classify the cultural category of an entry from its photograph and the question",
        "about it spoken by a native speaker, across 16 African languages.",
        "",
        "Labels index this list:",
        "",
        *[f"{i}. {name}" for i, name in enumerate(_CATEGORIES)],
        "",
        f"Source: `{_SOURCE_REPO}` at revision `{_SOURCE_REV[:7]}`, {_LICENSE}. The official",
        "`dev` split is published as `train` and the official `test` split as `test`.",
        "",
        "Built by `scripts/data/afri_mcqa_classification/create_data.py` in the MTEB repo.",
    ]
    return "\n".join(lines) + "\n"


def stage_push(work: Path) -> None:
    api = HfApi()
    api.create_repo(_TARGET, repo_type="dataset", exist_ok=True)
    codes = [c for c in _LANGS.values() if (work / f"{c}_test.parquet").exists()]
    for c in codes:
        for split in ("train", "test"):
            f = work / f"{c}_{split}.parquet"
            if f.exists():
                api.upload_file(
                    path_or_fileobj=str(f),
                    path_in_repo=f.name,
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
    parser.add_argument("--work-dir", type=Path, default=Path("afri_cls_work"))
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    print(f"source: {_SOURCE_REPO}@{_SOURCE_REV[:7]} (license {_LICENSE})")
    if args.stage in ("build", "all"):
        stage_build(args.work_dir)
    if args.stage == "push" or (args.push and args.stage == "all"):
        stage_push(args.work_dir)


if __name__ == "__main__":
    main()
