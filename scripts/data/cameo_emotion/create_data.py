#!/usr/bin/env python3
"""Build the CAMEO multilingual speech emotion classification task for MTEB.

Source is `amu-cai/CAMEO` at a pinned revision (Christop and Czajka 2025), a curated
collection of emotional speech corpora with emotion labels already harmonised to one
inventory. mteb has thirteen audio emotion tasks and every one is monolingual, so the
point of this build is emotion recognition across languages.

CAMEO names its splits after the source corpora rather than train and test, so the split
here is made by speaker: a speaker appears in train or in test, never both. Emotion is a
property of the voice as much as the utterance, and letting one speaker straddle the split
would let a model match timbre instead of affect.

Three constraints narrow what is usable:

- Only the six emotions common to every included language are kept (anger, fear,
  happiness, neutral, sadness, surprise), so subsets share a label set and are comparable.
- A language needs several speakers to be split at all. German, Russian and Spanish are
  dropped: PAVOQUE is single speaker, and MESD and RESD carry no speaker identifiers.
- CREMA-D and RAVDESS are dropped because mteb already evaluates both on their own, and
  reusing them here would score the same recordings twice.

Recordings are re-encoded to 16 kHz Opus, which keeps the published data small.

Examples:
  # Build the reshaped tables locally.
  uv run python scripts/data/cameo_emotion/create_data.py --stage build

  # Build and publish.
  uv run python scripts/data/cameo_emotion/create_data.py --stage all --push
"""

from __future__ import annotations

import argparse
import io
import json
from math import gcd
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf
from datasets import Audio, Dataset
from huggingface_hub import HfApi, hf_hub_download
from scipy.signal import resample_poly

_SOURCE_REPO = "amu-cai/CAMEO"
_SOURCE_REV = "38e9e968deb4636377e76b28bbb2062f92b898ab"
_TARGET = "vnahata/CAMEO-emotion-classification"
_LICENSE = "cc-by-nc-sa-4.0"

# language -> source corpora, restricted to languages with enough speakers to split
_LANGUAGES = {
    "ben": ("Bengali", ["subesco"]),
    "eng": ("English", ["emns", "enterface", "jl_corpus"]),
    "fra": ("French", ["cafe", "oreau"]),
    "ita": ("Italian", ["emozionalmente"]),
    "pol": ("Polish", ["nemo"]),
}

# the emotions present in every included language; index here is the published label
_EMOTIONS = ["anger", "fear", "happiness", "neutral", "sadness", "surprise"]
_LABEL_OF = {name: i for i, name in enumerate(_EMOTIONS)}

_TEST_FRACTION = 0.35
_PER_LANGUAGE = 1500  # keeps the published audio near a few hundred megabytes
_TARGET_RATE = 16000


def _to_opus(raw: bytes) -> bytes:
    data, rate = sf.read(io.BytesIO(raw), dtype="float32")
    if data.ndim > 1:
        data = data[:, 0]
    if rate != _TARGET_RATE:
        factor = gcd(rate, _TARGET_RATE)
        data = resample_poly(data, _TARGET_RATE // factor, rate // factor)
        rate = _TARGET_RATE
    buf = io.BytesIO()
    sf.write(buf, data, rate, format="OGG", subtype="OPUS")
    return buf.getvalue()


def stage_build(work: Path) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    counts: dict[str, dict[str, int]] = {}

    for code, (language, corpora) in _LANGUAGES.items():
        paths = {s: work / f"{code}_{s}.parquet" for s in ("train", "test")}
        if all(p.exists() for p in paths.values()):
            counts[code] = {s: pq.read_metadata(p).num_rows for s, p in paths.items()}
            print(f"  {code}: cached {counts[code]}", flush=True)
            continue

        # metadata first, so the split is decided before any audio is decoded
        local = {
            corpus: hf_hub_download(
                _SOURCE_REPO,
                f"data/{corpus}-00000-of-00001.parquet",
                repo_type="dataset",
                revision=_SOURCE_REV,
            )
            for corpus in corpora
        }
        rows = []
        for corpus, path in local.items():
            meta = pq.read_table(path).drop(["audio"]).to_pylist()
            rows += [
                {
                    "speaker": f"{corpus}:{r.get('speaker_id')}",
                    "label": _LABEL_OF[r["emotion"]],
                    "corpus": corpus,
                    "index": i,
                }
                for i, r in enumerate(meta)
                if r["emotion"] in _LABEL_OF and r["language"] == language
            ]

        # Speakers hold very different numbers of clips, so test is filled by sample count
        # rather than by speaker count, which otherwise leaves some languages lopsided.
        per_speaker: dict[str, int] = {}
        for r in rows:
            per_speaker[r["speaker"]] = per_speaker.get(r["speaker"], 0) + 1
        target = len(rows) * _TEST_FRACTION
        test_speakers, taken = set(), 0
        for speaker in sorted(per_speaker, key=lambda s: (-per_speaker[s], s)):
            if taken >= target:
                break
            test_speakers.add(speaker)
            taken += per_speaker[speaker]
        wanted = {
            split: [
                r for r in rows if (r["speaker"] in test_speakers) == (split == "test")
            ][:_PER_LANGUAGE]
            for split in ("train", "test")
        }

        counts[code] = {}
        for split, want in wanted.items():
            out = []
            for corpus, path in local.items():
                need = [r for r in want if r["corpus"] == corpus]
                if not need:
                    continue
                audio = pq.read_table(path, columns=["audio"]).column("audio")
                out += [
                    {
                        "audio": {
                            "bytes": _to_opus(audio[r["index"]].as_py()["bytes"]),
                            "path": None,
                        },
                        "label": r["label"],
                    }
                    for r in need
                ]
            Dataset.from_list(out).cast_column(
                "audio", Audio(sampling_rate=_TARGET_RATE)
            ).to_parquet(str(paths[split]))
            counts[code][split] = len(out)
        for path in local.values():
            Path(path).unlink(missing_ok=True)
        counts[code]["speakers"] = len(per_speaker)
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
            lines += [f"      - split: {split}", f"        path: {c}_{split}.parquet"]
    lines += [
        "---",
        "",
        "# CAMEO multilingual speech emotion classification (MTEB)",
        "",
        "Speech emotion recognition across five languages, drawn from the CAMEO collection.",
        "",
        "Labels index this list:",
        "",
        *[f"{i}. {name}" for i, name in enumerate(_EMOTIONS)],
        "",
        f"Source: `{_SOURCE_REPO}` at revision `{_SOURCE_REV[:7]}`, {_LICENSE}. Split by",
        "speaker so no speaker appears in both train and test. Only the six emotions common",
        "to every included language are kept. Audio is 16 kHz Opus.",
        "",
        "Built by `scripts/data/cameo_emotion/create_data.py` in the MTEB repo.",
    ]
    return "\n".join(lines) + "\n"


def stage_push(work: Path) -> None:
    api = HfApi()
    api.create_repo(_TARGET, repo_type="dataset", exist_ok=True)
    codes = [c for c in _LANGUAGES if (work / f"{c}_test.parquet").exists()]
    for code in codes:
        for split in ("train", "test"):
            api.upload_file(
                path_or_fileobj=str(work / f"{code}_{split}.parquet"),
                path_in_repo=f"{code}_{split}.parquet",
                repo_id=_TARGET,
                repo_type="dataset",
            )
        print(f"  pushed {code}", flush=True)
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
    parser.add_argument("--work-dir", type=Path, default=Path("cameo_work"))
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    print(f"source: {_SOURCE_REPO}@{_SOURCE_REV[:7]} (license {_LICENSE})")
    if args.stage in ("build", "all"):
        stage_build(args.work_dir)
    if args.stage == "push" or (args.push and args.stage == "all"):
        stage_push(args.work_dir)


if __name__ == "__main__":
    main()
