#!/usr/bin/env python3
"""Build the MCIF multilingual audio-visual retrieval tasks for MTEB.

Source is `FBK-MT/MCIF` at a pinned revision, the MCIF benchmark (Papi et al. 2025) of
recorded ACL conference talks. MCIF ships only a `test` split and hosts its own media,
both under CC-BY-4.0, so nothing needs to be scraped or re-licensed here.

MCIF is an instruction-following benchmark rather than a retrieval one. The conversion
keeps only the question-answering samples, which are the sole entries that bind one
question to one clip; the ASR and translation samples group ~33 clips behind a single
reference and cannot be aligned per clip. Within the QA samples two further filters
apply: `qa_type="NA"` is dropped because those questions are deliberately unanswerable,
and `qa_origin="General"` is dropped because those ask about the speaker or affiliation
and so match many talks rather than one. That leaves 133 content-specific questions,
each unique in all four languages, against the full 755-clip corpus.

The instruction prefix ("Answer the following question concisely given the English
content: ") is stripped so the query is the bare question rather than a prompt template
repeated 133 times.

Media is written once with both tracks; each task exposes only the column it is allowed
to see, so the video tasks cannot be solved from the audio track.

Examples:
  # Build the reshaped tables locally.
  uv run python scripts/data/mcif_retrieval/create_data.py --stage build

  # Build and publish.
  uv run python scripts/data/mcif_retrieval/create_data.py --stage all --push
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Audio, Dataset, Video
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

_SOURCE_REPO = "FBK-MT/MCIF"
_SOURCE_REV = "e24065b919758263cfe5d157057278affe76ea7b"
_TARGET = "vnahata/MCIF-retrieval"
_LANGS = ["en", "de", "it", "zh"]
_LICENSE = "cc-by-4.0"

# QA samples only; these two filters remove the entries that cannot act as queries
_KEEP_TYPES = {"AV", "V", "A"}  # drops "NA", which is unanswerable by construction
_KEEP_ORIGINS = {"Transcript", "Abstract"}  # drops "General", which matches many talks


def _strip_prefix(prompt: str, lang: str) -> str:
    """Remove MCIF's fixed instruction prefix, leaving the bare question."""
    sep = "：" if lang == "zh" else ": "
    head, found, tail = prompt.partition(sep)
    return (tail if found else prompt).strip()


def stage_build(work: Path) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    meta = pq.read_table(
        hf_hub_download(
            _SOURCE_REPO,
            "short_fixedprompt/test-00000-of-00001.parquet",
            repo_type="dataset",
            revision=_SOURCE_REV,
        )
    ).to_pylist()

    qa = [
        r
        for r in meta
        if r["metadata"].get("qa_type") in _KEEP_TYPES
        and r["metadata"].get("qa_origin") in _KEEP_ORIGINS
    ]

    local = Path(
        snapshot_download(
            _SOURCE_REPO,
            repo_type="dataset",
            revision=_SOURCE_REV,
            allow_patterns=["MCIF_DATA/SHORT_VIDEOS/*", "MCIF_DATA/SHORT_AUDIOS/*"],
        )
    )
    clips = sorted(p.stem for p in (local / "MCIF_DATA" / "SHORT_VIDEOS").glob("*.mp4"))
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "id": c,
                    "video": str(local / "MCIF_DATA" / "SHORT_VIDEOS" / f"{c}.mp4"),
                    "audio": str(local / "MCIF_DATA" / "SHORT_AUDIOS" / f"{c}.wav"),
                }
                for c in clips
            ]
        ),
        work / "media.parquet",
    )

    rows = []
    for r in qa:
        row = {"id": f"q-{r['id']}", "media_id": Path(r["video"]).stem}
        for lang in _LANGS:
            row[f"text_{lang}"] = _strip_prefix(r[f"prompt_{lang}"], lang)
        rows.append(row)
    pq.write_table(pa.Table.from_pylist(rows), work / "questions.parquet")

    counts = {"clips": len(clips), "questions": len(rows)}
    for lang in _LANGS:
        counts[f"unique_{lang}"] = len({r[f"text_{lang}"] for r in rows})
    (work / "counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print("built", json.dumps(counts))
    return counts


def _card() -> str:
    return (
        "\n".join(
            [
                "---",
                f"license: {_LICENSE}",
                "task_categories:",
                "  - text-to-video",
                "  - text-to-speech",
                "language:",
                *[f"  - {lg}" for lg in _LANGS],
                "tags:",
                "  - mteb",
                "  - retrieval",
                "  - multilingual",
                "configs:",
                "  - config_name: media",
                "    data_files:",
                "      - split: test",
                "        path: media/test-*.parquet",
                "  - config_name: questions",
                "    data_files:",
                "      - split: test",
                "        path: questions/test-*.parquet",
                "---",
                "",
                "# MCIF multilingual audio-visual retrieval (MTEB)",
                "",
                "MCIF reshaped for retrieval: find the recorded conference-talk segment that",
                "answers a question asked in English, German, Italian or Chinese. The questions",
                "are parallel across the four languages while the talks are spoken in English,",
                "so the non-English subsets measure cross-lingual grounding.",
                "",
                f"Source: `{_SOURCE_REPO}` at revision `{_SOURCE_REV[:7]}`, {_LICENSE}. Only the",
                "question-answering samples are used, restricted to answerable and",
                "content-specific ones; the corpus keeps all 755 clips so unquestioned clips act",
                "as distractors.",
                "",
                "Built by `scripts/data/mcif_retrieval/create_data.py` in the MTEB repo.",
            ]
        )
        + "\n"
    )


def stage_push(work: Path) -> None:
    api = HfApi()
    api.create_repo(_TARGET, repo_type="dataset", exist_ok=True)
    (
        Dataset.from_parquet(str(work / "media.parquet"))
        .cast_column("video", Video())
        .cast_column("audio", Audio(sampling_rate=16000))
        .push_to_hub(_TARGET, config_name="media", split="test")
    )
    Dataset.from_parquet(str(work / "questions.parquet")).push_to_hub(
        _TARGET, config_name="questions", split="test"
    )
    api.upload_file(
        path_or_fileobj=io.BytesIO(_card().encode()),
        path_in_repo="README.md",
        repo_id=_TARGET,
        repo_type="dataset",
    )
    print(f"pushed {_TARGET}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["build", "push", "all"], default="all")
    parser.add_argument("--work-dir", type=Path, default=Path("mcif_work"))
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    print(f"source: {_SOURCE_REPO}@{_SOURCE_REV[:7]} (license {_LICENSE})")
    if args.stage in ("build", "all"):
        stage_build(args.work_dir)
    if args.stage == "push" or (args.push and args.stage == "all"):
        stage_push(args.work_dir)


if __name__ == "__main__":
    main()
