#!/usr/bin/env python3
"""Build the Vaani multilingual speech-transcript retrieval tasks for MTEB.

Uses Vaani's transcribed release rather than the main corpus. That matters for two
reasons: it carries human transcriptions, and unlike the main corpus (which ships only
`train`) it has an official `test` split, so the evaluation set is held out at source
rather than sampled out of training data.

The release has no speaker id, so the per-language sample is spread across row groups
rather than speaker-capped: Vaani records long sessions per speaker and a contiguous read
would return the same voice repeatedly, making retrieval easier than the language is.

Transcripts carry annotation markup - `<noise>`, `<pause>`, `<static>` and similar event
tags, plus `{...}` braces marking code-switched English. The tags are annotation
artefacts and are removed; the braces are unwrapped so the code-switched word survives,
since it was actually spoken.

Byte-identical audio and repeated transcript text are dropped per language: an identical
query marked relevant to only one of the clips it matches would make the qrels ambiguous.

Languages left with fewer than `--min-clips` usable utterances are excluded, which is why
45 of the 64 languages with a test split are kept.

Examples:
  # Sample the per-language evaluation subsets.
  uv run python scripts/data/vaani_speech_text_retrieval/create_data.py --stage sample

  # Clean, deduplicate and publish.
  uv run python scripts/data/vaani_speech_text_retrieval/create_data.py --stage push
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Audio, Dataset
from huggingface_hub import HfApi, HfFileSystem

_SOURCE_REPO = "ARTPARK-IISc/Vaani-transcription-part"
_SOURCE_URL = f"https://huggingface.co/datasets/{_SOURCE_REPO}"
_TARGET_REPO = "vnahata/vaani-speech-text-retrieval"
_LICENSE = "cc-by-4.0"
_SPLIT = "test"

_TARGET_PER_LANG = 100
_MIN_CHARS = 5
_ROW_GROUPS = 30

_COLS = [
    "audio",
    "language",
    "gender",
    "state",
    "district",
    "transcript",
    "referenceImage",
]

_TAG = re.compile(r"<[^>]*>")
_BRACES = re.compile(r"[{}]")
_WS = re.compile(r"\s+")


def clean(text: str) -> str:
    """Strip event tags and unwrap code-switch braces."""
    return _WS.sub(" ", _BRACES.sub("", _TAG.sub(" ", text))).strip()


def _test_shards() -> dict[str, list[str]]:
    info = HfApi().repo_info(_SOURCE_REPO, repo_type="dataset", files_metadata=True)
    shards: dict[str, list[str]] = defaultdict(list)
    for sibling in info.siblings:
        path = sibling.rfilename
        parts = path.split("/")
        if path.startswith("audio/") and path.endswith(".parquet") and len(parts) == 3:
            if parts[2].startswith(f"{_SPLIT}-"):
                shards[parts[1]].append(path)
    return {k: sorted(v) for k, v in shards.items()}


def stage_sample(work: Path) -> None:
    out = work / "audio_sample"
    out.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()
    for lang, files in sorted(_test_shards().items()):
        dest = out / f"{lang}.parquet"
        if dest.exists():
            continue
        rows: list[dict] = []
        for f in files:
            if len(rows) >= _TARGET_PER_LANG:
                break
            with fs.open(f"datasets/{_SOURCE_REPO}/{f}", "rb") as fh:
                pf = pq.ParquetFile(fh)
                n_rg = pf.metadata.num_row_groups
                picks = sorted(
                    {
                        round(k * (n_rg - 1) / max(_ROW_GROUPS - 1, 1))
                        for k in range(_ROW_GROUPS)
                    }
                )
                for rg in picks:
                    if len(rows) >= _TARGET_PER_LANG:
                        break
                    for r in pf.read_row_group(rg, columns=_COLS).to_pylist():
                        if len(rows) >= _TARGET_PER_LANG:
                            break
                        if len(
                            clean(r.get("transcript") or "")
                        ) < _MIN_CHARS or not r.get("audio"):
                            continue
                        rows.append(r)
        if rows:
            pq.write_table(pa.Table.from_pylist(rows), dest)
            print(f"  {lang}: {len(rows)} clips", flush=True)


def stage_push(work: Path, min_clips: int) -> None:
    api = HfApi()
    api.create_repo(_TARGET_REPO, repo_type="dataset", exist_ok=True)
    stats: dict[str, dict] = {}

    for path in sorted((work / "audio_sample").glob("*.parquet")):
        lang = path.stem
        rows = pq.read_table(path).to_pylist()
        seen_audio: set[str] = set()
        seen_text: set[str] = set()
        keep: list[dict] = []
        for r in rows:
            text = clean(r.get("transcript") or "")
            digest = hashlib.md5(r["audio"]["bytes"]).hexdigest()
            if len(text) < _MIN_CHARS or digest in seen_audio or text in seen_text:
                continue
            seen_audio.add(digest)
            seen_text.add(text)
            keep.append(
                {
                    "id": f"{lang}-{len(keep):05d}",
                    "audio": r["audio"],
                    "text": text,
                    "language": r.get("language"),
                    "gender": r.get("gender"),
                    "state": r.get("state"),
                    "district": r.get("district"),
                }
            )
        if len(keep) < min_clips:
            continue
        ds = Dataset.from_list(keep).cast_column("audio", Audio(sampling_rate=16000))
        ds.push_to_hub(_TARGET_REPO, config_name=lang, split=_SPLIT)
        stats[lang] = {"clips": len(keep)}
        print(f"  pushed {lang}: {len(keep)}", flush=True)

    (work / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"{len(stats)} languages, {sum(v['clips'] for v in stats.values())} clips")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["sample", "push", "all"], default="all")
    parser.add_argument("--work-dir", type=Path, default=Path("vaani_st_work"))
    parser.add_argument("--min-clips", type=int, default=25)
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    print(f"source: {_SOURCE_URL} (license {_LICENSE})")

    if args.stage in ("sample", "all"):
        stage_sample(args.work_dir)
    if args.stage in ("push", "all"):
        stage_push(args.work_dir, args.min_clips)


if __name__ == "__main__":
    main()
