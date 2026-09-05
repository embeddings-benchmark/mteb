#!/usr/bin/env python3
"""Build the WAXAL multilingual speech-text retrieval tasks for MTEB.

WAXAL is a corpus of image-prompted natural speech in Sub-Saharan African languages.
Most of its languages have no presence in mteb's existing multilingual audio tasks,
which cover mainly European and South/East Asian languages, so this is a coverage
addition rather than another dataset in an already well-served direction.

Only the official ASR test split is used. Within it the sample is speaker-capped and
spread across row groups: WAXAL records long sessions per speaker, so a contiguous read
would return one voice repeatedly and make the retrieval task easier than the language
actually is. Languages whose test split yields too few usable utterances are dropped
rather than shipped at a size that cannot be scored reliably.

Examples:
  # Sample the per-language evaluation subsets.
  uv run python scripts/data/waxal_retrieval/create_data.py --stage sample

  # Sample and publish.
  uv run python scripts/data/waxal_retrieval/create_data.py --stage all --push
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Audio, Dataset
from huggingface_hub import HfApi, HfFileSystem

_SOURCE_REPO = "google/WaxalNLP"
_SOURCE_URL = f"https://huggingface.co/datasets/{_SOURCE_REPO}"
_TARGET_REPO = "vnahata/waxal-audio-text-retrieval"
_LICENSE = "cc-by-4.0"
_SPLIT = "test"

_TARGET_PER_LANG = 150
_MAX_PER_SPEAKER = 6
_MIN_CHARS = 5
_MIN_CLIPS = 25
_ROW_GROUPS = 40

_COLS = ["id", "speaker_id", "transcription", "language", "gender", "audio"]


def _test_shards(api: HfApi) -> dict[str, list[str]]:
    """Map each ASR language to its test shards."""
    info = api.repo_info(_SOURCE_REPO, repo_type="dataset", files_metadata=True)
    shards: dict[str, list[str]] = defaultdict(list)
    for sibling in info.siblings:
        path = sibling.rfilename
        if "/ASR/" in path and path.endswith(".parquet") and "-test-" in path:
            shards[path.split("/")[2]].append(path)
    return {k: sorted(v) for k, v in shards.items()}


def stage_sample(work: Path) -> dict[str, dict]:
    """Sample a speaker-diverse evaluation subset per language."""
    out = work / "audio_sample"
    out.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()
    manifest: dict[str, dict] = {}
    t0 = time.time()

    for lang, shards in sorted(_test_shards(HfApi()).items()):
        dest = out / f"{lang}.parquet"
        if dest.exists():
            manifest[lang] = {"clips": pq.read_table(dest).num_rows}
            continue
        rows: list[dict] = []
        per_speaker: defaultdict[str, int] = defaultdict(int)
        for shard in shards:
            if len(rows) >= _TARGET_PER_LANG:
                break
            with fs.open(f"datasets/{_SOURCE_REPO}/{shard}", "rb") as fh:
                pf = pq.ParquetFile(fh)
                n_rg = pf.metadata.num_row_groups
                picks = sorted(
                    {
                        round(k * (n_rg - 1) / (_ROW_GROUPS - 1))
                        for k in range(_ROW_GROUPS)
                    }
                )
                for rg in picks:
                    if len(rows) >= _TARGET_PER_LANG:
                        break
                    for r in pf.read_row_group(rg, columns=_COLS).to_pylist():
                        if len(rows) >= _TARGET_PER_LANG:
                            break
                        text = (r.get("transcription") or "").strip()
                        if len(text) < _MIN_CHARS or not r.get("audio"):
                            continue
                        speaker = r.get("speaker_id") or "?"
                        if per_speaker[speaker] >= _MAX_PER_SPEAKER:
                            continue
                        per_speaker[speaker] += 1
                        r["transcription"] = text
                        rows.append(r)

        if len(rows) >= _MIN_CLIPS:
            pq.write_table(pa.Table.from_pylist(rows), dest)
            manifest[lang] = {"clips": len(rows), "speakers": len(per_speaker)}
            print(
                f"  {lang}: {len(rows)} clips / {len(per_speaker)} speakers "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )

    (work / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def stage_push(work: Path) -> None:
    """Publish one config per language with aligned audio and transcription."""
    api = HfApi()
    api.create_repo(_TARGET_REPO, repo_type="dataset", exist_ok=True)
    manifest = json.loads((work / "manifest.json").read_text(encoding="utf-8"))

    for lang in sorted(manifest):
        path = work / "audio_sample" / f"{lang}.parquet"
        rows = pq.read_table(path).to_pylist()
        ds = Dataset.from_dict(
            {
                "id": [f"{lang}-{i:05d}" for i in range(len(rows))],
                "audio": [r["audio"] for r in rows],
                "text": [r["transcription"] for r in rows],
                "speaker_id": [r.get("speaker_id") for r in rows],
                "gender": [r.get("gender") for r in rows],
                "language": [r.get("language") for r in rows],
            }
        ).cast_column("audio", Audio(sampling_rate=16000))
        ds.push_to_hub(_TARGET_REPO, config_name=lang, split=_SPLIT)
        print(f"  pushed {lang}: {len(ds)} clips", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["sample", "push", "all"], default="all")
    parser.add_argument("--work-dir", type=Path, default=Path("waxal_work"))
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    print(f"source: {_SOURCE_URL} (license {_LICENSE})")

    if args.stage in ("sample", "all"):
        stage_sample(args.work_dir)
    if args.stage == "push" or (args.push and args.stage == "all"):
        stage_push(args.work_dir)


if __name__ == "__main__":
    main()
