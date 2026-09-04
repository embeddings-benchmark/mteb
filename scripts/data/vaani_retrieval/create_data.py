#!/usr/bin/env python3
"""Build the multilingual Vaani audio/image retrieval tasks for MTEB.

Project Vaani records spontaneous *image-prompted* speech: a speaker is shown a
photograph and describes it aloud in their own language. Each recording is therefore
grounded in a specific image, which is what makes audio<->image retrieval well defined
without any additional annotation.

The source is not directly usable as a retrieval task. Audio and images live in
separate trees - roughly 3.8 TB of per-language audio shards and a 49 GB pool of
289,676 images - and the only link between them is the `referenceImage` filename on
each audio row. This script resolves that link by first indexing the image pool by
filename (reading only the path column, never the image bytes), then fetching just the
images the sampled audio actually references.

Sampling is deliberate rather than "first N rows". For each language it reads row
groups spread across the shard, caps how many clips any single speaker can contribute,
and drops clips outside a usable duration band. Without the speaker cap a handful of
prolific speakers dominate the smaller languages. Languages that still fall under
`--min-clips` after this are excluded rather than shipped at a size that cannot be
scored reliably.

The published corpus is shared across every language subset, so differences between
subsets reflect the query language rather than differing corpus difficulty.

Examples:
  # Index the image pool once (cached to image_index.json).
  uv run python scripts/data/vaani_retrieval/create_data.py --stage index

  # Sample audio for every language config.
  uv run python scripts/data/vaani_retrieval/create_data.py --stage sample

  # Resolve referenced images, dedupe, and write the dataset locally.
  uv run python scripts/data/vaani_retrieval/create_data.py --stage build --save-to-disk

  # Run every stage and publish with the authenticated Hugging Face account.
  uv run python scripts/data/vaani_retrieval/create_data.py --stage all --push
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Audio, Dataset, Image
from huggingface_hub import HfApi, HfFileSystem

_SOURCE_REPO = "ARTPARK-IISc/Vaani"
_SOURCE_URL = f"https://huggingface.co/datasets/{_SOURCE_REPO}"
_TARGET_REPO = "vnahata/vaani-audio-image-retrieval"
_LICENSE = "cc-by-4.0"
_N_IMAGE_SHARDS = 114
_SPLIT = "test"

# Sampling controls. See the module docstring for why the speaker cap exists.
_TARGET_PER_LANG = 150
_MAX_PER_SPEAKER = 8
_MIN_DUR, _MAX_DUR = 1.5, 30.0
_ROW_GROUPS = 14
_MAX_SHARDS = 3

_AUDIO_COLS = [
    "audio",
    "language",
    "duration",
    "speakerID",
    "gender",
    "state",
    "district",
    "transcript",
    "referenceImage",
]


def _language_shards(api: HfApi) -> dict[str, list[str]]:
    """Map each language config to its audio shards.

    Vaani stores some collections per language (`audio/<lang>/`) and others per
    state and district (`audio/<state>/<district>/`). Only the language-level
    configs carry a usable language label, so the deeper paths are skipped.
    """
    info = api.repo_info(_SOURCE_REPO, repo_type="dataset", files_metadata=True)
    shards: dict[str, list[str]] = defaultdict(list)
    for sibling in info.siblings:
        path = sibling.rfilename
        if (
            path.startswith("audio/")
            and path.endswith(".parquet")
            and len(path.split("/")) == 3
        ):
            shards[path.split("/")[1]].append(path)
    return {k: sorted(v) for k, v in shards.items()}


def stage_index(out_dir: Path) -> dict[str, list[int]]:
    """Index the image pool by filename, reading only the path column."""
    dest = out_dir / "image_index.json"
    if dest.exists():
        return json.loads(dest.read_text(encoding="utf-8"))

    fs = HfFileSystem()
    index: dict[str, list[int]] = {}
    for shard in range(_N_IMAGE_SHARDS):
        path = (
            f"datasets/{_SOURCE_REPO}/images/"
            f"train-{shard:05d}-of-{_N_IMAGE_SHARDS:05d}.parquet"
        )
        with fs.open(path, "rb") as fh:
            table = pq.ParquetFile(fh).read(columns=["image.path"])
        for row, record in enumerate(table.column(0).to_pylist()):
            index[record["path"]] = [shard, row]
        print(
            f"  indexed shard {shard + 1}/{_N_IMAGE_SHARDS} -> {len(index)}", flush=True
        )

    dest.write_text(json.dumps(index), encoding="utf-8")
    return index


def stage_sample(out_dir: Path) -> None:
    """Sample a speaker-diverse subset of clips for each language."""
    audio_dir = out_dir / "audio_sample"
    audio_dir.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()

    for lang, shards in _language_shards(HfApi()).items():
        dest = audio_dir / f"{lang.replace(' ', '_')}.parquet"
        if dest.exists():
            continue
        rows: list[dict] = []
        seen: set[tuple] = set()
        per_speaker: defaultdict[str, int] = defaultdict(int)

        for shard in shards[:_MAX_SHARDS]:
            if len(rows) >= _TARGET_PER_LANG:
                break
            with fs.open(f"datasets/{_SOURCE_REPO}/{shard}", "rb") as fh:
                pf = pq.ParquetFile(fh)
                n_rg = pf.metadata.num_row_groups
                # spread the reads across the shard rather than taking a contiguous block
                picks = sorted(
                    {
                        round(i * (n_rg - 1) / max(_ROW_GROUPS - 1, 1))
                        for i in range(_ROW_GROUPS)
                    }
                )
                for rg in picks:
                    if len(rows) >= _TARGET_PER_LANG:
                        break
                    for r in pf.read_row_group(rg, columns=_AUDIO_COLS).to_pylist():
                        if len(rows) >= _TARGET_PER_LANG:
                            break
                        if not r.get("referenceImage") or not r.get("audio"):
                            continue
                        duration = r.get("duration") or 0
                        if not (_MIN_DUR <= duration <= _MAX_DUR):
                            continue
                        key = (r.get("speakerID"), r.get("referenceImage"), duration)
                        if key in seen:
                            continue
                        speaker = r.get("speakerID") or "?"
                        if per_speaker[speaker] >= _MAX_PER_SPEAKER:
                            continue
                        seen.add(key)
                        per_speaker[speaker] += 1
                        rows.append(r)

        if rows:
            pq.write_table(pa.Table.from_pylist(rows), dest)
            print(
                f"  {lang}: {len(rows)} clips / {len(per_speaker)} speakers", flush=True
            )


def stage_build(out_dir: Path, min_clips: int) -> dict[str, dict]:
    """Resolve referenced images, drop duplicates, and write the dataset."""
    index = stage_index(out_dir)
    data_dir = out_dir / "dataset"
    data_dir.mkdir(parents=True, exist_ok=True)

    lang_rows: dict[str, list[dict]] = {}
    needed: set[str] = set()
    for path in sorted((out_dir / "audio_sample").glob("*.parquet")):
        lang = path.stem.replace("_", " ")
        keep = []
        for r in pq.read_table(path).to_pylist():
            base = (r.get("referenceImage") or "").split("/")[-1]
            if base in index:
                r["_image_id"] = base
                keep.append(r)
        if len(keep) >= min_clips:
            lang_rows[lang] = keep
            needed.update(r["_image_id"] for r in keep)

    # fetch only the referenced images, one shard at a time
    by_shard: dict[int, list[str]] = defaultdict(list)
    for name in needed:
        by_shard[index[name][0]].append(name)

    fs = HfFileSystem()
    images: list[dict] = []
    for shard, names in sorted(by_shard.items()):
        want = {index[n][1]: n for n in names}
        path = (
            f"datasets/{_SOURCE_REPO}/images/"
            f"train-{shard:05d}-of-{_N_IMAGE_SHARDS:05d}.parquet"
        )
        with fs.open(path, "rb") as fh:
            pf = pq.ParquetFile(fh)
            offset = 0
            for rg in range(pf.metadata.num_row_groups):
                n = pf.metadata.row_group(rg).num_rows
                hits = [i for i in want if offset <= i < offset + n]
                if hits:
                    table = pf.read_row_group(rg, columns=["image"]).to_pylist()
                    for i in hits:
                        images.append(
                            {"id": want[i], "image": table[i - offset]["image"]}
                        )
                offset += n

    # Two source images are the same photo under different filenames; collapse them
    # and repoint the affected audio rows at the surviving copy.
    by_hash: dict[str, str] = {}
    remap: dict[str, str] = {}
    unique_images = []
    for r in images:
        digest = hashlib.md5(r["image"]["bytes"]).hexdigest()
        if digest in by_hash:
            remap[r["id"]] = by_hash[digest]
            continue
        by_hash[digest] = r["id"]
        unique_images.append(r)
    pq.write_table(pa.Table.from_pylist(unique_images), data_dir / "images.parquet")

    have = {im["id"] for im in unique_images}
    seen_audio: set[str] = set()
    stats: dict[str, dict] = {}
    for lang, rows in sorted(lang_rows.items()):
        out_rows = []
        for i, r in enumerate(rows):
            image_id = remap.get(r["_image_id"], r["_image_id"])
            if image_id not in have:
                continue
            digest = hashlib.md5(r["audio"]["bytes"]).hexdigest()
            if digest in seen_audio:
                continue
            seen_audio.add(digest)
            out_rows.append(
                {
                    "id": f"{lang.replace(' ', '')}-{i:05d}",
                    "audio": r["audio"],
                    "image_id": image_id,
                    "language": r.get("language"),
                    "speaker_id": r.get("speakerID"),
                    "gender": r.get("gender"),
                    "state": r.get("state"),
                    "district": r.get("district"),
                    "duration": r.get("duration"),
                    "transcript": r.get("transcript") or "",
                }
            )
        if len(out_rows) < min_clips:
            continue
        pq.write_table(
            pa.Table.from_pylist(out_rows),
            data_dir / f"{lang.replace(' ', '_')}.parquet",
        )
        stats[lang] = {
            "clips": len(out_rows),
            "images": len({r["image_id"] for r in out_rows}),
            "speakers": len({r["speaker_id"] for r in out_rows}),
        }

    (out_dir / "dataset_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    print(f"built {len(stats)} languages, {len(unique_images)} images")
    return stats


def stage_push(out_dir: Path) -> None:
    """Publish the image corpus once and each language as its own config."""
    data_dir = out_dir / "dataset"
    api = HfApi()
    api.create_repo(_TARGET_REPO, repo_type="dataset", exist_ok=True)

    images = Dataset.from_parquet(str(data_dir / "images.parquet")).cast_column(
        "image", Image()
    )
    images.push_to_hub(_TARGET_REPO, config_name="images", split=_SPLIT)

    stats = json.loads((out_dir / "dataset_stats.json").read_text(encoding="utf-8"))
    for lang in sorted(stats):
        path = data_dir / f"{lang.replace(' ', '_')}.parquet"
        ds = Dataset.from_parquet(str(path)).cast_column(
            "audio", Audio(sampling_rate=16000)
        )
        ds.push_to_hub(_TARGET_REPO, config_name=lang, split=_SPLIT)
        print(f"  pushed {lang}: {len(ds)} clips", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=["index", "sample", "build", "push", "all"], default="all"
    )
    parser.add_argument("--work-dir", type=Path, default=Path("vaani_work"))
    parser.add_argument(
        "--min-clips",
        type=int,
        default=25,
        help="languages with fewer usable clips are excluded",
    )
    parser.add_argument("--push", action="store_true", help="publish to the Hub")
    parser.add_argument(
        "--save-to-disk", action="store_true", help="keep local parquet output"
    )
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    print(f"source: {_SOURCE_URL} (license {_LICENSE})")

    if args.stage in ("index", "all"):
        stage_index(args.work_dir)
    if args.stage in ("sample", "all"):
        stage_sample(args.work_dir)
    if args.stage in ("build", "all"):
        stage_build(args.work_dir, args.min_clips)
    if args.stage == "push" or (args.push and args.stage == "all"):
        stage_push(args.work_dir)


if __name__ == "__main__":
    main()
