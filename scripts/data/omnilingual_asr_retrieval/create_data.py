#!/usr/bin/env python3
"""Build the Omnilingual ASR speech-text retrieval tasks for MTEB.

Source is `facebook/omnilingual-asr-corpus` at a pinned revision (Meta, 2025), a read
speech corpus with human transcriptions covering 348 languages. It ships an official
`test` split, so nothing here is drawn from training data.

Every audio task currently in mteb reaches 165 distinct languages between them. 302 of
the 308 languages in this corpus appear in none of them, so the point of this build is
language coverage rather than another English benchmark.

Languages are chosen by a fixed rule: absent from every existing mteb audio task, test
shard between 120MB and 320MB so each has enough recordings without a single language
dominating the download, then taken round-robin across writing systems so the selection
is not one script. Recordings are re-encoded from FLAC to Opus, which is about eleven
times smaller and keeps the published dataset near a gigabyte.

Examples:
  # Build the reshaped tables locally.
  uv run python scripts/data/omnilingual_asr_retrieval/create_data.py --stage build

  # Build and publish.
  uv run python scripts/data/omnilingual_asr_retrieval/create_data.py --stage all --push
"""

from __future__ import annotations

import argparse
import io
import json
from collections import defaultdict
from math import gcd
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from datasets import Audio, Dataset
from scipy.signal import resample_poly
from huggingface_hub import HfApi, hf_hub_download

_SOURCE_REPO = "facebook/omnilingual-asr-corpus"
_SOURCE_REV = "8648ba8946377697b427ae952076e49fc0e5e44d"
_TARGET = "vnahata/OmnilingualASR-retrieval"
_LICENSE = "cc-by-4.0"

_MIN_SIZE, _MAX_SIZE = 120e6, 320e6
_N_LANGUAGES = 50
_MIN_ROWS = 40  # a language needs a corpus worth ranking


def _covered_languages() -> set[str]:
    """ISO codes already reachable through some mteb audio task."""
    import mteb

    covered = set()
    for task in mteb.get_tasks():
        if "audio" not in (task.metadata.modalities or []):
            continue
        langs = task.metadata.eval_langs
        values = (
            [v for vs in langs.values() for v in vs]
            if isinstance(langs, dict)
            else list(langs)
        )
        covered.update(v.split("-")[0] for v in values)
    return covered


def select_languages() -> list[str]:
    covered = _covered_languages()
    info = HfApi().repo_info(
        _SOURCE_REPO, repo_type="dataset", revision=_SOURCE_REV, files_metadata=True
    )
    # a language can be split over several shards, so size is summed per config
    total: dict[str, int] = defaultdict(int)
    for sibling in info.siblings:
        if "/test-" in sibling.rfilename:
            total[sibling.rfilename.split("/")[1]] += sibling.size or 0

    by_script: dict[str, list[str]] = defaultdict(list)
    for config, size in total.items():
        iso, _, script = config.partition("_")
        if iso in covered or not _MIN_SIZE <= size <= _MAX_SIZE:
            continue
        by_script[script].append(config)

    for configs in by_script.values():
        configs.sort()
    chosen: list[str] = []
    while len(chosen) < _N_LANGUAGES:
        added = False
        for script in sorted(by_script):
            if by_script[script]:
                chosen.append(by_script[script].pop(0))
                added = True
                if len(chosen) == _N_LANGUAGES:
                    break
        if not added:
            break
    return sorted(chosen)


_TARGET_RATE = 16000


def _to_opus(raw: bytes) -> bytes:
    data, rate = sf.read(io.BytesIO(raw), dtype="float32")
    if data.ndim > 1:
        data = data[:, 0]
    # Recordings arrive at several rates and Opus takes only a few of them, so everything
    # is resampled to the 16 kHz the task declares.
    if rate != _TARGET_RATE:
        factor = gcd(rate, _TARGET_RATE)
        data = resample_poly(data, _TARGET_RATE // factor, rate // factor)
        rate = _TARGET_RATE
    buf = io.BytesIO()
    sf.write(buf, data, rate, format="OGG", subtype="OPUS")
    return buf.getvalue()


def _test_shards() -> dict[str, list[str]]:
    """config -> its test shards; a language is not always a single file."""
    shards: dict[str, list[str]] = defaultdict(list)
    for name in HfApi().list_repo_files(
        _SOURCE_REPO, repo_type="dataset", revision=_SOURCE_REV
    ):
        if "/test-" in name:
            shards[name.split("/")[1]].append(name)
    return {k: sorted(v) for k, v in shards.items()}


def stage_build(work: Path) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    configs = select_languages()
    shards = _test_shards()
    print(f"selected {len(configs)} languages", flush=True)

    counts: dict[str, int] = {}
    for config in configs:
        code = config.split("_")[0]
        out = work / f"{code}.parquet"
        if out.exists():
            counts[code] = pq.read_metadata(out).num_rows
            print(f"  {code}: cached {counts[code]}", flush=True)
            continue

        locals_ = [
            hf_hub_download(
                _SOURCE_REPO, name, repo_type="dataset", revision=_SOURCE_REV
            )
            for name in shards[config]
        ]
        table = pa.concat_tables([pq.read_table(p) for p in locals_])
        texts = table.column("raw_text").to_pylist()
        audio = table.column("audio")

        rows, seen = [], set()
        for i, text in enumerate(texts):
            clean = (text or "").strip()
            # a repeated transcript would be relevant to only one of its recordings
            if not clean or clean in seen:
                continue
            seen.add(clean)
            rows.append(
                {
                    "id": f"{code}-{i}",
                    "audio": {
                        "bytes": _to_opus(audio[i].as_py()["bytes"]),
                        "path": None,
                    },
                    "text": clean,
                }
            )

        for path in locals_:
            Path(path).unlink(missing_ok=True)
        if len(rows) < _MIN_ROWS:
            print(f"  {code}: skipped, only {len(rows)} usable rows", flush=True)
            continue

        Dataset.from_list(rows).cast_column(
            "audio", Audio(sampling_rate=16000)
        ).to_parquet(str(out))
        counts[code] = len(rows)
        print(f"  {code}: {len(rows)}", flush=True)

    (work / "counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print("built", json.dumps(counts))
    return counts


def _card(codes: list[str]) -> str:
    lines = [
        "---",
        f"license: {_LICENSE}",
        "task_categories:",
        "  - automatic-speech-recognition",
        "language:",
        *[f"  - {c}" for c in codes],
        "tags:",
        "  - mteb",
        "  - retrieval",
        "  - multilingual",
        "configs:",
    ]
    for c in codes:
        lines += [
            f"  - config_name: {c}",
            "    data_files:",
            "      - split: test",
            f"        path: {c}.parquet",
        ]
    lines += [
        "---",
        "",
        "# Omnilingual ASR speech-text retrieval (MTEB)",
        "",
        "Read speech paired with its human transcription, for languages that no existing",
        "MTEB audio task covers.",
        "",
        f"Source: `{_SOURCE_REPO}` at revision `{_SOURCE_REV[:7]}`, {_LICENSE}, official",
        "`test` split. Recordings are re-encoded from FLAC to Opus at 16 kHz. Repeated",
        "transcripts are dropped, since one would otherwise be relevant to several",
        "recordings while only one is marked correct.",
        "",
        "Built by `scripts/data/omnilingual_asr_retrieval/create_data.py` in the MTEB repo.",
    ]
    return "\n".join(lines) + "\n"


def stage_push(work: Path) -> None:
    api = HfApi()
    api.create_repo(_TARGET, repo_type="dataset", exist_ok=True)
    codes = sorted(p.stem for p in work.glob("*.parquet"))
    for code in codes:
        api.upload_file(
            path_or_fileobj=str(work / f"{code}.parquet"),
            path_in_repo=f"{code}.parquet",
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
    parser.add_argument("--work-dir", type=Path, default=Path("omni_work"))
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    print(f"source: {_SOURCE_REPO}@{_SOURCE_REV[:7]} (license {_LICENSE})")
    if args.stage in ("build", "all"):
        stage_build(args.work_dir)
    if args.stage == "push" or (args.push and args.stage == "all"):
        stage_push(args.work_dir)


if __name__ == "__main__":
    main()
