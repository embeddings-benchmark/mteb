#!/usr/bin/env python3
"""Build the Indic DiarBench multilingual speaker retrieval tasks for MTEB.

Source is `sarvamai/indic-diarbench` at a pinned revision, a joint diarization and ASR
benchmark covering all 22 scheduled languages of India (Mehendale et al., Interspeech
2026). It ships only a `test` split, so nothing here is drawn from training data.

The corpus is conversational: each row is a chunk of a recording session carrying
human-corrected, time-aligned, speaker-attributed turns. Speaker labels are stable across
the chunks of one session, so a speaker recurs in several chunks and retrieving them is
not a matter of matching the same audio twice.

Turns are cut into clips by their annotated times. Three filters apply:

- clips shorter than 2s or longer than 15s are dropped, since very short turns carry too
  little voice to identify a speaker;
- a turn overlapping any turn of a different speaker is dropped, because the corpus has a
  12.8% overlap ratio and such a clip contains two voices;
- a speaker must keep at least five clips, so every query has a positive in the corpus.

Identity is the session and speaker together, since speaker labels are numbered per
session. The corpus therefore holds other speakers from the same session as the hardest
distractors: same room, same channel, different voice.

Examples:
  # Build the reshaped tables locally.
  uv run python scripts/data/indic_diarbench_speaker/create_data.py --stage build

  # Build and publish.
  uv run python scripts/data/indic_diarbench_speaker/create_data.py --stage all --push
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf
from datasets import Audio, Dataset
from huggingface_hub import HfApi, hf_hub_download

_SOURCE_REPO = "sarvamai/indic-diarbench"
_SOURCE_REV = "92877bad8aab6e598167d91c6ee02aa8ca6ede09"
_TARGET = "vnahata/IndicDiarBench-speaker-retrieval"
_LICENSE = "cc-by-4.0"

# source config name -> published subset name
_LANGS = {
    "Assamese": "asm",
    "Bengali": "ben",
    "Bodo": "brx",
    "Dogri": "doi",
    "Gujarati": "guj",
    "Hindi": "hin",
    "Kannada": "kan",
    "Kashmiri": "kas",
    "Konkani": "kok",
    "Maithili": "mai",
    "Malayalam": "mal",
    "Manipuri": "mni",
    "Marathi": "mar",
    "Nepali": "npi",
    "Odia": "ory",
    "Punjabi": "pan",
    "Sanskrit": "san",
    "Santali": "sat",
    "Sindhi": "snd",
    "Tamil": "tam",
    "Telugu": "tel",
    "Urdu": "urd",
}

_MIN_DUR, _MAX_DUR = 2.0, 15.0
# Turns may overlap a different speaker by up to this much. Conversational turns brush
# against each other at the boundaries; only a longer overlap puts two voices in the clip.
_OVERLAP_TOLERANCE = 0.5
_PER_IDENTITY = 12  # clips kept per speaker
_MAX_IDENTITIES = 60  # per language, keeping the speakers with the most clips
_QUERIES_PER_IDENTITY = 3
_MIN_CLIPS = 5  # queries plus at least two corpus clips
# A language is skipped below this many speakers. Ranking a handful of voices says little
# about a model, and the source gives Maithili and Sindhi only four speakers apiece.
_MIN_IDENTITIES = 5


def _usable_turns(rows: list[dict]) -> list[tuple[str, str, float, float, int]]:
    """Return (identity, sample_id, start, end, row_index) for non-overlapping turns."""
    out = []
    for idx, row in enumerate(rows):
        turns = row["annotated_transcript"]
        for i, turn in enumerate(turns):
            start, end = turn["start_time"], turn["end_time"]
            if not _MIN_DUR <= end - start <= _MAX_DUR:
                continue
            speaker = turn["speaker_id"]
            overlap = max(
                (
                    min(end, other["end_time"]) - max(start, other["start_time"])
                    for j, other in enumerate(turns)
                    if j != i and other["speaker_id"] != speaker
                ),
                default=0.0,
            )
            if overlap > _OVERLAP_TOLERANCE:
                continue
            out.append(
                (f"{row['recording_id']}|{speaker}", row["sample_id"], start, end, idx)
            )
    return out


def _slice(audio_bytes: bytes, start: float, end: float) -> bytes:
    data, rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if data.ndim > 1:
        data = data[:, 0]
    clip = data[int(start * rate) : int(end * rate)]
    buf = io.BytesIO()
    sf.write(buf, clip, rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def stage_build(work: Path) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    counts: dict[str, dict[str, int]] = {}

    for name, code in _LANGS.items():
        q_path, c_path = (
            work / f"{code}_queries.parquet",
            work / f"{code}_corpus.parquet",
        )
        if q_path.exists() and c_path.exists():
            counts[code] = {
                "queries": pq.read_metadata(q_path).num_rows,
                "corpus": pq.read_metadata(c_path).num_rows,
            }
            print(f"  {code}: cached {counts[code]}", flush=True)
            continue

        local = hf_hub_download(
            _SOURCE_REPO,
            f"{name}/test-00000-of-00001.parquet",
            repo_type="dataset",
            revision=_SOURCE_REV,
        )
        table = pq.read_table(local)
        meta = table.drop(["audio"]).to_pylist()
        turns = _usable_turns(meta)

        by_identity: dict[str, list] = {}
        for identity, sample_id, start, end, idx in turns:
            by_identity.setdefault(identity, []).append((sample_id, start, end, idx))
        # keep the speakers with the most turns, so subsets stay balanced in size
        ranked = sorted(
            ((k, v) for k, v in by_identity.items() if len(v) >= _MIN_CLIPS),
            key=lambda kv: (-len(kv[1]), kv[0]),
        )[:_MAX_IDENTITIES]
        if len(ranked) < _MIN_IDENTITIES:
            print(f"  {code}: skipped, only {len(ranked)} speakers", flush=True)
            Path(local).unlink(missing_ok=True)
            continue
        keep = {k: v[:_PER_IDENTITY] for k, v in ranked}

        audio_col = table.column("audio")
        queries, corpus = [], []
        for identity, items in sorted(keep.items()):
            for n, (sample_id, start, end, idx) in enumerate(items):
                raw = audio_col[idx].as_py()["bytes"]
                # identity has to be part of the id: one chunk holds turns from several
                # speakers, so the sample id alone collides across them
                row = {
                    "id": f"{code}-{identity.replace('|', '-')}-{n}",
                    "audio": {"bytes": _slice(raw, start, end), "path": None},
                    "identity": identity,
                }
                (queries if n < _QUERIES_PER_IDENTITY else corpus).append(row)

        for rows, path in ((queries, q_path), (corpus, c_path)):
            Dataset.from_list(rows).cast_column(
                "audio", Audio(sampling_rate=16000)
            ).to_parquet(str(path))

        counts[code] = {
            "queries": len(queries),
            "corpus": len(corpus),
            "speakers": len(keep),
        }
        print(f"  {code}: {counts[code]}", flush=True)
        Path(local).unlink(missing_ok=True)

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
        "  - retrieval",
        "  - multilingual",
        "configs:",
    ]
    for c in codes:
        for kind in ("queries", "corpus"):
            lines += [
                f"  - config_name: {c}-{kind}",
                "    data_files:",
                "      - split: test",
                f"        path: {c}_{kind}.parquet",
            ]
    lines += [
        "---",
        "",
        "# Indic DiarBench speaker retrieval (MTEB)",
        "",
        "Indic DiarBench reshaped for speaker retrieval across the 22 scheduled languages",
        "of India: given a clip of one speaker, find other clips of that same speaker.",
        "",
        f"Source: `{_SOURCE_REPO}` at revision `{_SOURCE_REV[:7]}`, {_LICENSE}, official",
        "`test` split. Turns are cut by their annotated times, restricted to 2 to 15",
        "seconds, and turns overlapping a different speaker are dropped. `identity` pairs",
        "the recording session with the speaker, because speaker labels are numbered per",
        "session.",
        "",
        "Built by `scripts/data/indic_diarbench_speaker/create_data.py` in the MTEB repo.",
    ]
    return "\n".join(lines) + "\n"


def stage_push(work: Path) -> None:
    api = HfApi()
    api.create_repo(_TARGET, repo_type="dataset", exist_ok=True)
    codes = [c for c in _LANGS.values() if (work / f"{c}_queries.parquet").exists()]
    for c in codes:
        for kind in ("queries", "corpus"):
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
    parser.add_argument("--work-dir", type=Path, default=Path("indic_diar_work"))
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    print(f"source: {_SOURCE_REPO}@{_SOURCE_REV[:7]} (license {_LICENSE})")
    if args.stage in ("build", "all"):
        stage_build(args.work_dir)
    if args.stage == "push" or (args.push and args.stage == "all"):
        stage_push(args.work_dir)


if __name__ == "__main__":
    main()
