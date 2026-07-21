#!/usr/bin/env python3
"""Package LabROSA Covers80 into MTEB a2a retrieval format.

Expects the official coversongs layout from
https://labrosa.ee.columbia.edu/projects/coversongs/covers80/ :

  covers32k/
    Claudette/
      everly_brothers+...+01-Claudette.mp3
      roy_orbison+...+15-Claudette.mp3
    Come_Together/
      ...

Each song folder is a clique; each mp3 is a recording. A query retrieves the
other recording(s) of the same work.

Usage:
  export HF_TOKEN=...
  uv run python scripts/data/covers80_retrieval/create_data.py \\
      --audio-root path/to/covers32k \\
      --repo-id {repo_id} \\
      --push
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from datasets import Audio, Dataset, DatasetDict
from huggingface_hub import create_repo
from tqdm import tqdm

_AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_")


def _collect_cliques(audio_root: Path) -> dict[str, list[Path]]:
    """Map work-title folder → list of audio files (≥2 kept later)."""
    cliques: dict[str, list[Path]] = {}
    for song_dir in sorted(audio_root.iterdir()):
        if not song_dir.is_dir() or song_dir.name.startswith("."):
            continue
        tracks = sorted(
            p for p in song_dir.iterdir() if p.suffix.lower() in _AUDIO_EXTS
        )
        if tracks:
            cliques[song_dir.name] = tracks
    return cliques


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio-root",
        type=Path,
        required=True,
        help="Path to LabROSA covers32k/ (song folders with mp3s)",
    )
    parser.add_argument("--repo-id", default="wissam-sib/Covers80-A2A")
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    audio_root: Path = args.audio_root.resolve()
    if not audio_root.is_dir():
        raise SystemExit(f"Not a directory: {audio_root}")

    cliques = {
        name: tracks
        for name, tracks in _collect_cliques(audio_root).items()
        if len(tracks) >= 2
    }
    print(f"Found {len(cliques)} song cliques with ≥2 tracks under {audio_root}")
    if not cliques:
        raise SystemExit(
            "No cliques found. Pass --audio-root pointing at covers32k/ "
            "(e.g. coversongs/covers32k)."
        )

    corpus = {"id": [], "audio": []}
    queries = {"id": [], "audio": []}
    qrels = {"query-id": [], "corpus-id": [], "score": []}

    for work, tracks in tqdm(cliques.items(), desc="build"):
        work_slug = _slug(work)
        ids: list[str] = []
        for track in tracks:
            cid = f"{work_slug}__{_slug(track.stem)}"
            corpus["id"].append(cid)
            corpus["audio"].append(str(track))
            ids.append(cid)

        for qid in ids:
            query_id = f"q-{qid}"
            queries["id"].append(query_id)
            queries["audio"].append(corpus["audio"][corpus["id"].index(qid)])
            for tid in ids:
                if tid == qid:
                    continue
                qrels["query-id"].append(query_id)
                qrels["corpus-id"].append(tid)
                qrels["score"].append(1)

    print(
        f"corpus={len(corpus['id'])} queries={len(queries['id'])} "
        f"qrels={len(qrels['query-id'])}"
    )

    corpus_ds = Dataset.from_dict(corpus).cast_column("audio", Audio())
    queries_ds = Dataset.from_dict(queries).cast_column("audio", Audio())
    qrels_ds = Dataset.from_dict(qrels)

    if args.push:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not token:
            raise SystemExit("Set HF_TOKEN to push")
        create_repo(args.repo_id, repo_type="dataset", token=token, exist_ok=True)
        DatasetDict({"test": corpus_ds}).push_to_hub(
            args.repo_id, "corpus", token=token
        )
        DatasetDict({"test": queries_ds}).push_to_hub(
            args.repo_id, "queries", token=token
        )
        DatasetDict({"test": qrels_ds}).push_to_hub(args.repo_id, "qrels", token=token)
        print(f"Pushed {args.repo_id}. Pin the commit SHA in TaskMetadata.")
    else:
        out = Path("/tmp/covers80_mteb_export")
        out.mkdir(exist_ok=True)
        corpus_ds.save_to_disk(out / "corpus")
        queries_ds.save_to_disk(out / "queries")
        qrels_ds.save_to_disk(out / "qrels")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
