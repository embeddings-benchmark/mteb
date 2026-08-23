#!/usr/bin/env python3
"""Package ecnu-aigc/EMID into MTEB audio↔image retrieval and pair classification.

EMID pairs music clips with three emotion-matched images (13 emotion categories).
The upstream Hub release only exposes a train split; we export it as the test split.

Retrieval (instance-level, up to three relevant targets per query):
  - {repo-prefix}-A2I  (audio query → image corpus)
  - {repo-prefix}-I2A  (image query → audio corpus)

Pair classification (emotion-aligned vs mismatched audio–image pairs):
  - {repo-prefix}-PC-AI

Usage:
  export HF_TOKEN=...
  uv run python scripts/data/emid_retrieval/create_data.py \\
      --repo-prefix Wissam42/EMID \\
      --work-dir /tmp/emid_mteb \\
      --push
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

from datasets import Audio, Dataset, DatasetDict, Image, load_dataset
from huggingface_hub import create_repo
from tqdm import tqdm

_SOURCE = "ecnu-aigc/EMID"
_AUDIO_COL = "Audio_Filename"
_IMAGE_COLS = ("Image1_filename", "Image2_filename", "Image3_filename")
_PC_SEED = 42
_PC_MAX_PER_SIDE = 2048


def _push_retrieval_direction(
    *,
    repo_id: str,
    queries: Dataset,
    corpus: Dataset,
    qrels: Dataset,
    token: str | None,
    out_dir: Path | None,
) -> None:
    if token:
        create_repo(repo_id, repo_type="dataset", token=token, exist_ok=True)
        DatasetDict({"test": corpus}).push_to_hub(repo_id, "corpus", token=token)
        DatasetDict({"test": queries}).push_to_hub(repo_id, "queries", token=token)
        DatasetDict({"test": qrels}).push_to_hub(repo_id, "qrels", token=token)
        print(f"Pushed {repo_id}")
        return

    assert out_dir is not None
    out = out_dir / repo_id.replace("/", "__")
    out.mkdir(parents=True, exist_ok=True)
    corpus.save_to_disk(out / "corpus")
    queries.save_to_disk(out / "queries")
    qrels.save_to_disk(out / "qrels")
    print(f"Wrote {out}")


def _media_ok(cell) -> bool:
    """True if an Audio/Image struct has non-empty bytes or an existing path."""
    if cell is None:
        return False
    if isinstance(cell, dict):
        raw = cell.get("bytes")
        if raw is not None and len(raw) > 0:
            return True
        path = cell.get("path")
        return bool(path) and Path(path).is_file()
    return False


def _expand_rows(
    ds,
) -> tuple[list[str], list, list[str], list, list[str], dict[str, list[str]]]:
    """Expand EMID rows; skip clips with empty/corrupt audio or missing images.

    Media columns must already be cast with ``decode=False`` so we never hit
    torchcodec on empty buffers (that crash is what broke the previous run).
    """
    audio_ids: list[str] = []
    audios: list = []
    image_ids: list[str] = []
    images: list = []
    emotions: list[str] = []
    images_by_audio: dict[str, list[str]] = {}
    skipped = 0

    for idx, row in enumerate(tqdm(ds, desc="expand EMID", total=len(ds))):
        audio = row[_AUDIO_COL]
        if not _media_ok(audio):
            skipped += 1
            continue

        imgs = []
        for col in _IMAGE_COLS:
            cell = row[col]
            if _media_ok(cell):
                imgs.append(cell)
        if not imgs:
            skipped += 1
            continue

        aid = f"aud-{idx}"
        audio_ids.append(aid)
        audios.append(audio)
        emotions.append(row["emotion"])
        images_by_audio[aid] = []
        for j, cell in enumerate(imgs, start=1):
            iid = f"img-{idx}-{j}"
            image_ids.append(iid)
            images.append(cell)
            images_by_audio[aid].append(iid)

    if skipped:
        print(f"Skipped {skipped} rows with empty/corrupt audio or no images")
    if not audio_ids:
        raise SystemExit("No valid EMID rows after filtering empty media")
    return audio_ids, audios, image_ids, images, emotions, images_by_audio


def _build_pc(
    audio_ids: list[str],
    audios,
    image_ids: list[str],
    images,
    emotions: list[str],
    images_by_audio: dict[str, list[str]],
    seed: int,
    max_per_side: int,
) -> Dataset:
    rng = random.Random(seed)
    audio_to_emotion = dict(zip(audio_ids, emotions, strict=True))

    by_emotion: dict[str, list[str]] = {}
    for aid, emotion in audio_to_emotion.items():
        by_emotion.setdefault(emotion, []).append(aid)

    pos_pairs: list[tuple[str, str]] = []
    neg_pairs: list[tuple[str, str]] = []
    for aid, iids in images_by_audio.items():
        for iid in iids:
            if len(pos_pairs) < max_per_side:
                pos_pairs.append((aid, iid))
        emotion = audio_to_emotion[aid]
        other_emotions = [e for e in by_emotion if e != emotion]
        if other_emotions and len(neg_pairs) < max_per_side:
            other_aid = rng.choice(
                [x for e in rng.sample(other_emotions, 1) for x in by_emotion[e]]
            )
            other_iid = rng.choice(images_by_audio[other_aid])
            neg_pairs.append((aid, other_iid))
        if len(pos_pairs) >= max_per_side and len(neg_pairs) >= max_per_side:
            break

    n = min(len(pos_pairs), len(neg_pairs))
    pairs = [(a, i, 1) for a, i in pos_pairs[:n]] + [
        (a, i, 0) for a, i in neg_pairs[:n]
    ]
    rng.shuffle(pairs)

    id_to_audio = dict(zip(audio_ids, audios, strict=True))
    id_to_image = dict(zip(image_ids, images, strict=True))
    return (
        Dataset.from_dict(
            {
                "audio": [id_to_audio[a] for a, _, _ in pairs],
                "image": [id_to_image[i] for _, i, _ in pairs],
                "label": [label for *_, label in pairs],
            }
        )
        .cast_column("audio", Audio())
        .cast_column("image", Image())
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-prefix", default="Wissam42/EMID")
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/emid_mteb"))
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(_SOURCE, split="train")
    # Critical: do not decode during iteration — some EMID rows have empty audio
    # bytes and torchcodec raises ValueError on zero-length buffers.
    ds = ds.cast_column(_AUDIO_COL, Audio(decode=False))
    for col in _IMAGE_COLS:
        ds = ds.cast_column(col, Image(decode=False))
    print(f"Loaded {len(ds)} rows from {_SOURCE} (exported as test)")

    audio_ids, audios, image_ids, images, emotions, images_by_audio = _expand_rows(ds)
    print(
        f"Kept {len(audio_ids)} audios / {len(image_ids)} images "
        f"({len(images_by_audio)} queries with >=1 image)"
    )

    a2i_corpus = Dataset.from_dict({"id": image_ids, "image": images}).cast_column(
        "image", Image()
    )
    a2i_queries = Dataset.from_dict({"id": audio_ids, "audio": audios}).cast_column(
        "audio", Audio()
    )
    a2i_qrels = {"query-id": [], "corpus-id": [], "score": []}
    for aid, iids in images_by_audio.items():
        for iid in iids:
            a2i_qrels["query-id"].append(aid)
            a2i_qrels["corpus-id"].append(iid)
            a2i_qrels["score"].append(1)

    i2a_corpus = Dataset.from_dict({"id": audio_ids, "audio": audios}).cast_column(
        "audio", Audio()
    )
    i2a_queries = Dataset.from_dict({"id": image_ids, "image": images}).cast_column(
        "image", Image()
    )
    i2a_qrels = {"query-id": [], "corpus-id": [], "score": []}
    for aid, iids in images_by_audio.items():
        for iid in iids:
            i2a_qrels["query-id"].append(iid)
            i2a_qrels["corpus-id"].append(aid)
            i2a_qrels["score"].append(1)

    pc_ds = _build_pc(
        audio_ids,
        audios,
        image_ids,
        images,
        emotions,
        images_by_audio,
        _PC_SEED,
        _PC_MAX_PER_SIDE,
    )
    print(f"PC pairs={len(pc_ds)} pos={sum(pc_ds['label'])}")

    token = os.environ.get("HF_TOKEN") if args.push else None
    if args.push and not token:
        raise SystemExit("Set HF_TOKEN to push")

    out = None if args.push else work / "mteb_export"
    _push_retrieval_direction(
        repo_id=f"{args.repo_prefix}-A2I",
        queries=a2i_queries,
        corpus=a2i_corpus,
        qrels=Dataset.from_dict(a2i_qrels),
        token=token,
        out_dir=out,
    )
    _push_retrieval_direction(
        repo_id=f"{args.repo_prefix}-I2A",
        queries=i2a_queries,
        corpus=i2a_corpus,
        qrels=Dataset.from_dict(i2a_qrels),
        token=token,
        out_dir=out,
    )

    if args.push:
        repo_id = f"{args.repo_prefix}-PC-AI"
        create_repo(repo_id, repo_type="dataset", token=token, exist_ok=True)
        pc_ds.push_to_hub(repo_id, split="test", token=token)
        print(f"Pushed {repo_id}")
    else:
        pc_out = work / "mteb_export" / (args.repo_prefix.replace("/", "__") + "-PC-AI")
        pc_out.mkdir(parents=True, exist_ok=True)
        pc_ds.save_to_disk(pc_out)
        print(f"Wrote {pc_out}")


if __name__ == "__main__":
    main()
