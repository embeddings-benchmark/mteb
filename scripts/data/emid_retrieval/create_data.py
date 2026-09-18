#!/usr/bin/env python3
"""Package ecnu-aigc/EMID into MTEB audio↔image retrieval and pair classification.

EMID pairs music clips with three emotion-matched images (13 emotion categories).
The upstream Hub release only exposes a train split; we export it as the test split.

Retrieval (instance-level, up to three relevant targets per query):
  - {repo-prefix}-A2I  (audio query → image corpus)
  - {repo-prefix}-I2A  (image query → audio corpus)

Exports a seeded subsample of 1,000 audio anchors with deduplicated media.

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
import hashlib
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
_RETRIEVAL_SEED = 42
_RETRIEVAL_MAX_QUERIES = 1000


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


def _media_bytes(cell) -> bytes | None:
    if cell is None:
        return None
    if isinstance(cell, dict):
        raw = cell.get("bytes")
        if raw:
            return bytes(raw)
        path = cell.get("path")
        if path and Path(path).is_file():
            return Path(path).read_bytes()
    return None


def _media_hash(cell) -> str | None:
    raw = _media_bytes(cell)
    if not raw:
        return None
    return hashlib.md5(raw, usedforsecurity=False).hexdigest()


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


def _canonicalize_retrieval(
    audio_ids: list[str],
    audios: list,
    image_ids: list[str],
    images: list,
    images_by_audio: dict[str, list[str]],
    *,
    seed: int,
    max_queries: int,
) -> tuple[
    list[str],
    list,
    list[str],
    list,
    dict[str, list[str]],
]:
    """Deduplicate media by content hash and subsample audio queries."""
    audio_hash_to_id: dict[str, str] = {}
    audio_id_remap: dict[str, str] = {}
    canonical_audio_ids: list[str] = []
    canonical_audios: list = []

    for audio_id, audio in zip(audio_ids, audios, strict=True):
        media_hash = _media_hash(audio)
        if media_hash is None:
            continue
        canonical_id = audio_hash_to_id.get(media_hash)
        if canonical_id is None:
            audio_hash_to_id[media_hash] = audio_id
            canonical_id = audio_id
            canonical_audio_ids.append(audio_id)
            canonical_audios.append(audio)
        audio_id_remap[audio_id] = canonical_id

    image_hash_to_id: dict[str, str] = {}
    image_id_remap: dict[str, str] = {}
    canonical_image_ids: list[str] = []
    canonical_images: list = []

    for image_id, image in zip(image_ids, images, strict=True):
        media_hash = _media_hash(image)
        if media_hash is None:
            continue
        canonical_id = image_hash_to_id.get(media_hash)
        if canonical_id is None:
            image_hash_to_id[media_hash] = image_id
            canonical_id = image_id
            canonical_image_ids.append(image_id)
            canonical_images.append(image)
        image_id_remap[image_id] = canonical_id

    remapped_images_by_audio: dict[str, list[str]] = {}
    for audio_id, image_ids_for_audio in images_by_audio.items():
        canonical_audio_id = audio_id_remap.get(audio_id)
        if canonical_audio_id is None:
            continue
        if canonical_audio_id not in remapped_images_by_audio:
            remapped_images_by_audio[canonical_audio_id] = []
        seen_images: set[str] = set(remapped_images_by_audio[canonical_audio_id])
        for image_id in image_ids_for_audio:
            canonical_image_id = image_id_remap.get(image_id)
            if canonical_image_id is None or canonical_image_id in seen_images:
                continue
            remapped_images_by_audio[canonical_audio_id].append(canonical_image_id)
            seen_images.add(canonical_image_id)

    remapped_images_by_audio = {
        audio_id: image_ids_for_audio
        for audio_id, image_ids_for_audio in remapped_images_by_audio.items()
        if image_ids_for_audio
    }

    rng = random.Random(seed)
    sampled_audio_ids = list(remapped_images_by_audio)
    rng.shuffle(sampled_audio_ids)
    sampled_audio_ids = sampled_audio_ids[: min(max_queries, len(sampled_audio_ids))]
    sampled_images_by_audio = {
        audio_id: remapped_images_by_audio[audio_id] for audio_id in sampled_audio_ids
    }

    return (
        canonical_audio_ids,
        canonical_audios,
        canonical_image_ids,
        canonical_images,
        sampled_images_by_audio,
    )


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
        f"Expanded to {len(audio_ids)} audios / {len(image_ids)} images "
        f"({len(images_by_audio)} queries with >=1 image)"
    )

    (
        retrieval_audio_ids,
        retrieval_audios,
        retrieval_image_ids,
        retrieval_images,
        retrieval_images_by_audio,
    ) = _canonicalize_retrieval(
        audio_ids,
        audios,
        image_ids,
        images,
        images_by_audio,
        seed=_RETRIEVAL_SEED,
        max_queries=_RETRIEVAL_MAX_QUERIES,
    )
    print(
        f"Retrieval export: {len(retrieval_audio_ids)} unique audios / "
        f"{len(retrieval_image_ids)} unique images / "
        f"{len(retrieval_images_by_audio)} sampled queries"
    )

    id_to_audio = dict(zip(retrieval_audio_ids, retrieval_audios, strict=True))
    id_to_image = dict(zip(retrieval_image_ids, retrieval_images, strict=True))

    used_audio_ids = list(retrieval_images_by_audio)
    used_image_ids: list[str] = []
    seen_image_hashes: set[str] = set()
    for image_ids_for_audio in retrieval_images_by_audio.values():
        for image_id in image_ids_for_audio:
            image_hash = _media_hash(id_to_image[image_id])
            if image_hash is None or image_hash in seen_image_hashes:
                continue
            seen_image_hashes.add(image_hash)
            used_image_ids.append(image_id)

    a2i_corpus = Dataset.from_dict(
        {
            "id": used_image_ids,
            "image": [id_to_image[iid] for iid in used_image_ids],
        }
    ).cast_column("image", Image())
    a2i_queries = Dataset.from_dict(
        {
            "id": used_audio_ids,
            "audio": [id_to_audio[aid] for aid in used_audio_ids],
        }
    ).cast_column("audio", Audio())
    a2i_qrels = {"query-id": [], "corpus-id": [], "score": []}
    for aid, iids in retrieval_images_by_audio.items():
        for iid in iids:
            a2i_qrels["query-id"].append(aid)
            a2i_qrels["corpus-id"].append(iid)
            a2i_qrels["score"].append(1)

    i2a_qrels = {"query-id": [], "corpus-id": [], "score": []}
    seen_image_hashes = set()
    for aid, iids in retrieval_images_by_audio.items():
        for iid in iids:
            image_hash = _media_hash(id_to_image[iid])
            if image_hash is None or image_hash in seen_image_hashes:
                continue
            seen_image_hashes.add(image_hash)
            i2a_qrels["query-id"].append(iid)
            i2a_qrels["corpus-id"].append(aid)
            i2a_qrels["score"].append(1)

    i2a_audio_ids = sorted({aid for aid in i2a_qrels["corpus-id"]})
    i2a_corpus = Dataset.from_dict(
        {
            "id": i2a_audio_ids,
            "audio": [id_to_audio[aid] for aid in i2a_audio_ids],
        }
    ).cast_column("audio", Audio())
    i2a_queries = Dataset.from_dict(
        {
            "id": used_image_ids,
            "image": [id_to_image[iid] for iid in used_image_ids],
        }
    ).cast_column("image", Image())

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
