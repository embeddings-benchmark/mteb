#!/usr/bin/env python3
"""Package KeisukeImoto/MIAO into MTEB audio↔image retrieval format.

MIAO pairs FSD50K sound-event clips with human-drawn onomatopoeic images
Ground truth is class-level: every audio of class C is relevant to every 
image of class C (and vice versa).

Builds two Hub datasets (corpus / queries / qrels configs, test split):
  - {repo-prefix}-A2I  (audio query → image corpus)
  - {repo-prefix}-I2A  (image query → audio corpus)

Images are resized (max side 1024, RGB) so the Hub dump stays manageable.

Usage:
  export HF_TOKEN=...
  uv run python scripts/data/miao_retrieval/create_data.py \\
      --repo-prefix {base_repo}/MIAO \\
      --work-dir /tmp/miao_mteb \\
      --push
"""

from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path

from datasets import Audio, Dataset, DatasetDict, Image
from huggingface_hub import create_repo, snapshot_download
from PIL import Image as PILImage
from tqdm import tqdm

_SOURCE = "KeisukeImoto/MIAO"
_IMAGE_RE = re.compile(
    r"^image/illustrator(?P<ill>\d+)/(?P<label>.+)_set(?P<set>\d+)_"
    r"(?P<fsd>\d+)_illustrator\d+\.png$"
)
_AUDIO_RE = re.compile(r"^audio/set(?P<set>\d+)/(?P<fsd>\d+)\.wav$")


def _resize_rgb(src: Path, dst: Path, max_side: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    with PILImage.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            im = im.resize(
                (int(w * scale), int(h * scale)), PILImage.Resampling.LANCZOS
            )
        im.save(dst, format="JPEG", quality=90)


def _index_media(root: Path) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    """Return label → audio paths and label → image paths."""
    fsd_to_label: dict[str, str] = {}
    images_by_label: dict[str, list[Path]] = defaultdict(list)
    for path in sorted((root / "image").rglob("*.png")):
        rel = path.relative_to(root).as_posix()
        m = _IMAGE_RE.match(rel)
        if not m:
            raise SystemExit(f"Unexpected image path: {rel}")
        label = m.group("label")
        fsd_to_label[m.group("fsd")] = label
        images_by_label[label].append(path)

    audios_by_label: dict[str, list[Path]] = defaultdict(list)
    for path in sorted((root / "audio").rglob("*.wav")):
        rel = path.relative_to(root).as_posix()
        m = _AUDIO_RE.match(rel)
        if not m:
            raise SystemExit(f"Unexpected audio path: {rel}")
        label = fsd_to_label.get(m.group("fsd"))
        if label is None:
            raise SystemExit(f"No class label for audio {rel}")
        audios_by_label[label].append(path)

    labels = sorted(set(images_by_label) | set(audios_by_label))
    for label in labels:
        if label not in images_by_label or label not in audios_by_label:
            raise SystemExit(f"Incomplete class {label}")
    return dict(audios_by_label), dict(images_by_label)


def _push_direction(
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
    else:
        assert out_dir is not None
        out = out_dir / repo_id.replace("/", "__")
        out.mkdir(parents=True, exist_ok=True)
        corpus.save_to_disk(out / "corpus")
        queries.save_to_disk(out / "queries")
        qrels.save_to_disk(out / "qrels")
        print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-prefix", default="Wissam42/MIAO")
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/miao_mteb"))
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    work: Path = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    src = Path(
        snapshot_download(
            _SOURCE,
            repo_type="dataset",
            local_dir=work / "source",
        )
    )
    audios_by_label, images_by_label = _index_media(src)
    print(
        f"Classes={len(audios_by_label)} "
        f"audio={sum(len(v) for v in audios_by_label.values())} "
        f"images={sum(len(v) for v in images_by_label.values())}"
    )

    img_cache = work / "images_jpg"
    image_paths: dict[str, Path] = {}
    for label, paths in tqdm(images_by_label.items(), desc="resize images"):
        for p in paths:
            iid = f"img-{label}-{p.stem}"
            dst = img_cache / f"{iid}.jpg"
            _resize_rgb(p, dst, args.max_side)
            image_paths[iid] = dst

    audio_ids_by_label: dict[str, list[str]] = {}
    audio_path_by_id: dict[str, Path] = {}
    for label, paths in audios_by_label.items():
        ids = []
        for p in paths:
            aid = f"aud-{label}-{p.parent.name}-{p.stem}"
            ids.append(aid)
            audio_path_by_id[aid] = p
        audio_ids_by_label[label] = ids

    image_ids_by_label = {
        label: [f"img-{label}-{p.stem}" for p in paths]
        for label, paths in images_by_label.items()
    }

    # --- A2I: audio queries, image corpus ---
    a2i_corpus = Dataset.from_dict(
        {
            "id": list(image_paths.keys()),
            "image": [str(p) for p in image_paths.values()],
        }
    ).cast_column("image", Image())
    a2i_queries = Dataset.from_dict(
        {
            "id": list(audio_path_by_id.keys()),
            "audio": [str(p) for p in audio_path_by_id.values()],
        }
    ).cast_column("audio", Audio())
    a2i_qrels = {"query-id": [], "corpus-id": [], "score": []}
    for label, qids in audio_ids_by_label.items():
        for qid in qids:
            for cid in image_ids_by_label[label]:
                a2i_qrels["query-id"].append(qid)
                a2i_qrels["corpus-id"].append(cid)
                a2i_qrels["score"].append(1)
    a2i_qrels_ds = Dataset.from_dict(a2i_qrels)

    # --- I2A: image queries, audio corpus ---
    i2a_corpus = Dataset.from_dict(
        {
            "id": list(audio_path_by_id.keys()),
            "audio": [str(p) for p in audio_path_by_id.values()],
        }
    ).cast_column("audio", Audio())
    i2a_queries = Dataset.from_dict(
        {
            "id": list(image_paths.keys()),
            "image": [str(p) for p in image_paths.values()],
        }
    ).cast_column("image", Image())
    i2a_qrels = {"query-id": [], "corpus-id": [], "score": []}
    for label, qids in image_ids_by_label.items():
        for qid in qids:
            for cid in audio_ids_by_label[label]:
                i2a_qrels["query-id"].append(qid)
                i2a_qrels["corpus-id"].append(cid)
                i2a_qrels["score"].append(1)
    i2a_qrels_ds = Dataset.from_dict(i2a_qrels)

    print(
        f"A2I corpus={len(a2i_corpus)} queries={len(a2i_queries)} "
        f"qrels={len(a2i_qrels_ds)}"
    )
    print(
        f"I2A corpus={len(i2a_corpus)} queries={len(i2a_queries)} "
        f"qrels={len(i2a_qrels_ds)}"
    )

    token = None
    if args.push:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise SystemExit("Set HF_TOKEN to push")

    out = None if args.push else work / "mteb_export"
    _push_direction(
        repo_id=f"{args.repo_prefix}-A2I",
        queries=a2i_queries,
        corpus=a2i_corpus,
        qrels=a2i_qrels_ds,
        token=token,
        out_dir=out,
    )
    _push_direction(
        repo_id=f"{args.repo_prefix}-I2A",
        queries=i2a_queries,
        corpus=i2a_corpus,
        qrels=i2a_qrels_ds,
        token=token,
        out_dir=out,
    )
    if args.push:
        print(
            "Pin the Hub commit SHAs in "
            "mteb/tasks/retrieval/eng/miao_retrieval.py TaskMetadata."
        )


if __name__ == "__main__":
    main()
