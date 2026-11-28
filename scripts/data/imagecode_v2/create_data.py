"""Rebuild the ImageCoDe corpus with images in their source order (#5499).

`mteb/imagecode` and `mteb/imagecode-multi` share one corpus whose image IDs were
assigned in unsorted file order, so the `text_k -> img_k` qrels pointed at the wrong
image. This rebuilds the corpus from the authors' source images: `<set>_img_<k>` is
the k-th frame of the set in numeric filename order, as in the official evaluator.
Row order, IDs, schema and the image format are kept; queries and qrels are unchanged.

Download the source files, rebuild, and with --push open a PR with the new corpus on
both dataset repos:

    hf download BennoKrojer/ImageCoDe image-sets.zip valid_data.jsonl --repo-type dataset \
        --revision b7b7a73dbc74b681b384048716232b243ad25a99 --local-dir imagecode
    python scripts/data/imagecode_v2/create_data.py --source-dir imagecode \
        --output-dir imagecode/rebuilt --push
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from PIL import Image

logger = logging.getLogger(__name__)

# Both repos pin the same corpus file; these are the revisions it was replaced in.
REPOS = {
    "mteb/imagecode": "3d13625935ec7e207e7e870e33a033cf51d60626",
    "mteb/imagecode-multi": "0a27b28e13841897d490a40fb88a003f04d7e314",
}
CORPUS_FILE = "corpus-00000-of-00001.parquet"
SOURCE_SHA256 = {
    "image-sets.zip": "6abe316c71ecc764d915ff6d92b18953025414ec56a30ac0aff114f144de869c",
    "valid_data.jsonl": "9296e3b43b5723f4aa6930f6997726d236315f2c5e2aba4fb37e3fe8b9829ebe",
}
FRAMES_PER_SET = 10
SIZE = 256
JPEG_QUALITY = 75  # matches the quantization tables of the old images


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 23):
            digest.update(chunk)
    return digest.hexdigest()


def source_frames(archive: zipfile.ZipFile, image_sets: set[str]) -> dict[str, str]:
    """Map `<set>_img_<k>` to the k-th archive member in frame-number order.

    Video sets keep source frame numbers (img0, img2, ..., img40), so k is the rank
    in numeric order, not the number itself.
    """
    members: dict[str, dict[int, str]] = {image_set: {} for image_set in image_sets}
    for name in archive.namelist():
        parts = name.split("/")  # image-sets/<set>/img<number>.jpg
        if len(parts) == 3 and parts[1] in members and parts[2].endswith(".jpg"):
            number = int(parts[2].removeprefix("img").removesuffix(".jpg"))
            members[parts[1]][number] = name
    frames = {}
    for image_set, by_number in members.items():
        assert len(by_number) == FRAMES_PER_SET, (image_set, sorted(by_number))
        for k, number in enumerate(sorted(by_number)):
            frames[f"{image_set}_img_{k}"] = by_number[number]
    return frames


def to_old_format(data: bytes) -> bytes:
    """Resize the short side to 256 (bilinear), center-crop, encode as JPEG q75."""
    image = Image.open(io.BytesIO(data)).convert("RGB")
    scale = SIZE / min(image.size)
    width, height = round(image.width * scale), round(image.height * scale)
    image = image.resize((width, height), Image.BILINEAR)
    left, top = (width - SIZE) // 2, (height - SIZE) // 2
    image = image.crop((left, top, left + SIZE, top + SIZE))
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=JPEG_QUALITY)
    return out.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    for name, expected in SOURCE_SHA256.items():
        assert sha256(args.source_dir / name) == expected, f"unexpected {name}"
    with (args.source_dir / "valid_data.jsonl").open(encoding="utf-8") as f:
        annotations = [json.loads(line) for line in f]

    # The old corpus lists all ten candidates for every description, so a set with
    # several descriptions repeats; keep that layout and replace only the images.
    ids = [
        f"{row['image_set']}_img_{k}"
        for row in annotations
        for k in range(FRAMES_PER_SET)
    ]
    repo, revision = "mteb/imagecode-multi", REPOS["mteb/imagecode-multi"]
    old = pq.read_table(
        hf_hub_download(repo, CORPUS_FILE, repo_type="dataset", revision=revision)
    )
    assert old.column("id").to_pylist() == ids

    with zipfile.ZipFile(args.source_dir / "image-sets.zip") as archive:
        frames = source_frames(archive, {row["image_set"] for row in annotations})
        sources = [archive.read(member) for member in frames.values()]
    with ThreadPoolExecutor() as pool:  # Pillow releases the GIL
        images = dict(zip(frames, pool.map(to_old_format, sources), strict=True))

    column = pa.array([images[corpus_id] for corpus_id in ids], type=pa.binary())
    index = old.schema.get_field_index("image")
    new = old.set_column(index, old.schema.field("image"), column)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / CORPUS_FILE
    pq.write_table(new, out_path, row_group_size=100)
    # The schema metadata carries the Hugging Face feature types.
    assert pq.read_schema(out_path).equals(old.schema, check_metadata=True)
    logger.info("Wrote %s (sha256 %s)", out_path, sha256(out_path))

    if args.push:
        api = HfApi()
        for repo, revision in REPOS.items():
            commit = api.create_commit(
                repo_id=repo,
                repo_type="dataset",
                operations=[CommitOperationAdd(CORPUS_FILE, str(out_path))],
                commit_message="Fix image order in corpus (mteb#5499)",
                create_pr=True,
                parent_commit=revision,
            )
            logger.info("Opened %s", commit.pr_url)


if __name__ == "__main__":
    main()
