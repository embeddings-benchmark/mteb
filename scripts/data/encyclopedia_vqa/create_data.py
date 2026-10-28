"""Copy izhx/UMRB-EncyclopediaVQA with its two truncated corpus images repaired.

Corpus images `14:391` and `14:392` are JPEGs missing the last 6 bytes of scan
data, so they fail a strict decode. They are loaded with
`ImageFile.LOAD_TRUNCATED_IMAGES = True` and re-encoded as JPEG; every other
file is copied unchanged:

    python scripts/data/encyclopedia_vqa/create_data.py OUT_DIR --repo-id mteb/UMRB-EncyclopediaVQA
"""

from __future__ import annotations

import argparse
import io
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image, ImageFile

SOURCE = "izhx/UMRB-EncyclopediaVQA"
REVISION = "d6eae4f06e260664eb3f276fd1bdb5d4d4c9f32b"
TRUNCATED_IDS = {"14:391", "14:392"}


def _decodes(data: bytes) -> bool:
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    try:
        Image.open(io.BytesIO(data)).load()
    except OSError:
        return False
    return True


def _repair(data: bytes) -> bytes:
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    image = Image.open(io.BytesIO(data))
    image.load()
    out = io.BytesIO()
    image.convert("RGB").save(out, format="JPEG", quality=95)
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    return out.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--repo-id", default="mteb/UMRB-EncyclopediaVQA")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    repaired = 0
    for sibling in api.dataset_info(SOURCE, revision=REVISION).siblings:
        name = sibling.rfilename
        local = Path(
            hf_hub_download(SOURCE, name, repo_type="dataset", revision=REVISION)
        )
        target = args.out_dir / name
        table = pq.read_table(local) if name.startswith("corpus") else None
        rows = (
            [i for i, x in enumerate(table["id"].to_pylist()) if x in TRUNCATED_IDS]
            if table
            else []
        )
        if not rows:
            shutil.copyfile(local, target)
            continue
        images = table["image"].to_pylist()
        for i in rows:
            assert not _decodes(images[i]["bytes"])
            images[i] = {**images[i], "bytes": _repair(images[i]["bytes"])}
            assert _decodes(images[i]["bytes"])
            repaired += 1
        table = table.set_column(
            table.column_names.index("image"),
            "image",
            pa.array(images, type=table.schema.field("image").type),
        )
        row_group_size = pq.ParquetFile(local).metadata.row_group(0).num_rows
        pq.write_table(
            table.replace_schema_metadata(table.schema.metadata),
            target,
            row_group_size=row_group_size,
        )
    assert repaired == len(TRUNCATED_IDS)

    api.create_repo(args.repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        repo_id=args.repo_id, repo_type="dataset", folder_path=str(args.out_dir)
    )


if __name__ == "__main__":
    main()
