"""Copy izhx/UMRB-EncyclopediaVQA with its two truncated corpus images repaired.

Corpus images 14:391 and 14:392 are missing the last bytes of the JPEG, so they
fail to decode. They are loaded with `LOAD_TRUNCATED_IMAGES` and saved again;
all other files are copied as they are:

    python scripts/data/encyclopedia_vqa/create_data.py OUT_DIR
"""

from __future__ import annotations

import io
import shutil
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image, ImageFile

SOURCE = "izhx/UMRB-EncyclopediaVQA"
REVISION = "d6eae4f06e260664eb3f276fd1bdb5d4d4c9f32b"
TARGET = "mteb/UMRB-EncyclopediaVQA"
TRUNCATED_IDS = {"14:391", "14:392"}


def repair(data: bytes) -> bytes:
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    image = Image.open(io.BytesIO(data))
    image.load()
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    out = io.BytesIO()
    image.convert("RGB").save(out, format="JPEG", quality=95)
    return out.getvalue()


def main() -> None:
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    repaired = 0
    for file in api.dataset_info(SOURCE, revision=REVISION).siblings:
        name = file.rfilename
        local = hf_hub_download(SOURCE, name, repo_type="dataset", revision=REVISION)
        table = pq.read_table(local) if name.startswith("corpus") else None
        ids = table["id"].to_pylist() if table else []
        if not TRUNCATED_IDS & set(ids):
            shutil.copyfile(local, out_dir / name)
            continue
        images = table["image"].to_pylist()
        for i, doc_id in enumerate(ids):
            if doc_id in TRUNCATED_IDS:
                images[i]["bytes"] = repair(images[i]["bytes"])
                # raises if the repaired image still doesn't decode
                Image.open(io.BytesIO(images[i]["bytes"])).load()
                repaired += 1
        image_type = table.schema.field("image").type
        table = table.set_column(
            table.column_names.index("image"), "image", pa.array(images, image_type)
        )
        row_group_size = pq.ParquetFile(local).metadata.row_group(0).num_rows
        pq.write_table(table, out_dir / name, row_group_size=row_group_size)
    assert repaired == len(TRUNCATED_IDS)

    api.create_repo(TARGET, repo_type="dataset", exist_ok=True)
    api.upload_folder(repo_id=TARGET, repo_type="dataset", folder_path=out_dir)


if __name__ == "__main__":
    main()
