"""Repair non-RGB images in the listed Hub datasets, preserving configs and splits.

Authenticate with `hf auth login` or HF_TOKEN. Requires datasets, Pillow and
huggingface_hub. Scans and prepares locally by default; --push uploads changes::

    python scripts/reupload_rgb_datasets.py
    python scripts/reupload_rgb_datasets.py --push
    python scripts/reupload_rgb_datasets.py --datasets mteb/OxfordPets --push

Counts below are reported problem counts, not limits: every image is checked.
RGB images retain their original bytes; non-RGB images are saved as RGB PNGs.
Each conversion is logged and recorded in the JSON report with its original mode.
Unreadable images abort that repository before uploading; no rows are dropped.
Uploads replace affected configs in place. Each config is a separate Hub commit,
so an interrupted upload can leave a repository partially updated. Rerunning is
safe. The JSON report records source and uploaded revisions for metadata updates.
"""

from __future__ import annotations

import argparse
import json
import logging
from io import BytesIO
from pathlib import Path

from datasets import DatasetDict, Image, get_dataset_config_names, load_dataset
from huggingface_hub import HfApi

logger = logging.getLogger(__name__)
DATASETS = {
    "mteb/wds_country211": 30,
    "mteb/OxfordPets": 1,
    "mteb/wds_fer2013": "all",
    "mteb/StanfordCars": 16,
    "mteb/tiny-imagenet": 168,
    "mteb/cub200_retrieval": 4,
    "mteb/docvqa_test_subsampled_beir": 494,
    "mteb/vqa-2": 413,
}


def repair_split(dataset, context):
    image_columns = [
        column
        for column, feature in dataset.features.items()
        if isinstance(feature, Image)
    ]
    if not image_columns:
        return dataset, {}
    for column in image_columns:
        dataset = dataset.cast_column(column, Image(decode=False))

    decoder = Image()
    conversions = []

    def repair_row(row, index):
        images = {}
        for column in image_columns:
            original = row[column]
            if original is None:
                images[column] = None
                continue
            with decoder.decode_example(original) as image:
                if image.mode == "RGB":
                    # Return the storage representation, not a PIL image: the
                    # latter can re-encode JPEGs and change their pixels.
                    images[column] = original
                else:
                    conversion = {
                        "row": index,
                        "column": column,
                        "original_mode": image.mode,
                        "original_format": image.format,
                        "original_path": original.get("path"),
                    }
                    buffer = BytesIO()
                    image.convert("RGB").save(buffer, format="PNG")
                    images[column] = {"bytes": buffer.getvalue(), "path": None}
                    conversions.append(conversion)
                    logger.info(
                        "%s row=%s column=%s: %s (%s, path=%r) -> RGB PNG",
                        context,
                        index,
                        column,
                        image.mode,
                        image.format,
                        original.get("path"),
                    )
        return images

    fixed = dataset.map(
        repair_row,
        with_indices=True,
        load_from_cache_file=False,
        writer_batch_size=32,
        desc=context,
    )
    for column in image_columns:
        fixed = fixed.cast_column(column, Image())
    return fixed, {
        "rows": len(fixed),
        "image_columns": image_columns,
        "conversions": conversions,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS)
    )
    parser.add_argument(
        "--push", action="store_true", help="Upload affected configs in place"
    )
    parser.add_argument(
        "--revision", default="main", help="Source and destination branch"
    )
    parser.add_argument("--report", type=Path, default=Path("rgb_reupload_report.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    api = HfApi()
    report = {}
    failed = []
    for repo in args.datasets:
        entry = report[repo] = {"reported_problems": DATASETS[repo], "configs": {}}
        try:
            source = api.dataset_info(repo, revision=args.revision).sha
            entry["source_revision"] = source
            prepared = {}
            # Prepare ALL configs before making the first remote change.
            for config in get_dataset_config_names(repo, revision=source):
                data = load_dataset(repo, name=config, revision=source)
                fixed = DatasetDict()
                stats = entry["configs"][config] = {}
                for split, dataset in data.items():
                    fixed[split], stats[split] = repair_split(
                        dataset, f"{repo}/{config}/{split}"
                    )
                if any(stats.values()):
                    prepared[config] = fixed
                logger.info("%s/%s: %s", repo, config, stats)
            if args.push and prepared:
                for config, fixed in prepared.items():
                    commit = fixed.push_to_hub(
                        repo,
                        config_name=config,
                        revision=args.revision,
                        commit_message="Convert non-RGB images to RGB",
                    )
                    entry.setdefault("uploaded_configs", {})[config] = commit.oid
                    args.report.write_text(json.dumps(report, indent=2) + "\n")
                entry["uploaded_revision"] = api.dataset_info(
                    repo, revision=args.revision
                ).sha
            entry["status"] = (
                "uploaded"
                if args.push and prepared
                else "prepared"
                if prepared
                else "no_images"
            )
        except Exception as exc:
            logger.exception("%s failed", repo)
            entry.update(status="failed", error=str(exc))
            failed.append(repo)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    logger.info("Report: %s", args.report)
    if failed:
        raise SystemExit(f"Failed repositories: {', '.join(failed)}")


if __name__ == "__main__":
    main()
