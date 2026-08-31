"""Build GLAMI-1M text-to-image retrieval data from the official archive.

Source: https://huggingface.co/datasets/glami/glami-1m/tree/befda45d8d4e8b8082bb8a1912d1f9eb9483991c
The source ``GLAMI-1M-dataset--test-only.zip`` is used unchanged (SHA-256
``814c1fb456b86a1a4f1c44fe3fb15c6bc645480e70ea4bfe06bd6a76e29afe9b``).
Extract it before running this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from datasets import Dataset, DatasetDict, Image

GEO_TO_LANGUAGE = {
    "bg": "bg",
    "cz": "cs",
    "ee": "et",
    "es": "es",
    "gr": "el",
    "hr": "hr",
    "hu": "hu",
    "lt": "lt",
    "lv": "lv",
    "ro": "ro",
    "si": "sl",
    "sk": "sk",
    "tr": "tr",
}
EXPECTED_COUNTS = {
    "bg": (11_362, 11_263, 11_295, 11_362),
    "cs": (11_755, 11_720, 11_355, 11_754),
    "el": (3_533, 3_525, 3_518, 3_533),
    "es": (3_405, 3_401, 3_398, 3_405),
    "et": (8_911, 8_834, 8_895, 8_911),
    "hr": (8_117, 8_077, 7_990, 8_117),
    "hu": (8_024, 7_982, 7_871, 8_024),
    "lt": (17_906, 17_827, 17_658, 17_906),
    "lv": (4_181, 4_142, 4_174, 4_181),
    "ro": (2_434, 2_426, 2_400, 2_434),
    "sk": (19_252, 19_213, 18_812, 19_252),
    "sl": (6_332, 6_283, 6_287, 6_332),
    "tr": (10_792, 10_760, 10_252, 10_791),
}
SOURCE_ARCHIVE_SHA256 = (
    "814c1fb456b86a1a4f1c44fe3fb15c6bc645480e70ea4bfe06bd6a76e29afe9b"
)


def build(source_dir: Path) -> tuple[dict[str, dict[str, Dataset]], dict]:
    rows_by_language: dict[str, list[dict[str, str]]] = defaultdict(list)
    with (source_dir / "GLAMI-1M-test.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            rows_by_language[GEO_TO_LANGUAGE[row["geo"]]].append(row)

    configs = {}
    manifest = {}
    for language, rows in sorted(rows_by_language.items()):
        query_ids: dict[tuple[str, str], str] = {}
        image_paths: dict[str, Path] = {}
        qrels = set()

        for row in rows:
            query = (row["name"].strip(), row["description"].strip())
            query_id = query_ids.setdefault(query, f"query-{row['item_id']}")
            corpus_id = f"image-{row['image_id']}"
            image_paths.setdefault(
                corpus_id, source_dir / "images" / f"{row['image_id']}.jpg"
            )
            qrels.add((query_id, corpus_id))

        missing = [path for path in image_paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(missing[0])

        query_fields = list(query_ids)
        queries = Dataset.from_dict(
            {
                "id": list(query_ids.values()),
                "title": [title for title, _ in query_fields],
                "text": [description for _, description in query_fields],
            }
        )
        corpus = Dataset.from_dict(
            {
                "id": list(image_paths.keys()),
                "image": [str(path) for path in image_paths.values()],
            }
        ).cast_column("image", Image())
        qrel_rows = sorted(qrels)
        relevant_docs = Dataset.from_dict(
            {
                "query-id": [query_id for query_id, _ in qrel_rows],
                "corpus-id": [corpus_id for _, corpus_id in qrel_rows],
                "score": [1] * len(qrel_rows),
            }
        )

        counts = (len(rows), len(queries), len(corpus), len(relevant_docs))
        if counts != EXPECTED_COUNTS[language]:
            raise ValueError(f"Unexpected {language} counts: {counts}")
        configs[language] = {
            "queries": queries,
            "corpus": corpus,
            "qrels": relevant_docs,
        }
        manifest[language] = dict(
            zip(("rows", "queries", "corpus", "qrels"), counts, strict=True)
        )

    if sum(entry["rows"] for entry in manifest.values()) != 116_004:
        raise ValueError("The full official test split was not retained")
    return configs, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--repo-id")
    args = parser.parse_args()
    if args.output_dir is None and args.repo_id is None:
        parser.error("provide --output-dir or --repo-id")

    digest = hashlib.sha256()
    with args.source_archive.open("rb") as archive:
        for chunk in iter(lambda: archive.read(1024 * 1024), b""):
            digest.update(chunk)
    archive_sha256 = digest.hexdigest()
    if archive_sha256 != SOURCE_ARCHIVE_SHA256:
        raise ValueError(f"Unexpected source archive SHA-256: {archive_sha256}")

    configs, manifest = build(args.source_dir)
    for language, datasets in configs.items():
        for kind, dataset in datasets.items():
            config_name = f"{language}-{kind}"
            if args.output_dir is not None:
                dataset.save_to_disk(args.output_dir / config_name)
            if args.repo_id is not None:
                DatasetDict({"test": dataset}).push_to_hub(
                    args.repo_id, config_name=config_name
                )

    if args.output_dir is not None:
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
    print(json.dumps(manifest, indent=2))  # noqa: T201


if __name__ == "__main__":
    main()
