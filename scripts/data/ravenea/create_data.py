#!/usr/bin/env python3
"""Normalize the pinned official RAVENEA release for MTEB.

The output keeps the original test image bytes, query IDs, Wikipedia IDs/text,
candidate order, and raw cultural-relevance grades. The qrels ``score`` is the
gain used by the official evaluator, ``2 ** (grade + 3) - 1``. Storing the gain
lets pytrec_eval's standard nDCG reproduce RAVENEA's exponential-gain nDCG.

Without ``--push``, configs are saved below ``--work-dir/mteb_export``. With
``--push``, they are uploaded to the requested Hugging Face dataset repository
and the resulting immutable revision is printed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, Image
from huggingface_hub import DatasetCard, HfApi, create_repo, hf_hub_download
from PIL import Image as PILImage

SOURCE_REPO = "jaagli/ravenea"
SOURCE_REVISION = "ff5a212a9bfa1515f82e0930b37b7d64e3e9ee2e"
SOURCE_FILENAME = "ravenea.zip"
ARCHIVE_SHA256 = "4ef30a2c35ce2ebee39f05128e5a1042f562158ff4a2929d835af3e5f20772bc"
EXPECTED_COUNTS = {
    "metadata.jsonl": 1868,
    "metadata_train.jsonl": 1505,
    "metadata_val.jsonl": 74,
    "metadata_test.jsonl": 161,
    "cic_downstream.jsonl": 49,
    "cvqa_downstream.jsonl": 112,
    "wiki_documents.jsonl": 11396,
}
EXPECTED_TEST_COUNTRIES = {
    "China": 26,
    "India": 30,
    "Indonesia": 18,
    "Korea": 12,
    "Mexico": 28,
    "Nigeria": 11,
    "Russia": 18,
    "Spain": 18,
}
EXPECTED_TEST_CATEGORIES = {
    "Architecture": 48,
    "Art": 5,
    "Companies": 8,
    "Cuisine": 13,
    "Daily Life": 40,
    "History": 12,
    "Nature": 14,
    "Religion": 3,
    "Sports & Recreation": 9,
    "Tools": 1,
    "Transportation": 8,
}
EXPECTED_DUPLICATE_IMAGE_GROUPS = {
    frozenset({"ccub_307_India_88.jpg", "ccub_308_India_89.jpg"})
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe archive path: {member.filename}")
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f"Archive symlink is not allowed: {member.filename}")
        source.extractall(destination)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _image_path(root: Path, source_name: str) -> Path:
    prefix = "./ravenea/"
    if not source_name.startswith(prefix):
        raise ValueError(f"Unexpected image path: {source_name}")
    path = (root / source_name.removeprefix(prefix)).resolve()
    if root.resolve() not in path.parents:
        raise ValueError(f"Unsafe image path: {source_name}")
    return path


def _gain(grade: int) -> int:
    if not -3 <= grade <= 3:
        raise ValueError(f"Culture relevance must be in [-3, 3], got {grade}")
    return 2 ** (grade + 3) - 1


def _validate_release(root: Path) -> dict[str, list[dict[str, Any]]]:
    data = {name: _load_jsonl(root / name) for name in EXPECTED_COUNTS}
    counts = {name: len(rows) for name, rows in data.items()}
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Unexpected release counts: {counts}")

    docs = data["wiki_documents.jsonl"]
    doc_ids = [str(doc["id"]) for doc in docs]
    if len(doc_ids) != len(set(doc_ids)):
        raise ValueError("Duplicate Wikipedia document IDs found")
    if any(not str(doc.get("text", "")).strip() for doc in docs):
        raise ValueError("Empty Wikipedia article found")
    doc_id_set = set(doc_ids)

    metadata_names = [
        "metadata.jsonl",
        "metadata_train.jsonl",
        "metadata_val.jsonl",
        "metadata_test.jsonl",
    ]
    for name in metadata_names:
        rows = data[name]
        query_ids = [str(row["file_name"]) for row in rows]
        if len(query_ids) != len(set(query_ids)):
            raise ValueError(f"Duplicate image query IDs in {name}")
        for row in rows:
            candidates = [str(doc_id) for doc_id in row["enwiki_ids"]]
            grades = [int(grade) for grade in row["culture_relevance"]]
            if len(candidates) != 10 or len(grades) != 10:
                raise ValueError(f"Expected ten judgments for {row['file_name']}")
            if len(candidates) != len(set(candidates)):
                raise ValueError(f"Duplicate candidate document for {row['file_name']}")
            missing = set(candidates) - doc_id_set
            if missing:
                raise ValueError(
                    f"Missing corpus IDs for {row['file_name']}: {missing}"
                )
            for grade in grades:
                _gain(grade)

    split_sets = {
        name: {str(row["file_name"]) for row in data[name]}
        for name in metadata_names[1:]
    }
    split_names = list(split_sets)
    for index, name in enumerate(split_names):
        for other in split_names[index + 1 :]:
            if split_sets[name] & split_sets[other]:
                raise ValueError(f"Split leakage between {name} and {other}")
    full_ids = {str(row["file_name"]) for row in data["metadata.jsonl"]}
    split_union = set().union(*split_sets.values())
    if not split_union <= full_ids:
        raise ValueError("Split query not present in metadata.jsonl")
    excluded = [
        row
        for row in data["metadata.jsonl"]
        if str(row["file_name"]) not in split_union
    ]
    if len(excluded) != 128 or any(
        max(int(grade) for grade in row["culture_relevance"]) > 0 for row in excluded
    ):
        raise ValueError("Unexpected unsplit examples in the official release")

    downstream_ids = {
        str(row["file_name"])
        for name in ("cic_downstream.jsonl", "cvqa_downstream.jsonl")
        for row in data[name]
    }
    if downstream_ids != split_sets["metadata_test.jsonl"]:
        raise ValueError("Downstream test records do not match metadata_test.jsonl")

    images = [path for path in (root / "images").iterdir() if path.is_file()]
    source_image_paths = {
        _image_path(root, str(row["file_name"])) for row in data["metadata.jsonl"]
    }
    if set(images) != source_image_paths:
        raise ValueError("Image directory does not exactly match metadata.jsonl")
    image_hashes: dict[str, list[Path]] = {}
    for image in images:
        with PILImage.open(image) as opened:
            opened.verify()
        with PILImage.open(image) as opened:
            opened.convert("RGB").load()
        digest = _sha256(image)
        image_hashes.setdefault(digest, []).append(image)
    duplicate_image_groups = {
        frozenset(path.name for path in paths)
        for paths in image_hashes.values()
        if len(paths) > 1
    }
    if duplicate_image_groups != EXPECTED_DUPLICATE_IMAGE_GROUPS:
        raise ValueError(f"Unexpected duplicate image groups: {duplicate_image_groups}")

    referenced_docs = {
        str(doc_id) for row in data["metadata.jsonl"] for doc_id in row["enwiki_ids"]
    }
    if referenced_docs != doc_id_set:
        raise ValueError("The corpus is not the exact union of judged documents")

    test = data["metadata_test.jsonl"]
    countries = Counter(str(row["country"]) for row in test)
    categories = Counter(str(row["category"]) for row in test)
    if dict(countries) != EXPECTED_TEST_COUNTRIES:
        raise ValueError(f"Unexpected test country distribution: {dict(countries)}")
    if dict(categories) != EXPECTED_TEST_CATEGORIES:
        raise ValueError(f"Unexpected test category distribution: {dict(categories)}")

    test_grades = Counter(
        int(grade) for row in test for grade in row["culture_relevance"]
    )
    print(f"source={SOURCE_REPO}@{SOURCE_REVISION}")
    print(f"counts={counts}")
    print("split_union=1740 excluded_without_positive_judgment=128")
    print(f"test_relevance_distribution={dict(sorted(test_grades.items()))}")
    print(f"test_country_distribution={dict(sorted(countries.items()))}")
    print(f"test_category_distribution={dict(sorted(categories.items()))}")
    print("images=1868 corrupt=0 duplicate_byte_groups=1 (both train-only)")
    print("corpus=11396 empty_articles=0 duplicate_ids=0 missing_references=0")
    return data


def _build_configs(
    root: Path, data: dict[str, list[dict[str, Any]]]
) -> dict[str, Dataset]:
    test = data["metadata_test.jsonl"]
    corpus = Dataset.from_list(
        [
            {"id": str(doc["id"]), "text": str(doc["text"])}
            for doc in data["wiki_documents.jsonl"]
        ]
    )
    queries = Dataset.from_list(
        [
            {
                "id": str(row["file_name"]),
                "image": str(_image_path(root, str(row["file_name"]))),
                "country": str(row["country"]),
                "category": str(row["category"]),
                "task_type": str(row["task_type"]),
            }
            for row in test
        ]
    ).cast_column("image", Image())

    qrels_rows: list[dict[str, str | int]] = []
    top_ranked_rows: list[dict[str, str | list[str]]] = []
    for row in test:
        query_id = str(row["file_name"])
        candidate_ids = [str(doc_id) for doc_id in row["enwiki_ids"]]
        grades = [int(grade) for grade in row["culture_relevance"]]
        top_ranked_rows.append({"query-id": query_id, "corpus-ids": candidate_ids})
        for doc_id, grade in zip(candidate_ids, grades, strict=True):
            qrels_rows.append(
                {
                    "query-id": query_id,
                    "corpus-id": doc_id,
                    "score": _gain(grade),
                    "culture_relevance": grade,
                }
            )

    return {
        "corpus": corpus,
        "queries": queries,
        "default": Dataset.from_list(qrels_rows),
        "top_ranked": Dataset.from_list(top_ranked_rows),
    }


def _card_body() -> str:
    return f"""
# RAVENEA for MTEB

This repository is a deterministic MTEB normalization of
[`jaagli/ravenea`](https://huggingface.co/datasets/jaagli/ravenea) at revision
`{SOURCE_REVISION}`. It contains the official 161-image test split, the complete
11,396-document Wikipedia corpus, 1,610 human judgments, and each query's ten
official retrieval candidates.

The `culture_relevance` column preserves the source grade from -3 to 3. The
qrels `score` column stores `2 ** (culture_relevance + 3) - 1`, exactly matching
the gain used by the official RAVENEA nDCG implementation.

The source dataset card does not declare a dataset-wide license. The images are
adapted from CVQA and CCUB, and the corpus contains Wikipedia-derived text.
Users should review the upstream dataset and source-item terms before reuse.

Generated by `scripts/data/ravenea/create_data.py` in MTEB. The source archive
SHA-256 is `{ARCHIVE_SHA256}`.
"""


def _save_local(configs: dict[str, Dataset], output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)
    for config, dataset in configs.items():
        DatasetDict({"test": dataset}).save_to_disk(output / config)
    print(f"saved={output}")


def _push(configs: dict[str, Dataset], repo_id: str) -> None:
    create_repo(repo_id, repo_type="dataset", exist_ok=True)
    for config, dataset in configs.items():
        DatasetDict({"test": dataset}).push_to_hub(
            repo_id, config_name=config, commit_message=f"Add RAVENEA {config} config"
        )
    # Preserve the dataset_infos/config metadata written by datasets while
    # replacing only this script's human-readable card body.
    card = DatasetCard.load(repo_id, repo_type="dataset")
    generated_metadata = card.content.split("# RAVENEA for MTEB", maxsplit=1)[0]
    card.content = generated_metadata.rstrip() + "\n" + _card_body()
    card.push_to_hub(
        repo_id,
        repo_type="dataset",
        commit_message="Document RAVENEA normalization",
    )
    revision = HfApi().dataset_info(repo_id).sha
    print(f"pushed={repo_id}@{revision}")


def _source_root(args: argparse.Namespace, work_dir: Path) -> tuple[Path, Path | None]:
    if args.source_dir is not None:
        return args.source_dir.resolve(), None
    archive = Path(
        hf_hub_download(
            repo_id=SOURCE_REPO,
            filename=SOURCE_FILENAME,
            repo_type="dataset",
            revision=SOURCE_REVISION,
        )
    )
    digest = _sha256(archive)
    if digest != ARCHIVE_SHA256:
        raise ValueError(f"Archive SHA-256 mismatch: {digest}")
    temp_dir = Path(tempfile.mkdtemp(prefix="ravenea-extract-", dir=work_dir))
    _safe_extract(archive, temp_dir)
    return temp_dir / "ravenea", temp_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Already-extracted ravenea directory; otherwise download the pinned archive.",
    )
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/ravenea_mteb"))
    parser.add_argument("--repo-id", default="Cerru02/RAVENEA")
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    root, temporary_extract = _source_root(args, work_dir)
    try:
        data = _validate_release(root)
        configs = _build_configs(root, data)
        print(
            "normalized="
            f"queries:{len(configs['queries'])} corpus:{len(configs['corpus'])} "
            f"qrels:{len(configs['default'])} top_ranked:{len(configs['top_ranked'])}"
        )
        if args.push:
            _push(configs, args.repo_id)
        else:
            _save_local(configs, work_dir / "mteb_export")
    finally:
        if temporary_extract is not None:
            shutil.rmtree(temporary_extract)


if __name__ == "__main__":
    main()
