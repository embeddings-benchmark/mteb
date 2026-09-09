"""Build and verify the native MTEB MVL-SIB sentence-to-image artifact.

The source revision, official generator, and output semantics are fixed. By
default this writes a local artifact only. ``--push`` uploads it privately unless
``--visibility public`` is explicitly selected.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from io import BytesIO
from pathlib import Path
from types import ModuleType
from typing import Any

import pyarrow.parquet as pq
from datasets import (
    Dataset,
    Features,
    Image,
    Sequence,
    Value,
    get_dataset_config_names,
    load_dataset,
)
from huggingface_hub import HfApi, snapshot_download
from PIL import Image as PILImage

SOURCE_REPO = "WueNLP/mvl-sib"
SOURCE_REVISION = "1df5974e8fb204e91ee70cef2b3b7196a14b390f"
OFFICIAL_SCRIPT_SHA256 = (
    "c926f160fe758aa3c56fca4caead0fe3833bff80bccde40265e1628aea6877bc"
)
PAPER_SCALED_IMAGES_SHA256 = (
    "91d49d24e0224d6d54aee285d0564c4fef001283c07758ef38626253d990f56d"
)
DEFAULT_TARGET_REPO = "artist/mvl-sib-sent2img-mteb"
VERIFY_LANGUAGES = ("eng_Latn", "nqo_Nkoo")
CATEGORIES = (
    "entertainment",
    "geography",
    "health",
    "politics",
    "science",
    "sports",
    "travel",
)

CORPUS_FEATURES = Features(
    {"id": Value("string"), "image": Image(), "modality": Value("string")}
)
QUERY_FEATURES = Features(
    {"id": Value("string"), "text": Value("string"), "modality": Value("string")}
)
QRELS_FEATURES = Features(
    {
        "query-id": Value("string"),
        "corpus-id": Value("string"),
        "score": Value("int32"),
    }
)
TOP_RANKED_FEATURES = Features(
    {"query-id": Value("string"), "corpus-ids": Sequence(Value("string"))}
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("official_mvl_sib", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _download_source() -> tuple[Path, ModuleType, list[str]]:
    script_root = Path(
        snapshot_download(
            SOURCE_REPO,
            repo_type="dataset",
            revision=SOURCE_REVISION,
            allow_patterns=["mvl-sib.py"],
        )
    )
    script_path = script_root / "mvl-sib.py"
    if _sha256(script_path) != OFFICIAL_SCRIPT_SHA256:
        raise ValueError("Pinned official MVL-SIB generator checksum mismatch")
    official = _load_module(script_path)
    languages = list(official.LANGS)
    if len(languages) != 205 or len(set(languages)) != 205:
        raise ValueError("Expected 205 unique MVL-SIB languages")
    if tuple(official.CATEGORIES) != CATEGORIES:
        raise ValueError("Unexpected MVL-SIB categories")

    source_root = Path(
        snapshot_download(
            SOURCE_REPO,
            repo_type="dataset",
            revision=SOURCE_REVISION,
            allow_patterns=[
                "mvl-sib.py",
                "data/images/sib200/*.jpg",
                *(f"data/sib200/{lang}/*.tsv" for lang in languages),
            ],
        )
    )
    return source_root, official, languages


def _source_paths(root: Path, lang: str) -> list[str]:
    return [
        str(root / "data" / "sib200" / lang / f"{split}.tsv")
        for split in ("train", "dev", "test")
    ]


def _read_records(root: Path, official: ModuleType, lang: str) -> list[dict[str, Any]]:
    records = [
        {
            "index_id": int(row["index_id"]),
            "category": str(row["category"]),
            "text": str(row["text"]),
        }
        for row in official.read_lang_tsv(_source_paths(root, lang))
    ]
    if (
        len(records) != 1004
        or len({row["index_id"] for row in records}) != 1004
        or {row["category"] for row in records} != set(CATEGORIES)
        or any(not row["text"] for row in records)
    ):
        raise ValueError(f"Invalid MVL-SIB source rows for {lang}")
    return records


def _build_official_candidates(
    root: Path,
    official: ModuleType,
    lang: str,
    records: list[dict[str, Any]],
) -> list[tuple[list[str], int]]:
    """Read the paper's single-reference candidate sets from its pinned builder."""
    builder = official.MVLSIB(config_name=f"sent2img.{lang}")
    builder.config.num_references = 1
    image_paths = {
        category: [
            str(root / "data" / "images" / "sib200" / f"{category}_{index}.jpg")
            for index in range(10)
        ]
        for category in official.CATEGORIES
    }
    candidates: list[tuple[list[str], int]] = []
    official_rows = builder._generate_examples(_source_paths(root, lang), image_paths)
    for index, ((_, official_row), record) in enumerate(
        zip(official_rows, records * 3, strict=True)
    ):
        candidate_ids = [f"image-{Path(path).stem}" for path in official_row["images"]]
        label = int(official_row["label"])
        if (
            int(official_row["index_id"]) != record["index_id"]
            or official_row["sentences"] != [record["text"]]
            or len(candidate_ids) != 4
            or len(set(candidate_ids)) != 4
            or label not in range(4)
        ):
            raise ValueError(f"Invalid official candidate row {index} for {lang}")
        candidates.append((candidate_ids, label))
    if len(candidates) != 3012:
        raise ValueError("Expected 3,012 official candidate rows")
    return candidates


def _paper_scaled_image(path: Path) -> bytes:
    """Reproduce the paper's aspect-preserving image preprocessing."""
    with PILImage.open(path) as image:
        image.load()
        bounds = (640, 480) if image.width > image.height else (480, 640)
        image.thumbnail(bounds, PILImage.Resampling.LANCZOS, reducing_gap=2.0)
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
    return buffer.getvalue()


def _build_corpus(root: Path) -> tuple[Dataset, int, str]:
    rows: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    total_bytes = 0
    for category in CATEGORIES:
        for index in range(10):
            path = root / "data" / "images" / "sib200" / f"{category}_{index}.jpg"
            image_bytes = _paper_scaled_image(path)
            digest.update(path.stem.encode())
            digest.update(b"\0")
            digest.update(image_bytes)
            total_bytes += len(image_bytes)
            rows.append(
                {
                    "id": f"image-{category}_{index}",
                    "image": {"bytes": image_bytes, "path": path.name},
                    "modality": "image",
                }
            )
    scaled_images_sha256 = digest.hexdigest()
    if scaled_images_sha256 != PAPER_SCALED_IMAGES_SHA256:
        raise ValueError("Paper-scaled MVL-SIB image checksum mismatch")
    return (
        Dataset.from_list(rows, features=CORPUS_FEATURES),
        total_bytes,
        scaled_images_sha256,
    )


def _build_language_configs(
    records: list[dict[str, Any]], candidates: list[tuple[list[str], int]]
) -> tuple[Dataset, Dataset, Dataset]:
    query_ids = [f"query-{index}" for index in range(len(candidates))]
    queries = Dataset.from_dict(
        {
            "id": query_ids,
            "text": [row["text"] for row in records * 3],
            "modality": ["text"] * len(candidates),
        },
        features=QUERY_FEATURES,
    )
    relevant_ids = [candidate_ids[label] for candidate_ids, label in candidates]
    qrels = Dataset.from_dict(
        {
            "query-id": query_ids,
            "corpus-id": relevant_ids,
            "score": [1] * len(candidates),
        },
        features=QRELS_FEATURES,
    )
    top_ranked = Dataset.from_dict(
        {
            "query-id": query_ids,
            "corpus-ids": [candidate_ids for candidate_ids, _ in candidates],
        },
        features=TOP_RANKED_FEATURES,
    )
    return queries, qrels, top_ranked


def _dataset_card(languages: list[str]) -> str:
    lines = [
        "---",
        "license: cc-by-sa-4.0",
        "configs:",
    ]
    for lang in languages:
        paths = {
            "corpus": "data/corpus/test.parquet",
            "queries": f"data/{lang}/queries.parquet",
            "qrels": f"data/{lang}/qrels.parquet",
            "top_ranked": f"data/{lang}/top_ranked.parquet",
        }
        for suffix, path in paths.items():
            lines.extend(
                [
                    f"- config_name: {lang}-{suffix}",
                    "  data_files:",
                    "  - split: test",
                    f"    path: {path}",
                ]
            )
    lines.extend(
        [
            "---",
            "",
            "# MVL-SIB sentence-to-image for MTEB",
            "",
            "Native MTEB retrieval packaging of the official MVL-SIB single-reference",
            "(`k=1`) sentence-to-image task. Each of 205 language subsets has 3,012",
            "sentence queries, four candidate images per query, and one correct image.",
            "All subsets reference one shared 70-image corpus file.",
            "",
            "## Source and changes",
            "",
            "Derived from the official [WueNLP/MVL-SIB](https://huggingface.co/datasets/WueNLP/mvl-sib)",
            "dataset and [MVL-SIB paper](https://aclanthology.org/2025.findings-acl.838/),",
            f"pinned at `{SOURCE_REVISION}`. The official builder's single-reference",
            "sentence-to-image output is converted to MTEB's retrieval schema. Query",
            "texts, candidate sets, their order, and labels are unchanged. Images are",
            "downsampled with the paper's aspect-preserving 640x480 preprocessing.",
            "Only IDs and packaging are added: candidate images are stored as",
            "`top_ranked`, and every language-specific corpus config references",
            "the same physical file.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_artifact(
    output: Path, root: Path, official: ModuleType, languages: list[str]
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    (output / "data" / "corpus").mkdir(parents=True)

    corpus, scaled_image_bytes, scaled_images_sha256 = _build_corpus(root)
    corpus.to_parquet(output / "data" / "corpus" / "test.parquet")
    corpus_ids = set(corpus["id"])
    if len(corpus) != 70 or len(corpus_ids) != 70:
        raise ValueError("Invalid shared corpus")

    source_structure: list[tuple[int, str]] | None = None
    candidates: list[tuple[list[str], int]] | None = None
    language_digests: dict[str, str] = {}
    for lang in languages:
        records = _read_records(root, official, lang)
        structure = [(row["index_id"], row["category"]) for row in records]
        if source_structure is None:
            source_structure = structure
            candidates = _build_official_candidates(root, official, lang, records)
            for candidate_ids, label in candidates:
                if len(candidate_ids) != 4 or len(set(candidate_ids)) != 4:
                    raise ValueError("Invalid candidate manifest")
                if not set(candidate_ids) <= corpus_ids or label not in range(4):
                    raise ValueError("Unknown corpus ID in candidate manifest")
        elif structure != source_structure:
            raise ValueError(f"MVL-SIB source alignment differs for {lang}")
        if candidates is None or len(candidates) != 3012:
            raise ValueError("Invalid candidate manifest")

        queries, qrels, top_ranked = _build_language_configs(records, candidates)
        if any(len(dataset) != 3012 for dataset in (queries, qrels, top_ranked)):
            raise ValueError(f"Invalid row counts for {lang}")

        lang_dir = output / "data" / lang
        lang_dir.mkdir(parents=True)
        queries.to_parquet(lang_dir / "queries.parquet")
        qrels.to_parquet(lang_dir / "qrels.parquet")
        top_ranked.to_parquet(lang_dir / "top_ranked.parquet")
        digest = hashlib.sha256()
        for row, (candidate_ids, label) in zip(records * 3, candidates, strict=True):
            digest.update(row["text"].encode())
            digest.update(b"\0")
            digest.update("\0".join(candidate_ids).encode())
            digest.update(bytes([label]))
        language_digests[lang] = digest.hexdigest()

    if source_structure is None or candidates is None:
        raise ValueError("No MVL-SIB languages were built")
    (output / "README.md").write_text(_dataset_card(languages), encoding="utf-8")
    manifest = {
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "official_script_sha256": OFFICIAL_SCRIPT_SHA256,
        "languages": len(languages),
        "queries_per_language": 3012,
        "images": 70,
        "image_preprocessing": "paper 640x480 aspect-preserving downsampling",
        "scaled_images_sha256": scaled_images_sha256,
        "scaled_image_bytes": scaled_image_bytes,
        "candidate_manifest_sha256": hashlib.sha256(
            json.dumps(candidates, separators=(",", ":")).encode()
        ).hexdigest(),
        "language_sha256": language_digests,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _verify_local_artifact(output, languages)
    return manifest


def _verify_local_artifact(output: Path, languages: list[str]) -> None:
    if (
        pq.ParquetFile(output / "data" / "corpus" / "test.parquet").metadata.num_rows
        != 70
    ):
        raise ValueError("Invalid local corpus row count")
    for lang in languages:
        lang_dir = output / "data" / lang
        for name in ("queries", "qrels", "top_ranked"):
            rows = pq.ParquetFile(lang_dir / f"{name}.parquet").metadata.num_rows
            if rows != 3012:
                raise ValueError(f"Invalid {lang}-{name} row count: {rows}")
    expected_files = 1 + 3 * len(languages)
    parquet_files = list((output / "data").rglob("*.parquet"))
    if len(parquet_files) != expected_files:
        raise ValueError(f"Expected {expected_files} parquet files")


def _push(
    output: Path,
    repo_id: str,
    token: str,
    languages: list[str],
    *,
    private: bool,
) -> str:
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    info = api.dataset_info(repo_id)
    if bool(info.private) != private:
        raise ValueError(
            f"Refusing to change the visibility of existing dataset {repo_id}"
        )
    api.upload_folder(
        folder_path=output,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add native MVL-SIB sentence-to-image artifact",
    )
    info = api.dataset_info(repo_id)
    if bool(info.private) != private:
        raise ValueError(f"Unexpected visibility for {repo_id}")
    revision = info.sha

    expected_configs = {
        f"{lang}-{suffix}"
        for lang in languages
        for suffix in ("corpus", "queries", "qrels", "top_ranked")
    }
    configs = set(get_dataset_config_names(repo_id, revision=revision, token=token))
    if configs != expected_configs:
        raise ValueError(
            f"Remote config mismatch: missing={expected_configs - configs}, "
            f"extra={configs - expected_configs}"
        )
    for lang in VERIFY_LANGUAGES:
        expected_rows = {
            "corpus": 70,
            "queries": 3012,
            "qrels": 3012,
            "top_ranked": 3012,
        }
        for suffix, expected in expected_rows.items():
            dataset = load_dataset(
                repo_id,
                f"{lang}-{suffix}",
                split="test",
                revision=revision,
                token=token,
            )
            if len(dataset) != expected:
                raise ValueError(f"Remote {lang}-{suffix} has {len(dataset)} rows")
    return revision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-id", default=DEFAULT_TARGET_REPO)
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--token-file", type=Path)
    parser.add_argument(
        "--visibility", choices=("private", "public"), default="private"
    )
    args = parser.parse_args()

    root, official, languages = _download_source()
    manifest = _write_artifact(args.output_dir.resolve(), root, official, languages)
    print(
        f"built={args.output_dir} languages={manifest['languages']} "
        f"queries={manifest['languages'] * manifest['queries_per_language']}"
    )
    if args.push:
        if args.token_file is None:
            parser.error("--push requires --token-file")
        token = args.token_file.read_text(encoding="utf-8").strip()
        if not token:
            raise ValueError("Token file is empty")
        private = args.visibility == "private"
        revision = _push(
            args.output_dir.resolve(),
            args.repo_id,
            token,
            languages,
            private=private,
        )
        print(
            f"pushed=https://huggingface.co/datasets/{args.repo_id}@{revision} "
            f"visibility={args.visibility}"
        )


if __name__ == "__main__":
    main()
