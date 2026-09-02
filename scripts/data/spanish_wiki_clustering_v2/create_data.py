#!/usr/bin/env python3
"""Build SpanishWikiClustering v2 from the official Spanish Wikipedia dump.

Source: ``eswiki-20260801-pages-articles-multistream.xml.bz2`` from
https://dumps.wikimedia.org/eswiki/20260801/ . The text-free selection manifest
committed beside this script records the 1,200 post-audit article/revision IDs,
labels, category evidence, and expected normalized-text hashes. It is the
versioned record of the human selection decision; this script recreates all
evaluation text from the primary Wikimedia source.

Usage:
  curl -O https://dumps.wikimedia.org/eswiki/20260801/\
eswiki-20260801-pages-articles-multistream.xml.bz2
  uv run python scripts/data/spanish_wiki_clustering_v2/create_data.py \\
      --dump eswiki-20260801-pages-articles-multistream.xml.bz2 \\
      --output-dir /tmp/spanish_wiki_clustering_v2
"""

from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import quote

from datasets import Dataset
from huggingface_hub import HfApi


SOURCE_SNAPSHOT = "eswiki-20260801"
SOURCE_DUMP_URL = (
    "https://dumps.wikimedia.org/eswiki/20260801/"
    "eswiki-20260801-pages-articles-multistream.xml.bz2"
)
NAMESPACE = "{http://www.mediawiki.org/xml/export-0.11/}"
MANIFEST_PATH = Path(__file__).with_name("selection_manifest.jsonl")
LABELS = (
    "Artes y medios",
    "Deporte",
    "Historia y arqueología",
    "Matemáticas y computación",
    "Medicina clínica",
    "Política, derecho y sociedad",
)
EXPECTED_RECORDS = 1_200
EXPECTED_PER_LABEL = 200
COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
REF_RE = re.compile(r"<ref\b[^>/]*(?:/>|>.*?</ref\s*>)", re.IGNORECASE | re.DOTALL)
TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}")
FILE_LINK_RE = re.compile(r"\[\[(?:Archivo|File):[^\]]+\]\]", re.IGNORECASE)
CATEGORY_RE = re.compile(r"\[\[Categoría:([^\]|]+)(?:\|[^\]]*)?\]\]", re.IGNORECASE)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_intro(raw_wikitext: str) -> str:
    """Apply the frozen first-section wikitext normalization rule."""
    text = raw_wikitext.split("\n=", 1)[0]
    text = COMMENT_RE.sub("", text)
    text = REF_RE.sub("", text)
    text = FILE_LINK_RE.sub("", text)
    previous = None
    while text != previous:
        previous = text
        text = TEMPLATE_RE.sub("", text)
    text = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"'{2,}", "", text)
    return re.sub(r"\s+", " ", text).strip()


def direct_categories(raw_wikitext: str) -> list[str]:
    return sorted(
        {match.strip().replace("_", " ") for match in CATEGORY_RE.findall(raw_wikitext)}
    )


def read_selection(path: Path) -> dict[int, dict[str, Any]]:
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    if len(records) != EXPECTED_RECORDS:
        raise ValueError(
            f"Expected {EXPECTED_RECORDS} selection rows, found {len(records)}"
        )
    selected = {record["article_id"]: record for record in records}
    if len(selected) != EXPECTED_RECORDS:
        raise ValueError("Selection manifest contains duplicate article IDs")
    expected = Counter({index: EXPECTED_PER_LABEL for index in range(len(LABELS))})
    if Counter(record["label"] for record in records) != expected:
        raise ValueError("Selection manifest is not balanced 6 × 200")
    for record in records:
        if record["label_name"] != LABELS[record["label"]]:
            raise ValueError(f"Invalid label mapping for {record['article_id']}")
    return selected


def selected_pages(
    dump_path: Path, selected: dict[int, dict[str, Any]]
) -> Iterator[dict[str, Any]]:
    """Stream only selected namespace-0 pages from the full source dump."""
    found: set[int] = set()
    with bz2.open(dump_path, "rb") as stream:
        for _, element in ET.iterparse(stream, events=("end",)):
            if element.tag != f"{NAMESPACE}page":
                continue
            page_id_text = element.findtext(f"{NAMESPACE}id")
            if page_id_text is None or int(page_id_text) not in selected:
                element.clear()
                continue
            page_id = int(page_id_text)
            record = selected[page_id]
            title = element.findtext(f"{NAMESPACE}title")
            namespace = element.findtext(f"{NAMESPACE}ns")
            revision = element.find(f"{NAMESPACE}revision")
            revision_id = (
                revision.findtext(f"{NAMESPACE}id") if revision is not None else None
            )
            raw_text = (
                revision.findtext(f"{NAMESPACE}text") if revision is not None else None
            )
            element.clear()
            if namespace != "0" or title != record["title"]:
                raise ValueError(f"Page identity mismatch for article {page_id}")
            if revision_id is None or int(revision_id) != record["revision_id"]:
                raise ValueError(f"Revision mismatch for article {page_id}")
            if raw_text is None:
                raise ValueError(f"Missing wikitext for article {page_id}")
            found.add(page_id)
            yield {"record": record, "raw_wikitext": raw_text}
    missing = sorted(set(selected).difference(found))
    if missing:
        raise ValueError(
            f"Source dump is missing {len(missing)} selected articles: {missing[:5]}"
        )


def build_rows(dump_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected = read_selection(MANIFEST_PATH)
    evaluation: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for item in selected_pages(dump_path, selected):
        record = item["record"]
        raw_wikitext = item["raw_wikitext"]
        text = first_intro(raw_wikitext)
        if sha256_text(text) != record["text_sha256"]:
            raise ValueError(f"Text hash mismatch for article {record['article_id']}")
        expected_categories = record["category_evidence"].get("xml_direct_categories")
        if (
            expected_categories is not None
            and direct_categories(raw_wikitext) != expected_categories
        ):
            raise ValueError(f"Category mismatch for article {record['article_id']}")
        evaluation.append({"sentences": text, "labels": record["label"]})
        provenance.append(
            {
                "article_id": record["article_id"],
                "title": record["title"],
                "source_revision": record["source_revision"],
                "permalink": (
                    "https://es.wikipedia.org/w/index.php?title="
                    f"{quote(record['title'].replace(' ', '_'))}&oldid={record['revision_id']}"
                ),
                "category_evidence": json.dumps(
                    record["category_evidence"], ensure_ascii=False, sort_keys=True
                ),
                "extraction_rule": record["extraction_rule"],
            }
        )
    if (
        len(evaluation) != EXPECTED_RECORDS
        or len({row["sentences"] for row in evaluation}) != EXPECTED_RECORDS
    ):
        raise ValueError("Output does not contain 1,200 unique texts")
    return evaluation, provenance


def write_release(
    evaluation: list[dict[str, Any]], provenance: list[dict[str, Any]], output_dir: Path
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    (output_dir / "data").mkdir(parents=True)
    (output_dir / "provenance").mkdir()
    Dataset.from_list(evaluation).to_parquet(output_dir / "data" / "test.parquet")
    Dataset.from_list(provenance).to_parquet(
        output_dir / "provenance" / "provenance.parquet"
    )
    mapping = {str(index): label for index, label in enumerate(LABELS)}
    (output_dir / "label_mapping.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "source_snapshot": SOURCE_SNAPSHOT,
        "source_dump": SOURCE_DUMP_URL,
        "selection_manifest": MANIFEST_PATH.name,
        "selection_manifest_sha256": sha256_file(MANIFEST_PATH),
        "records": len(evaluation),
        "schema": {"sentences": "string", "labels": "int64"},
        "label_mapping": mapping,
    }
    (output_dir / "transformation_manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dump", required=True, type=Path, help="Official eswiki XML multistream dump"
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-id", help="Optional Hugging Face dataset repository")
    parser.add_argument(
        "--push", action="store_true", help="Upload the generated release"
    )
    args = parser.parse_args()
    if not args.dump.is_file():
        parser.error(f"Dump not found: {args.dump}")
    if args.push and not args.repo_id:
        parser.error("--push requires --repo-id")

    evaluation, provenance = build_rows(args.dump)
    write_release(evaluation, provenance, args.output_dir)
    print(json.dumps({"records": len(evaluation), "output": str(args.output_dir)}))
    if args.push:
        api = HfApi()
        api.create_repo(args.repo_id, repo_type="dataset", exist_ok=True)
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="dataset",
            folder_path=args.output_dir,
            commit_message="Build SpanishWikiClustering v2 from Wikimedia dump",
        )


if __name__ == "__main__":
    main()
