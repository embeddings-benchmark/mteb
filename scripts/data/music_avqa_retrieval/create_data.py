"""Build cross-clip MUSIC-AVQA audio/video retrieval datasets.

The source test split contains music-performance clips labelled with one of 22
instrument classes. This script creates a class-level retrieval task: for each
instrument, five clips become queries and ten *different* clips become corpus
items. Thus neither direction can retrieve media from the query's source clip.

Two standard MTEB retrieval datasets are produced:

* ``MusicAVQA-A2V-Retrieval``: audio queries and video corpus items.
* ``MusicAVQA-V2A-Retrieval``: video queries and audio corpus items.

Usage:
    python scripts/data/music_avqa_retrieval/create_data.py --push
"""

from __future__ import annotations

import argparse
import io
import random
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Audio, Dataset, Video
from huggingface_hub import HfApi, snapshot_download

SOURCE_DATASET = "mteb/MUSIC-AVQA_cls-preprocessed"
SOURCE_REVISION = "29f50ae80ad4e8c1cfdbc0148aefe6fe050833dd"
SOURCE_SPLIT = "test"
SEED = 42
QUERIES_PER_CLASS = 5
CORPUS_PER_CLASS = 10
MEDIA_ROWS_PER_SHARD = 20


@dataclass(frozen=True)
class RetrievalDirection:
    """The source modality retained by each side of one retrieval direction."""

    name: str
    query_column: str
    corpus_column: str
    repo_name: str


@dataclass(frozen=True)
class SourceRecord:
    """A source row's small metadata plus its local Parquet position."""

    path: Path
    row: int
    label: int
    video_id: str


DIRECTIONS = (
    RetrievalDirection("a2v", "audio", "video", "MusicAVQA-A2V-Retrieval"),
    RetrievalDirection("v2a", "video", "audio", "MusicAVQA-V2A-Retrieval"),
)


def load_source_records() -> list[SourceRecord]:
    """Load only source labels and IDs, without decoding any media."""
    snapshot_path = Path(
        snapshot_download(
            SOURCE_DATASET,
            repo_type="dataset",
            revision=SOURCE_REVISION,
            allow_patterns=[f"data/{SOURCE_SPLIT}-*.parquet"],
        )
    )
    files = sorted((snapshot_path / "data").glob(f"{SOURCE_SPLIT}-*.parquet"))
    assert files, "no source Parquet files found"

    records: list[SourceRecord] = []
    for path in files:
        metadata = pq.read_table(path, columns=["label", "video_id"])
        for row, (label, video_id) in enumerate(
            zip(metadata["label"].to_pylist(), metadata["video_id"].to_pylist())
        ):
            records.append(SourceRecord(path, row, int(label), video_id))
    return records


def select_source_indices(
    records: list[SourceRecord], seed: int = SEED
) -> tuple[list[int], list[int]]:
    """Return deterministic, class-balanced, disjoint query and corpus indices."""
    by_label: dict[int, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_label[record.label].append(index)

    required = QUERIES_PER_CLASS + CORPUS_PER_CLASS
    too_small = {
        label: len(indices)
        for label, indices in by_label.items()
        if len(indices) < required
    }
    assert not too_small, f"classes with fewer than {required} clips: {too_small}"

    rng = random.Random(seed)
    query_indices: list[int] = []
    corpus_indices: list[int] = []
    for label in sorted(by_label):
        indices = list(by_label[label])
        rng.shuffle(indices)
        query_indices.extend(indices[:QUERIES_PER_CLASS])
        corpus_indices.extend(indices[QUERIES_PER_CLASS:required])

    return query_indices, corpus_indices


def validate_source_split(
    records: list[SourceRecord], query_indices: list[int], corpus_indices: list[int]
) -> None:
    """Prove that query and corpus media originate from different source clips."""
    assert set(query_indices).isdisjoint(corpus_indices)

    query_source_ids = [records[index].video_id for index in query_indices]
    corpus_source_ids = [records[index].video_id for index in corpus_indices]
    assert len(query_source_ids) == len(set(query_source_ids)), (
        "duplicate query video_id"
    )
    assert len(corpus_source_ids) == len(set(corpus_source_ids)), (
        "duplicate corpus video_id"
    )
    assert set(query_source_ids).isdisjoint(corpus_source_ids), "source video overlap"


def _make_media_split(
    records: list[SourceRecord], indices: list[int], modality: str, prefix: str
) -> Dataset:
    """Read only selected media from local Parquet row groups."""
    selected_by_path: dict[Path, list[tuple[int, int]]] = defaultdict(list)
    for output_index, record_index in enumerate(indices):
        record = records[record_index]
        selected_by_path[record.path].append((record.row, output_index))

    media_by_output_index: dict[int, dict] = {}
    for path, selected_rows in selected_by_path.items():
        parquet_file = pq.ParquetFile(path)
        row_start = 0
        for row_group in range(parquet_file.num_row_groups):
            row_count = parquet_file.metadata.row_group(row_group).num_rows
            row_end = row_start + row_count
            rows_in_group = [
                (row, output_index)
                for row, output_index in selected_rows
                if row_start <= row < row_end
            ]
            if rows_in_group:
                column = parquet_file.read_row_group(row_group, columns=[modality])[
                    modality
                ]
                for row, output_index in rows_in_group:
                    media_by_output_index[output_index] = column[
                        row - row_start
                    ].as_py()
            row_start = row_end

    assert len(media_by_output_index) == len(indices)
    rows = [
        {
            "id": f"{prefix}-{output_index:04d}",
            modality: media_by_output_index[output_index],
        }
        for output_index in range(len(indices))
    ]
    feature = Audio() if modality == "audio" else Video()
    return Dataset.from_list(rows).cast_column(modality, feature)


def build_direction(
    records: list[SourceRecord],
    query_indices: list[int],
    corpus_indices: list[int],
    direction: RetrievalDirection,
) -> tuple[Dataset, Dataset, Dataset]:
    """Build one modality direction in the standard queries/corpus/qrels layout."""
    query_labels = [records[index].label for index in query_indices]
    corpus_labels = [records[index].label for index in corpus_indices]

    queries = _make_media_split(
        records, query_indices, direction.query_column, prefix="q"
    )
    corpus = _make_media_split(
        records, corpus_indices, direction.corpus_column, prefix="c"
    )

    corpus_ids_by_label: dict[int, list[str]] = defaultdict(list)
    for corpus_id, label in zip(corpus["id"], corpus_labels):
        corpus_ids_by_label[label].append(corpus_id)

    qrels = Dataset.from_list(
        [
            {"query-id": query_id, "corpus-id": corpus_id, "score": 1}
            for query_id, label in zip(queries["id"], query_labels)
            for corpus_id in corpus_ids_by_label[label]
        ]
    )

    assert set(queries["id"]).isdisjoint(corpus["id"])
    assert all(
        len(corpus_ids_by_label[label]) == CORPUS_PER_CLASS for label in query_labels
    )
    assert len(qrels) == len(queries) * CORPUS_PER_CLASS
    return queries, corpus, qrels


def dataset_card(direction: RetrievalDirection) -> str:
    """Return a concise provenance and construction card for a published dataset."""
    modality_description = (
        "audio queries and video corpus items"
        if direction.name == "a2v"
        else "video queries and audio corpus items"
    )
    return f"""---
license: cc-by-nc-4.0
configs:
- config_name: queries
  data_files:
  - split: test
    path: queries/**
- config_name: corpus
  data_files:
  - split: test
    path: corpus/**
- config_name: qrels
  data_files:
  - split: test
    path: qrels/**
---

# {direction.repo_name}

This is a derived retrieval benchmark from the `test` split of
[`{SOURCE_DATASET}`](https://huggingface.co/datasets/{SOURCE_DATASET}) at
revision `{SOURCE_REVISION}`. It uses {modality_description}.

## Construction

The source clips are labelled with 22 musical-instrument classes. For every
class, a deterministic seed (42) selects five clips as queries and ten distinct
clips as corpus items. Relevance is class membership, so each query has ten
relevant corpus items. The resulting test split has 110 queries, 220 corpus
items, and 1,100 qrels.

Query and corpus source `video_id` values are unique and disjoint. This is
therefore cross-clip instrument retrieval, rather than matching a modality to
the synchronous counterpart of its own clip. It evaluates instrument-level
audio-video association and may retain task/scene cues from the source data.

## Provenance and license

The source is MUSIC-AVQA from [Music Audio-Visual Question Answering
(Li et al., 2022)](https://arxiv.org/abs/2203.14072). The MUSIC-AVQA project
states that its datasets and benchmarks are available under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). This derived
dataset is published under the same non-commercial license.

The exact construction is in
[`scripts/data/music_avqa_retrieval/create_data.py`](https://github.com/embeddings-benchmark/mteb/blob/main/scripts/data/music_avqa_retrieval/create_data.py).
"""


def push_direction(
    api: HfApi,
    repo_id: str,
    direction: RetrievalDirection,
    queries: Dataset,
    corpus: Dataset,
    qrels: Dataset,
) -> None:
    """Publish a direction and its card in MTEB's standard Hub configuration layout."""
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    # ``Dataset.push_to_hub`` can hang on Windows while uploading large Video
    # feature shards. Writing small standard Parquet shards and using
    # ``upload_file`` produces the identical Hub layout without that path.
    with tempfile.TemporaryDirectory(prefix="music_avqa_") as temporary_directory:
        temporary_path = Path(temporary_directory)
        for configuration, split in (
            ("queries", queries),
            ("corpus", corpus),
            ("qrels", qrels),
        ):
            existing_paths = [
                path
                for path in api.list_repo_files(repo_id, repo_type="dataset")
                if path.startswith(f"{configuration}/")
            ]
            for path in existing_paths:
                api.delete_file(
                    path_in_repo=path,
                    repo_id=repo_id,
                    repo_type="dataset",
                    commit_message=f"Replace {configuration} test split",
                )

            num_shards = (
                1
                if configuration == "qrels"
                else -(-len(split) // MEDIA_ROWS_PER_SHARD)
            )
            for shard_index in range(num_shards):
                parquet_path = temporary_path / f"{configuration}_{shard_index}.parquet"
                split.shard(num_shards=num_shards, index=shard_index).to_parquet(
                    parquet_path
                )
                api.upload_file(
                    path_or_fileobj=str(parquet_path),
                    path_in_repo=(
                        f"{configuration}/test-{shard_index:05d}-of-{num_shards:05d}.parquet"
                    ),
                    repo_id=repo_id,
                    repo_type="dataset",
                    commit_message=f"Add {configuration} test shard {shard_index + 1}",
                )
    api.upload_file(
        path_or_fileobj=io.BytesIO(dataset_card(direction).encode()),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add dataset card",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner", default="iamfortytwo", help="HF namespace")
    parser.add_argument("--push", action="store_true", help="publish derived datasets")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate real source labels and IDs without reading selected media",
    )
    args = parser.parse_args()
    assert not (args.push and args.validate_only), "choose --push or --validate-only"

    records = load_source_records()
    query_indices, corpus_indices = select_source_indices(records)
    validate_source_split(records, query_indices, corpus_indices)
    print(
        "validated disjoint source clips: "
        f"{len(query_indices)} queries, {len(corpus_indices)} corpus items"
    )
    if args.validate_only:
        return

    api = HfApi() if args.push else None
    for direction in DIRECTIONS:
        queries, corpus, qrels = build_direction(
            records, query_indices, corpus_indices, direction
        )
        print(
            f"{direction.name}: {len(queries)} queries, {len(corpus)} corpus, "
            f"{len(qrels)} qrels"
        )
        if api is not None:
            repo_id = f"{args.owner}/{direction.repo_name}"
            push_direction(api, repo_id, direction, queries, corpus, qrels)
            print(f"published https://huggingface.co/datasets/{repo_id}")
        # Raw media payloads are large; release a direction before building the next.
        del queries, corpus, qrels


if __name__ == "__main__":
    main()
