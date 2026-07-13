"""Reupload the MSR-VTT retrieval dataset in the standard MTEB format.

The source repo ``mteb/MSR-VTT`` ships a single flat ``test`` split with
``video`` / ``audio`` / ``caption`` columns, which forced every MSR-VTT
retrieval task to implement a custom ``load_data`` that materialised queries,
corpus and qrels on the fly (see issue #4375).

This script rebuilds the dataset so that a single repo hosts one set of
``<direction>-queries`` / ``<direction>-corpus`` / ``<direction>-qrels`` configs
per retrieval direction. Each MSR-VTT task then loads through the standard
``RetrievalDatasetLoader`` by pointing ``dataset["name"]`` at its direction, and
the custom loaders can be removed.

Every direction is an identity mapping: row ``i`` on the query side is relevant
to row ``i`` on the corpus side (879 examples, msrvtt_ret_test1k split).

Usage:
    python scripts/upload_msr_vtt_retrieval.py --owner mteb --token "$HF_TOKEN"
    python scripts/upload_msr_vtt_retrieval.py --dry-run

After uploading, update ``_DATASET_REVISION`` in
``mteb/tasks/retrieval/eng/msr_vtt.py`` to the new commit hash.
"""

from __future__ import annotations

import argparse
import os

from datasets import Dataset, load_dataset

_SOURCE_PATH = "mteb/MSR-VTT"
_SOURCE_REVISION = "4661603cee25c1fd370e5478a2953203cf37155b"
_SPLIT = "test"

# direction name -> (query columns, corpus columns)
# The source columns are ``video``, ``audio`` and ``caption``; ``caption`` is
# renamed to ``text`` on whichever side it appears on so the standard loader
# treats it as the text modality.
DIRECTIONS: dict[str, tuple[list[str], list[str]]] = {
    "MSRVTTV2T": (["video"], ["caption"]),
    "MSRVTTT2V": (["caption"], ["video"]),
    "MSRVTTVA2T": (["video", "audio"], ["caption"]),
    "MSRVTTT2VA": (["caption"], ["video", "audio"]),
    "MSRVTTV2A": (["video"], ["audio"]),
    "MSRVTTA2V": (["audio"], ["video"]),
    "MSRVTTVT2A": (["video", "caption"], ["audio"]),
    "MSRVTTAT2V": (["audio", "caption"], ["video"]),
}


def _select(dataset: Dataset, columns: list[str]) -> Dataset:
    """Select ``id`` + the requested modality columns, renaming caption->text."""
    out = dataset.select_columns(["id"] + columns)
    if "caption" in columns:
        out = out.rename_column("caption", "text")
    return out


def build_configs(dataset: Dataset) -> dict[str, Dataset]:
    """Build every ``<direction>-{queries,corpus,qrels}`` config."""
    n = len(dataset)
    ids = [str(i) for i in range(n)]
    qrels = Dataset.from_dict({"query-id": ids, "corpus-id": ids, "score": [1] * n})

    configs: dict[str, Dataset] = {}
    for direction, (query_cols, corpus_cols) in DIRECTIONS.items():
        configs[f"{direction}-queries"] = _select(dataset, query_cols)
        configs[f"{direction}-corpus"] = _select(dataset, corpus_cols)
        configs[f"{direction}-qrels"] = qrels
    return configs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--owner",
        default="mteb",
        help="HF user/org that will own the reuploaded repo.",
    )
    p.add_argument(
        "--repo",
        default="MSR-VTT",
        help="Repo name (combined with --owner) to push to.",
    )
    p.add_argument(
        "--token", default=None, help="HF access token (or use the HF_TOKEN env var)."
    )
    p.add_argument("--dry-run", action="store_true", help="Build but do not upload.")
    args = p.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    repo_id = f"{args.owner}/{args.repo}"

    dataset = load_dataset(_SOURCE_PATH, revision=_SOURCE_REVISION, split=_SPLIT)
    dataset = dataset.add_column("id", [str(i) for i in range(len(dataset))])
    print(
        f"Loaded {len(dataset)} rows from {_SOURCE_PATH}, columns={dataset.column_names}"
    )

    configs = build_configs(dataset)
    for config_name, ds in configs.items():
        if args.dry_run:
            print(
                f"  [dry-run] {config_name}: {len(ds)} rows, columns={ds.column_names}"
            )
            continue
        ds.push_to_hub(repo_id, config_name=config_name, split=_SPLIT, token=token)
        print(f"  -> {repo_id} [{config_name}] ({len(ds)} rows)")


if __name__ == "__main__":
    main()
