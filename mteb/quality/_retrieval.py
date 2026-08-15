"""Filtering a retrieval task, which has to keep the relevance judgements valid as it removes documents."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mteb.abstasks.retrieval import _filter_queries_without_positives

from ._filters import _iter_row_keys, _row_key, _warn

if TYPE_CHECKING:
    from collections.abc import Mapping

    from datasets import Dataset

    from mteb.abstasks.retrieval import AbsTaskRetrieval
    from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
    from mteb.types import Modalities

    from ._filters import KeepIndicesFn, TextNormalization

logger = logging.getLogger(__name__)


def _warn_about_empty_retrieval_splits(task: AbsTaskRetrieval) -> None:
    """Report splits that a filter left without documents or without queries."""
    for subset, splits_data in task.dataset.items():
        for split, split_data in splits_data.items():
            empty = sorted(
                name
                for name in ("corpus", "queries")
                if len(split_data[name]) == 0  # type: ignore[literal-required]
            )
            if empty:
                _warn(
                    f"Filtering left the {' and the '.join(empty)} of the '{split}' split of "
                    f"'{task.metadata.name}' (subset '{subset}') empty. Evaluating it will fail."
                )


def _select_kept_entries(
    dataset: Dataset,
    keep_fn: KeepIndicesFn,
    col_modalities: Mapping[str, Modalities],
    *,
    normalize: TextNormalization,
    remap_duplicates: bool,
    num_proc: int | None,
) -> tuple[Dataset, set[str], dict[str, str]]:
    """Apply `keep_fn` to a corpus or query dataset.

    Remapping assumes that `keep_fn` keeps the *first* entry of each group of equal rows, which lets the
    replacements be collected in a single pass: a removed entry always follows the entry it is remapped onto.

    Returns:
        The filtered dataset, the ids it kept, and a mapping from the id of a removed entry to the id of the first
        kept entry with the same content. That mapping is empty unless `remap_duplicates` is set.
    """
    keys = _iter_row_keys(
        dataset, col_modalities, normalize=normalize, num_proc=num_proc
    )
    keep = keep_fn(keys)
    ids = dataset["id"]
    kept_ids = {ids[i] for i in keep}

    replacements: dict[str, str] = {}
    if remap_duplicates:
        keep_set = set(keep)
        canonical: dict[bytes, str] = {}
        rows = _iter_row_keys(
            dataset, col_modalities, normalize=normalize, num_proc=num_proc
        )
        for i, row in enumerate(rows):
            key = _row_key(row)
            if i in keep_set:
                canonical.setdefault(key, ids[i])
            elif (target := canonical.get(key)) is not None:
                replacements[ids[i]] = target

    return dataset.select(keep), kept_ids, replacements


def _filter_retrieval_split(  # noqa: PLR0914
    split_data: RetrievalSplitData,
    keep_fn: KeepIndicesFn,
    col_modalities: Mapping[str, Modalities],
    *,
    normalize: TextNormalization = "strip",
    remap_duplicates: bool = False,
    num_proc: int | None = None,
) -> tuple[RetrievalSplitData, int]:
    """Apply `keep_fn` to the corpus and the queries of a single split, keeping the relevance judgements valid.

    Args:
        split_data: The corpus, queries, relevance judgements and top-ranked documents of one split.
        keep_fn: Decides which documents and queries to keep.
        col_modalities: The columns of the corpus and the queries to compare, mapped to their modality.
        normalize: How to normalize text before comparing it.
        remap_duplicates: Whether a removed document or query should hand its relevance judgements over to the
            first kept entry with the same content. This is what makes deduplication lossless; for a filter that
            removes entries on their own merit, such as a length filter, it must be False.
        num_proc: Number of processes to use for hashing non-text content.

    Returns:
        The filtered split and the number of documents and queries that were removed.

    Raises:
        ValueError: If a compared column is missing from the corpus or from the queries.
    """
    old_corpus, old_queries = split_data["corpus"], split_data["queries"]
    missing = [
        column
        for column in col_modalities
        if column not in old_corpus.column_names
        or column not in old_queries.column_names
    ]
    if missing:
        raise ValueError(
            f"Cannot filter on {missing}: the corpus has the columns {old_corpus.column_names} and the queries "
            f"have the columns {old_queries.column_names}."
        )

    corpus, kept_doc_ids, doc_replacements = _select_kept_entries(
        old_corpus,
        keep_fn,
        col_modalities,
        normalize=normalize,
        remap_duplicates=remap_duplicates,
        num_proc=num_proc,
    )
    queries, kept_query_ids, query_replacements = _select_kept_entries(
        old_queries,
        keep_fn,
        col_modalities,
        normalize=normalize,
        remap_duplicates=remap_duplicates,
        num_proc=num_proc,
    )

    relevant_docs: dict[str, dict[str, int]] = {}
    for query_id, docs in split_data["relevant_docs"].items():
        query_id = query_replacements.get(query_id, query_id)  # noqa: PLW2901
        if query_id not in kept_query_ids:
            continue
        scores = relevant_docs.setdefault(query_id, {})
        for doc_id, score in docs.items():
            doc_id = doc_replacements.get(doc_id, doc_id)  # noqa: PLW2901
            if doc_id in kept_doc_ids:
                scores[doc_id] = max(scores.get(doc_id, score), score)

    relevant_docs, queries = _filter_queries_without_positives(  # type: ignore[assignment]
        relevant_docs, queries
    )

    top_ranked = split_data["top_ranked"]
    if top_ranked is not None:
        remaining_query_ids = set(queries["id"])
        new_top_ranked: dict[str, list[str]] = {}
        for query_id, doc_ids in top_ranked.items():
            query_id = query_replacements.get(query_id, query_id)  # noqa: PLW2901
            if query_id not in remaining_query_ids:
                continue
            ranked = new_top_ranked.setdefault(query_id, [])
            for doc_id in doc_ids:
                doc_id = doc_replacements.get(doc_id, doc_id)  # noqa: PLW2901
                if doc_id in kept_doc_ids and doc_id not in ranked:
                    ranked.append(doc_id)
        top_ranked = new_top_ranked

    n_removed = len(old_corpus) + len(old_queries) - len(corpus) - len(queries)
    return {
        "corpus": corpus,
        "queries": queries,
        "relevant_docs": relevant_docs,
        "top_ranked": top_ranked,
    }, n_removed
