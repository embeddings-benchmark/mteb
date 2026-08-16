"""Filtering a retrieval task, which has to keep the relevance judgements valid as it removes documents."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mteb.abstasks.retrieval import _filter_queries_without_positives

from ._filters import _iter_row_content, _row_key, _warn

if TYPE_CHECKING:
    from collections.abc import Mapping

    from datasets import Dataset

    from mteb.abstasks.retrieval import AbsTaskRetrieval
    from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
    from mteb.types import Modalities

    from ._filters import KeepIndicesFn, TextNormalization

logger = logging.getLogger(__name__)


def _check_empty_retrieval_splits(task: AbsTaskRetrieval) -> None:
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


def _columns_present_in(
    dataset: Dataset, col_modalities: Mapping[str, Modalities], side: str
) -> dict[str, Modalities]:
    """The compared columns that `dataset` actually has.

    The corpus and the queries do not hold the same columns -- only a corpus entry has a `title` -- so each side is
    compared on the columns it has rather than on their intersection.

    Raises:
        ValueError: If the dataset has none of the compared columns.
    """
    present = {
        column: modality
        for column, modality in col_modalities.items()
        if column in dataset.column_names
    }
    if not present:
        raise ValueError(
            f"Cannot filter the {side} on {sorted(col_modalities)}: it only has the columns "
            f"{dataset.column_names}."
        )
    return present


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
    rows = _iter_row_content(
        dataset, col_modalities, normalize=normalize, num_proc=num_proc
    )
    keep = keep_fn(rows)
    ids = dataset["id"]
    kept_ids = {ids[i] for i in keep}

    replacements: dict[str, str] = {}
    if remap_duplicates:
        keep_set = set(keep)
        canonical: dict[bytes, str] = {}
        all_rows = _iter_row_content(
            dataset, col_modalities, normalize=normalize, num_proc=num_proc
        )
        for i, row in enumerate(all_rows):
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
        ValueError: If the corpus or the queries have none of the compared columns.
    """
    old_corpus, old_queries = split_data["corpus"], split_data["queries"]
    corpus_columns = _columns_present_in(old_corpus, col_modalities, "corpus")
    query_columns = _columns_present_in(old_queries, col_modalities, "queries")

    corpus, kept_doc_ids, doc_replacements = _select_kept_entries(
        old_corpus,
        keep_fn,
        corpus_columns,
        normalize=normalize,
        remap_duplicates=remap_duplicates,
        num_proc=num_proc,
    )
    queries, kept_query_ids, query_replacements = _select_kept_entries(
        old_queries,
        keep_fn,
        query_columns,
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
