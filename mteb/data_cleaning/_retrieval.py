"""Filtering a retrieval task, which has to keep the relevance judgements valid as it removes documents."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from mteb._create_dataloaders import _retrieval_texts
from mteb.abstasks.retrieval import _filter_queries_without_positives
from mteb.types import PromptType

from ._filters import _content_readers, _iter_row_content, _normalize, _row_key

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from datasets import Dataset

    from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Modalities

    from ._filters import Normalization, _Filter

logger = logging.getLogger(__name__)


def _side_columns(
    dataset: Dataset,
    declared_columns: Mapping[str, Modalities],
    modalities: Sequence[Modalities],
    filter_: _Filter,
) -> dict[str, Modalities]:
    """The declared content columns that the model reads from one side of a retrieval split.

    A task declares the union of its content columns, but the corpus and the queries rarely hold the same ones: an
    any-to-any task puts a different modality on each side, e.g. an image corpus searched by text queries. Each side
    is compared on the modalities the task's category gives it, as it is when evaluated, which also passes over a
    column that a side merely carries, such as an empty `text` column next to the images of a corpus. A side left
    without any compared column is not filtered.

    Neither is a side holding a modality that `filter_` does not apply to: each of its entries is a single input
    combining its modalities, so e.g. a document page whose text is empty still has its image.
    """
    if filter_.modalities is not None and not filter_.modalities.issuperset(modalities):
        return {}
    return {
        column: modality
        for column, modality in declared_columns.items()
        if modality in modalities and column in dataset.column_names
    }


def _side_readers(
    dataset: Dataset,
    col_modalities: Mapping[str, Modalities],
    prompt_type: PromptType,
    *,
    normalization: Normalization,
    hash_non_text: bool,
    num_proc: int | None,
) -> list[Callable[[], Iterable[Any]]]:
    """Like `_content_readers`, but reading the text of each entry the way the model reads it.

    A document is its title and text joined, a query carries its instruction, and a conversation is flattened into a
    single string, so that e.g. two conversations are not compared as two missing texts.
    """
    readers = _content_readers(
        dataset,
        {
            column: modality
            for column, modality in col_modalities.items()
            if column != "text"
        },
        normalization=normalization,
        hash_non_text=hash_non_text,
        num_proc=num_proc,
    )
    if "text" in col_modalities:
        texts = _retrieval_texts(dataset, prompt_type)
        readers.append(lambda: (_normalize(text, normalization) for text in texts))
    return readers


def _select_kept_entries(
    dataset: Dataset,
    filter_: _Filter,
    col_modalities: Mapping[str, Modalities],
    prompt_type: PromptType,
    *,
    normalization: Normalization,
    num_proc: int | None,
) -> tuple[Dataset, set[str], dict[str, str]]:
    """Apply `filter_` to a corpus or query dataset.

    Remapping assumes that a filter removing duplicates keeps the *first* entry of each group of equal rows, which
    lets the replacements be collected in a single pass: a removed entry always follows the entry it is remapped
    onto.

    Returns:
        The filtered dataset, the ids it kept, and a mapping from the id of a removed entry to the id of the first
        kept entry with the same content. That mapping is empty unless `filter_` removes duplicates.
    """
    ids = dataset["id"]
    if not col_modalities:
        return dataset, set(ids), {}

    # built once: reading the rows twice below must not hash the same images or audio twice
    readers = _side_readers(
        dataset,
        col_modalities,
        prompt_type,
        normalization=normalization,
        hash_non_text=filter_.compares_rows,
        num_proc=num_proc,
    )
    keep = filter_.keep_fn(_iter_row_content(readers))
    kept_ids = {ids[i] for i in keep}

    replacements: dict[str, str] = {}
    if filter_.removes_duplicates:
        keep_set = set(keep)
        canonical: dict[bytes, str] = {}
        for i, row in enumerate(_iter_row_content(readers)):
            key = _row_key(row)
            if i in keep_set:
                canonical.setdefault(key, ids[i])
            elif (target := canonical.get(key)) is not None:
                replacements[ids[i]] = target

    return dataset.select(keep), kept_ids, replacements


def _filter_retrieval_split(  # noqa: PLR0914
    split_data: RetrievalSplitData,
    filter_: _Filter,
    declared_columns: Mapping[str, Modalities],
    metadata: TaskMetadata,
    *,
    normalization: Normalization,
    num_proc: int | None = None,
) -> tuple[RetrievalSplitData, int]:
    """Apply `filter_` to the corpus and the queries of a single split, keeping the relevance judgements valid.

    A removed document or query that duplicates a kept one hands its relevance judgements over to it when
    `filter_.removes_duplicates` is set, which is what makes deduplication lossless.

    Args:
        split_data: The corpus, queries, relevance judgements and top-ranked documents of one split.
        filter_: Decides which documents and queries to keep.
        declared_columns: Every content column the task declares, mapped to its modality. Each of the corpus
            and the queries is compared on those of its own modalities, so the two sides need not match.
        metadata: The task's metadata, whose category gives the modalities of the corpus and of the queries.
        normalization: How to rewrite text before comparing it.
        num_proc: Number of processes to use for hashing non-text content.

    Returns:
        The filtered split and the number of documents and queries that were removed.
    """
    old_corpus, old_queries = split_data["corpus"], split_data["queries"]
    corpus_columns = _side_columns(
        old_corpus,
        declared_columns,
        metadata.get_modalities(prompt_type=PromptType.document),
        filter_,
    )
    query_columns = _side_columns(
        old_queries,
        declared_columns,
        metadata.get_modalities(prompt_type=PromptType.query),
        filter_,
    )

    corpus, kept_doc_ids, doc_replacements = _select_kept_entries(
        old_corpus,
        filter_,
        corpus_columns,
        PromptType.document,
        normalization=normalization,
        num_proc=num_proc,
    )
    queries, kept_query_ids, query_replacements = _select_kept_entries(
        old_queries,
        filter_,
        query_columns,
        PromptType.query,
        normalization=normalization,
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
