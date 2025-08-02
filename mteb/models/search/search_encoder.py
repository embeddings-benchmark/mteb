from __future__ import annotations

from typing import Any

from mteb import Encoder, TaskMetadata
from mteb.types import (
    CorpusDatasetType,
    InstructionDatasetType,
    QueryDatasetType,
    RetrievalOutputType,
    TopRankedDocumentsType,
)


class SearchEncoder:
    """Interface for searching models."""

    def __init__(
        self,
        model: Encoder,
        **kwargs: dict[str, Any],
    ) -> None:
        self.model = model

    def index(
        self,
        corpus: CorpusDatasetType,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        encoding_kwargs: dict[str, Any],
    ) -> None:
        """Index the corpus for retrieval.

        Args:
            corpus: Corpus dataset to index.
            task_metadata: Metadata of the task, used to determine how to index the corpus.
            hf_split: Split of current task, allows to know some additional information about current split.
            hf_subset: Subset of current task. Similar to `hf_split` to get more information
            encoding_kwargs: Additional arguments to pass to the encoder during indexing.
        """
        ...

    def search(
        self,
        queries: QueryDatasetType,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        encoding_kwargs: dict[str, Any],
        instructions: InstructionDatasetType | None = None,
    ) -> RetrievalOutputType:
        """Search the corpus for the given queries.

        Args:
            queries: Queries to find
            task_metadata: Task metadata
            hf_split: split of the dataset
            hf_subset: subset of the dataset
            instructions: Optional instructions to use for the search.
            encoding_kwargs: Additional arguments to pass to the encoder during indexing.

        Returns:
            Dictionary with query IDs as keys with dict as values, where each value is a mapping of document IDs to their relevance scores.
        """
        ...

    def rerank(
        self,
        queries: QueryDatasetType,
        top_ranked: TopRankedDocumentsType,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        encoding_kwargs: dict[str, Any],
        instructions: InstructionDatasetType | None = None,
    ) -> RetrievalOutputType:
        """Rerank the top-ranked documents for the given queries.

        Args:
            queries: Queries to find
            top_ranked: Top-ranked documents for each query, mapping query IDs to a list of document IDs.
            task_metadata: Task metadata
            hf_split: split of the dataset
            hf_subset: subset of the dataset
            encoding_kwargs: Additional arguments to pass to the encoder during indexing.
            instructions: Optional instructions to use for the search.

        Returns:
            Dictionary with query IDs as keys with dict as values, where each value is a mapping of document IDs to their relevance scores.
        """
        ...
