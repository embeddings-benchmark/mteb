from collections.abc import Callable
from typing import Protocol

from mteb.types import Array, TopRankedDocumentsType


class IndexEncoderSearchProtocol(Protocol):
    """Protocol for search backends used in encoder-based retrieval."""

    def add_documents(
        self,
        embeddings: Array,
        idxs: list[str],
    ) -> None:
        """Add documents to the search backend.

        Args:
            embeddings: Embeddings of the documents to add.
            idxs: IDs of the documents to add.
        """

    def search(
        self,
        embeddings: Array,
        top_k: int,
        similarity_fn: Callable[[Array, Array], Array],
        top_ranked: TopRankedDocumentsType | None = None,
        query_idx_to_id: dict[int, str] | None = None,
    ) -> tuple[list[list[float]], list[list[int]]]:
        """Search through added corpus embeddings or rerank top-ranked documents.

        Supports both full-corpus and reranking search modes:
            - Full-corpus mode: `top_ranked=None`, uses added corpus embeddings.
            - Reranking mode:  `top_ranked` contains mapping {query_id: [doc_ids]}.

        Args:
            embeddings: Query embeddings, shape (num_queries, dim).
            top_k: Number of top results to return.
            similarity_fn: Function to compute similarity between query and corpus.
            top_ranked: Mapping of query_id -> list of candidate doc_ids. Used for reranking.
            query_idx_to_id: Mapping of query index -> query_id. Used for reranking.

        Returns:
            A tuple (top_k_values, top_k_indices), for each query:
                - top_k_values: List of top-k similarity scores.
                - top_k_indices: List of indices of the top-k documents in the added corpus.
        """

    def clear(self) -> None:
        """Clear all stored documents and embeddings from the backend."""
