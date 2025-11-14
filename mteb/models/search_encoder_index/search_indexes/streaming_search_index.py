import logging
from collections.abc import Callable

import torch

from mteb.types import Array, TopRankedDocumentsType

logger = logging.getLogger(__name__)


class StreamingSearchIndex:
    """Streaming backend for encoder-based search.

    - Does not store the entire corpus in memory.
    - Encodes and searches corpus in chunks.
    """

    sub_corpus_embeddings: Array | None = None
    idxs: list[str]

    def add_document(
        self,
        embeddings: Array,
        idxs: list[str],
    ) -> None:
        """Add all document embeddings and their IDs to the backend."""
        self.sub_corpus_embeddings = embeddings
        self.idxs = idxs

    def search(
        self,
        embeddings: Array,
        top_k: int,
        similarity_fn: Callable[[Array, Array], Array],
        top_ranked: TopRankedDocumentsType | None = None,
        query_idx_to_id: dict[int, str] | None = None,
    ) -> tuple[list[list[float]], list[list[int]]]:
        """Search through added corpus embeddings or rerank top-ranked documents."""
        if self.sub_corpus_embeddings is None:
            raise ValueError("No corpus embeddings found. Did you call add_document()?")

        if top_ranked is not None:
            if query_idx_to_id is None:
                raise ValueError("query_idx_to_id is required when using top_ranked.")

            scores_all: list[list[float]] = []
            idxs_all: list[list[int]] = []

            doc_id_to_idx = {doc_id: i for i, doc_id in enumerate(self.idxs)}

            for query_idx, query_emb in enumerate(embeddings):
                query_id = query_idx_to_id[query_idx]
                ranked_ids = top_ranked.get(query_id)
                if not ranked_ids:
                    logger.warning(f"No top-ranked docs for query {query_id}")
                    scores_all.append([])
                    idxs_all.append([])
                    continue

                candidate_idx = [doc_id_to_idx[doc_id] for doc_id in ranked_ids]
                candidate_embs = self.sub_corpus_embeddings[candidate_idx]

                scores = similarity_fn(
                    torch.as_tensor(query_emb).unsqueeze(0),
                    torch.as_tensor(candidate_embs),
                )

                values, indices = torch.topk(
                    torch.as_tensor(scores),
                    k=min(top_k, len(candidate_idx)),
                    dim=1,
                    largest=True,
                )
                scores_all.append(values.squeeze(0).cpu().tolist())
                idxs_all.append(indices.squeeze(0).cpu().tolist())

            return scores_all, idxs_all

        scores = similarity_fn(embeddings, self.sub_corpus_embeddings)
        self.sub_corpus_embeddings = None

        cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(
            torch.tensor(scores),
            min(
                top_k + 1,
                len(scores[1]) if len(scores) > 1 else len(scores[-1]),
            ),
            dim=1,
            largest=True,
        )
        return (
            cos_scores_top_k_values.cpu().tolist(),
            cos_scores_top_k_idx.cpu().tolist(),
        )

    def clear(self) -> None:
        """Clear all stored documents and embeddings from the backend."""
        self.sub_corpus_embeddings = None
        self.idxs = []
