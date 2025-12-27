import logging
import warnings
from collections.abc import Callable

import numpy as np
import torch

from mteb._requires_package import requires_package
from mteb.models.model_meta import ScoringFunction
from mteb.models.models_protocols import EncoderProtocol
from mteb.types import Array, TopRankedDocumentsType

logger = logging.getLogger(__name__)


class FaissSearchIndex:
    """FAISS-based backend for encoder-based search.

    Supports both full-corpus retrieval and reranking (via `top_ranked`).

    Notes:
        - Stores *all* embeddings in memory (IndexFlatIP or IndexFlatL2).
        - Expects embeddings to be normalized if cosine similarity is desired.
    """

    _normalize: bool = False

    def __init__(self, model: EncoderProtocol) -> None:
        requires_package(
            self,
            "faiss",
            "FAISS-based search",
            install_instruction="pip install mteb[faiss-cpu]",
        )

        import faiss
        from faiss import IndexFlatIP, IndexFlatL2

        # https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
        if model.mteb_model_meta.similarity_fn_name is ScoringFunction.DOT_PRODUCT:
            self.index_type = IndexFlatIP
        elif model.mteb_model_meta.similarity_fn_name is ScoringFunction.COSINE:
            self.index_type = IndexFlatIP
            self._normalize = True
        elif model.mteb_model_meta.similarity_fn_name is ScoringFunction.EUCLIDEAN:
            self.index_type = IndexFlatL2
        else:
            raise ValueError(
                f"FAISS backend does not support similarity function {model.mteb_model_meta.similarity_fn_name}. "
                f"Available: {ScoringFunction.DOT_PRODUCT}, {ScoringFunction.COSINE}."
            )

        self.idxs: list[str] = []
        self.index: faiss.Index | None = None

    def add_documents(self, embeddings: Array, idxs: list[str]) -> None:
        """Add all document embeddings and their IDs to FAISS index."""
        import faiss

        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()

        embeddings = embeddings.astype(np.float32)
        self.idxs.extend(idxs)

        if self._normalize:
            faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        if self.index is None:
            self.index = self.index_type(dim)

        self.index.add(embeddings)
        logger.info(f"FAISS index built with {len(idxs)} vectors of dim {dim}.")

    def search(
        self,
        embeddings: Array,
        top_k: int,
        similarity_fn: Callable[[Array, Array], Array],
        top_ranked: TopRankedDocumentsType | None = None,
        query_idx_to_id: dict[int, str] | None = None,
    ) -> tuple[list[list[float]], list[list[int]]]:
        """Search using FAISS."""
        import faiss

        if self.index is None:
            raise ValueError("No index built. Call add_document() first.")

        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()

        if self._normalize:
            faiss.normalize_L2(embeddings)

        if top_ranked is not None:
            if query_idx_to_id is None:
                raise ValueError("query_idx_to_id must be provided when reranking.")

            similarities, ids = self._reranking(
                embeddings,
                top_k,
                top_ranked=top_ranked,
                query_idx_to_id=query_idx_to_id,
            )
        else:
            similarities, ids = self.index.search(embeddings.astype(np.float32), top_k)
            similarities = similarities.tolist()
            ids = ids.tolist()

        if issubclass(self.index_type, faiss.IndexFlatL2):
            similarities = (-np.sqrt(np.maximum(similarities, 0))).tolist()

        return similarities, ids

    def _reranking(
        self,
        embeddings: Array,
        top_k: int,
        top_ranked: TopRankedDocumentsType,
        query_idx_to_id: dict[int, str],
    ) -> tuple[list[list[float]], list[list[int]]]:
        doc_id_to_idx = {doc_id: i for i, doc_id in enumerate(self.idxs)}
        scores_all: list[list[float]] = []
        idxs_all: list[list[int]] = []

        for query_idx, query_emb in enumerate(embeddings):
            query_id = query_idx_to_id[query_idx]
            ranked_ids = top_ranked.get(query_id)
            if not ranked_ids:
                msg = f"No top-ranked documents for query {query_id}"
                logger.warning(msg)
                warnings.warn(msg)
                scores_all.append([])
                idxs_all.append([])
                continue

            candidate_indices = [doc_id_to_idx[doc_id] for doc_id in ranked_ids]
            d = self.index.d  # type: ignore[union-attr]
            candidate_embs = np.vstack(
                [self.index.reconstruct(idx) for idx in candidate_indices]  # type: ignore[union-attr]
            )
            sub_reranking_index = self.index_type(d)
            sub_reranking_index.add(candidate_embs)

            # Search returns scores and indices in one call
            scores, local_indices = sub_reranking_index.search(
                query_emb.reshape(1, -1).astype(np.float32),
                min(top_k, len(candidate_indices)),
            )
            # faiss will output 2d arrays even for single query
            scores_all.append(scores[0].tolist())
            idxs_all.append(local_indices[0].tolist())

        return scores_all, idxs_all

    def clear(self) -> None:
        """Clear all stored documents and embeddings from the backend."""
        self.index = None
        self.idxs = []
